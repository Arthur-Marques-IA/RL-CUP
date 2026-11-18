"""Egocentric features, batched in torch over cuju's raw observations.

MUST stay identical to `encode` in submission_template/agent.py (numpy, used in the
tournament). tests/test_features.py checks both on the same observations.

Layout (87 floats):
  ball xyz (3) | ball dir xyz *20 (3) | ball rel. to me (2) | ball dist (1)
  ownership one-hot none/us/them (3) | I have the ball (1) | game mode one-hot (7)
  my role one-hot (10) | my pos (2) | my dir *100 (2) | my tiredness (1) | sticky actions (10)
  4 teammates (index order, me excluded): rel. pos (2) + dir *100 (2)  -> 16
  5 opponents: rel. pos (2) + dir *100 (2)                              -> 20
  opponent goal rel. (2) | own goal rel. (2) | tanh(score diff / 2) (1) | steps_left / 3000 (1)
"""
from __future__ import annotations

import torch

N_PLAYERS = 5
PLAYER_DIR_SCALE = 100.0
BALL_DIR_SCALE = 20.0
MATCH_STEPS = 3000.0
FEATURE_DIM = 87

# For each active index, the other four indices in ascending order.
_MATES = torch.tensor([[j for j in range(N_PLAYERS) if j != i] for i in range(N_PLAYERS)])


def _gather_player(x, idx):
    """x [E, A, P, D], idx [E, A] -> [E, A, D]."""
    return torch.gather(x, 2, idx[..., None, None].expand(*idx.shape, 1, x.shape[-1])).squeeze(2)


def encode(raw: dict) -> torch.Tensor:
    """cuju raw dict of tensors [E, A, ...] (each agent's own point of view) -> [E, A, 87] float32."""
    lt = raw["left_team"].float()[:, :, :N_PLAYERS]
    ltd = raw["left_team_direction"].float()[:, :, :N_PLAYERS]
    rt = raw["right_team"].float()[:, :, :N_PLAYERS]
    rtd = raw["right_team_direction"].float()[:, :, :N_PLAYERS]
    a = raw["active"].long()
    me = _gather_player(lt, a)
    my_dir = _gather_player(ltd, a)
    ball = raw["ball"].float()
    ball_dir = raw["ball_direction"].float()
    owned_team = raw["ball_owned_team"].long()
    owned_player = raw["ball_owned_player"].long()

    ball_rel = ball[..., :2] - me
    ball_dist = torch.sqrt((ball_rel * ball_rel).sum(-1, keepdim=True))
    own = torch.nn.functional.one_hot(owned_team + 1, 3).float()
    i_have = ((owned_team == 0) & (owned_player == a)).float()[..., None]
    mode = torch.nn.functional.one_hot(raw["game_mode"].long(), 7).float()
    roles = raw["left_team_roles"].long()[:, :, :N_PLAYERS]
    role = torch.nn.functional.one_hot(torch.gather(roles, 2, a[..., None]).squeeze(2), 10).float()
    tired = torch.gather(raw["left_team_tired_factor"].float()[:, :, :N_PLAYERS], 2, a[..., None])
    sticky = raw["sticky_actions"].float()

    mates = _MATES.to(a.device)[a]                                   # [E, A, 4]
    m_idx = mates[..., None].expand(*mates.shape, 2)
    mate_pos = torch.gather(lt, 2, m_idx) - me[..., None, :]
    mate_dir = torch.gather(ltd, 2, m_idx) * PLAYER_DIR_SCALE
    mate_feat = torch.cat([mate_pos, mate_dir], -1).flatten(2)     # [E, A, 16]
    opp_feat = torch.cat([rt - me[..., None, :], rtd * PLAYER_DIR_SCALE], -1).flatten(2)  # [E, A, 20]
    goals = torch.stack([1.0 - me[..., 0], -me[..., 1], -1.0 - me[..., 0], -me[..., 1]], -1)
    score = raw["score"].float()
    match = torch.stack([torch.tanh((score[..., 0] - score[..., 1]) / 2.0),
                         raw["steps_left"].float() / MATCH_STEPS], -1)

    return torch.cat([
        ball, ball_dir * BALL_DIR_SCALE, ball_rel, ball_dist, own, i_have, mode, role,
        me, my_dir * PLAYER_DIR_SCALE, tired, sticky, mate_feat, opp_feat, goals, match,
    ], -1)


def raw_to_dicts(raw: dict, env: int, agents: int, n_left: int, n_right: int) -> list[dict]:
    """One environment's raw tensors -> list of GRF-style observation dicts (numpy / python
    scalars), one per agent: the format the arena passes to Agent.act. Used for probes and
    for the feature parity test."""
    out = []
    for k in range(agents):
        d = {}
        for key, val in raw.items():
            v = val[env, k].cpu().numpy().copy()  # cuju overwrites its buffers every step
            if key.startswith("left_team"):
                v = v[:n_left]
            elif key.startswith("right_team"):
                v = v[:n_right]
            d[key] = v.item() if v.ndim == 0 else v
        out.append(d)
    return out

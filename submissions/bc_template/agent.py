"""Submission agent: egocentric features + MLP policy, numpy only.

Everything needed to turn a raw observation into an action lives in this file
and in weights/policy.npz. The feature code MUST stay identical to
cup/features.py (the torch version used in training); tests/test_features.py
checks the two against each other.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

N_PLAYERS = 5          # 5_vs_5: keeper + 4 outfield per side
PLAYER_DIR_SCALE = 100.0
BALL_DIR_SCALE = 20.0
MATCH_STEPS = 3000.0
FEATURE_DIM = 87


def _f32(x):
    return np.asarray(x, dtype=np.float32)


def encode(obs: dict) -> np.ndarray:
    """One raw gfootball observation (our point of view) -> float32 vector (FEATURE_DIM,)."""
    lt = _f32(obs["left_team"])[:N_PLAYERS]
    ltd = _f32(obs["left_team_direction"])[:N_PLAYERS]
    rt = _f32(obs["right_team"])[:N_PLAYERS]
    rtd = _f32(obs["right_team_direction"])[:N_PLAYERS]
    a = int(obs["active"])
    me = lt[a]
    ball = _f32(obs["ball"])
    ball_dir = _f32(obs["ball_direction"])
    owned_team = int(obs["ball_owned_team"])
    owned_player = int(obs["ball_owned_player"])

    ball_rel = ball[:2] - me
    ball_dist = np.sqrt(np.sum(ball_rel * ball_rel, dtype=np.float32), dtype=np.float32)
    own = np.zeros(3, np.float32)
    own[owned_team + 1] = 1.0
    i_have = np.float32(owned_team == 0 and owned_player == a)
    mode = np.zeros(7, np.float32)
    mode[int(obs["game_mode"])] = 1.0
    role = np.zeros(10, np.float32)
    role[int(obs["left_team_roles"][a])] = 1.0
    tired = _f32(obs["left_team_tired_factor"])[a]
    sticky = _f32(obs["sticky_actions"])

    mates = [i for i in range(N_PLAYERS) if i != a]
    mate_feat = np.concatenate([lt[mates] - me, ltd[mates] * PLAYER_DIR_SCALE], axis=1).reshape(-1)
    opp_feat = np.concatenate([rt - me, rtd * PLAYER_DIR_SCALE], axis=1).reshape(-1)
    goals = np.array([1.0 - me[0], -me[1], -1.0 - me[0], -me[1]], np.float32)
    score = obs["score"]
    match = np.array([np.tanh((float(score[0]) - float(score[1])) / 2.0),
                      float(obs["steps_left"]) / MATCH_STEPS], np.float32)

    return np.concatenate([
        ball, ball_dir * BALL_DIR_SCALE, ball_rel, [ball_dist], own, [i_have], mode, role,
        me, ltd[a] * PLAYER_DIR_SCALE, [tired], sticky, mate_feat, opp_feat, goals, match,
    ]).astype(np.float32)


class MLPPolicy:
    """Actor MLP from weights/policy.npz: keys w0,b0,w1,b1,... ; tanh hidden layers."""

    def __init__(self, path: Path):
        data = np.load(path)
        n = len([k for k in data.files if k.startswith("w")])
        self.layers = [(data[f"w{i}"].astype(np.float32), data[f"b{i}"].astype(np.float32))
                       for i in range(n)]
        self.mean = data["obs_mean"].astype(np.float32) if "obs_mean" in data.files else None
        self.std = data["obs_std"].astype(np.float32) if "obs_std" in data.files else None

    def logits(self, x: np.ndarray) -> np.ndarray:
        if self.mean is not None:
            x = (x - self.mean) / self.std
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return x


class Agent:
    def __init__(self, path):
        self.path = Path(path)
        self.policy = MLPPolicy(self.path / "weights" / "policy.npz")

    def reset(self) -> None:
        pass

    def act(self, observations: list[dict]) -> list[int]:
        x = np.stack([encode(o) for o in observations])
        return [int(a) for a in np.argmax(self.policy.logits(x), axis=1)]

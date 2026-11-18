"""Turn a trained ActorCritic into an arena submission folder, with a probe.

The probe is recorded on the *training* side: cuju raw observations are converted to the
arena's dict format, the torch features + torch network pick the actions. arena.check then
replays the dicts through the submitted numpy agent; a mismatch means the two feature
implementations (or the weights) differ.
"""
from __future__ import annotations

import pickle
import shutil
from pathlib import Path

import torch

import cuju
from cup import features
from cup.model import ActorCritic

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "submission_template" / "agent.py"
PROBE_FORMAT = 1  # arena/probe.py


def record_probe(model: ActorCritic, path: Path, steps: int = 300, envs: int = 2, seed: int = 0) -> int:
    """Probe in arena/probe.py's format, controlled_players=4, one episode per env."""
    episodes = [[] for _ in range(envs)]
    model.eval()
    with cuju.BatchEnvironment(features_scenario(), environment_count=envs, device="cpu",
                               representation="raw", rewards="scoring", seed=seed) as batch:
        n_left = int(batch.left_players[0])
        n_right = int(batch.right_players[0])
        raw = batch.reset()
        for _ in range(steps):
            with torch.no_grad():
                actions = model.logits(features.encode(raw)).argmax(-1)      # [E, 4]
            for e in range(envs):
                obs = features.raw_to_dicts(raw, e, 4, n_left, n_right)
                episodes[e].append({"obs": obs, "actions": [int(a) for a in actions[e]]})
            raw, *_ = batch.step(actions.to(torch.int32))
    with open(path, "wb") as f:
        pickle.dump({"format": PROBE_FORMAT, "controlled_players": 4, "episodes": episodes},
                    f, protocol=pickle.HIGHEST_PROTOCOL)
    return steps * envs * 4


def features_scenario() -> str:
    return "5_vs_5_l4_r0"


def export_submission(model: ActorCritic, out: Path, team: str, description: str,
                      members: list[str] | None = None) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(TEMPLATE, out / "agent.py")
    model.export_npz(out / "weights" / "policy.npz")
    members = members or []
    (out / "manifest.yml").write_text(
        f"team: {team}\n"
        f"members: {members!r}\n"
        f"controlled_players: 4\n"
        f"deterministic: true\n"
        f"description: >\n  {description}\n", encoding="utf-8")
    n = record_probe(model, out / "probe.pkl")
    print(f"submission written to {out} (probe: {n} actions)")
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="export a model.pt to a submission folder")
    p.add_argument("model", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--team", default="ufg-rl-cup")
    p.add_argument("--description", default="egocentric MLP policy")
    a = p.parse_args()
    st = torch.load(a.model, map_location="cpu", weights_only=False)
    sd = st.get("model", st)           # PPO checkpoint (latest.pt) or BC state_dict (model.pt)
    m = ActorCritic(hidden=sd["actor.0.weight"].shape[0])
    m.load_state_dict(sd)
    export_submission(m, a.out, team=a.team, description=a.description)

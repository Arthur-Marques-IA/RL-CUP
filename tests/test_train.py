"""Config / shaping / evaluation sanity checks (fast, CPU)."""
import torch

from cup.config import TrainCfg, drift, load_config, to_dict
from cup.rewards import PotentialShaping


def test_configs_load():
    from pathlib import Path
    for p in Path(__file__).resolve().parent.parent.glob("configs/*.yaml"):
        cfg = load_config(p)
        assert isinstance(cfg, TrainCfg) and cfg.line


def test_drift_ignores_machine_settings():
    a = to_dict(TrainCfg())
    b = to_dict(TrainCfg())
    b["env"]["num_envs"] = 4096
    b["env"]["device"] = "cuda"
    assert drift(a, b) == []
    b["ppo"]["gamma"] = 0.99
    assert drift(a, b) == ["ppo.gamma"]


def _raw(ball_x, owned):
    E = len(ball_x)
    ball = torch.zeros(E, 1, 3)
    ball[:, 0, 0] = torch.tensor(ball_x)
    return {"ball": ball, "ball_owned_team": torch.tensor(owned).reshape(E, 1)}


def test_potential_shaping_telescopes():
    """Sum of shaping over a match (gamma=1) = Phi(terminal)=0 - Phi(s0)."""
    sh = PotentialShaping(1, alpha=0.2, beta=0.05, gamma=1.0, device="cpu")
    sh.reset(_raw([0.0], [-1]))
    total = 0.0
    traj = [(0.3, 0), (0.5, -1), (0.6, 1), (0.1, 1), (0.8, 0)]
    for x, o in traj:
        total += float(sh.step(_raw([x], [o]), torch.tensor([False])))
    total += float(sh.step(_raw([0.0], [-1]), torch.tensor([True])))   # match ends
    assert abs(total - 0.0) < 1e-6   # Phi(s0) = 0 here
    # last touch: a loose ball keeps the previous owner
    sh.reset(_raw([0.0], [0]))
    sh.step(_raw([0.1], [-1]), torch.tensor([False]))
    assert float(sh.last_touch[0]) == 1.0

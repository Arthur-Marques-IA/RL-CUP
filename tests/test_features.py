"""Training-side (torch, cuju raw) and submission-side (numpy, arena dicts) features must agree."""
import importlib.util
from pathlib import Path

import numpy as np
import torch

import cuju
from cup import features
from cup.teachers import ScriptedTemplate

ROOT = Path(__file__).resolve().parent.parent


def _load_agent_module():
    spec = importlib.util.spec_from_file_location("submission_agent", ROOT / "submission_template" / "agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_torch_and_numpy_features_match():
    agent = _load_agent_module()
    assert agent.FEATURE_DIM == features.FEATURE_DIM
    teacher = ScriptedTemplate()
    with cuju.BatchEnvironment("5_vs_5_l4_r0", environment_count=4, device="cpu",
                               representation="raw", rewards="scoring", seed=3) as batch:
        n_left, n_right = int(batch.left_players[0]), int(batch.right_players[0])
        raw = batch.reset()
        worst = 0.0
        for step in range(400):
            x = features.encode(raw)
            assert x.shape == (4, 4, features.FEATURE_DIM)
            if step % 7 == 0:
                for e in range(4):
                    for k, obs in enumerate(features.raw_to_dicts(raw, e, 4, n_left, n_right)):
                        ref = agent.encode(obs)
                        worst = max(worst, float(np.abs(ref - x[e, k].numpy()).max()))
            raw, *_ = batch.step(teacher(raw).to(torch.int32))
    assert worst < 1e-4, worst


def test_active_minus_one_is_handled_identically():
    """A red card can leave an agent without a player (active = -1): no crash, same features."""
    agent = _load_agent_module()
    with cuju.BatchEnvironment("5_vs_5_l4_r0", environment_count=1, device="cpu",
                               representation="raw", rewards="scoring", seed=3) as batch:
        raw = batch.reset()
        raw = {k: v.clone() for k, v in raw.items()}
        raw["active"][0, 2] = -1
        x = features.encode(raw)
        obs = features.raw_to_dicts(raw, 0, 4, 5, 5)
        ref = agent.encode(obs[2])
        assert np.abs(ref - x[0, 2].numpy()).max() < 1e-5
        assert ScriptedTemplate()(raw).shape == (1, 4)


def test_template_teacher_matches_numpy_template():
    """The torch teacher reproduces t-zero's arena/template/agent.py decisions."""
    import pytest
    template_path = ROOT.parent / "ref" / "t-zero-learning" / "arena" / "template" / "agent.py"
    if not template_path.is_file():
        pytest.skip("t-zero-learning checkout not found next to the repo (../ref/t-zero-learning)")
    spec = importlib.util.spec_from_file_location("tz_template", template_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ref_agent = mod.Agent(template_path.parent)
    teacher = ScriptedTemplate()
    with cuju.BatchEnvironment("5_vs_5_l4_r0", environment_count=2, device="cpu",
                               representation="raw", rewards="scoring", seed=5) as batch:
        n_left, n_right = int(batch.left_players[0]), int(batch.right_players[0])
        raw = batch.reset()
        agree = total = 0
        for _ in range(300):
            a = teacher(raw)
            for e in range(2):
                ref = ref_agent.act(features.raw_to_dicts(raw, e, 4, n_left, n_right))
                agree += sum(int(r == int(t)) for r, t in zip(ref, a[e]))
                total += 4
            raw, *_ = batch.step(a.to(torch.int32))
    assert agree / total > 0.99, agree / total

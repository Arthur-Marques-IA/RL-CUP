"""GRF_MARL 5v5 checkpoints (github.com/jidiai/GRF_MARL, Apache-2.0) as teachers / opponents.

    python -m cup.grf_marl extract <GRF_MARL clone>   # once: pickled modules -> plain state_dicts
    python -m cup.grf_marl measure --matches 32       # each checkpoint vs builtin at 0.05/0.05

The original actor.pt files are whole pickled light_malib modules. `extract` reads them with an
unpickler that maps every light_malib class to an empty nn.Module, so no light_malib code runs;
only the tensors are kept (third_party/grf_marl/weights/<name>.pt).

Their policy: features (encoder_basic: 133 dims, or the "enhanced" encoder: 192 dims) whose first
19 entries are the legal-action mask -> LayerNorm (partial for 192) -> MLP (ReLU) -> 19 logits ->
argmax over legal actions. The encoders and State are vendored unmodified in third_party/grf_marl.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "third_party" / "grf_marl" / "weights"
CHECKPOINTS = ["3-1_LongPass", "3-1_formation", "defense_v3", "GKBug_v0", "GKBug_v1", "GKBug_v2",
               "GKBug_v3", "PassingMain_v1", "PassingMain_v2"]
N_ACTIONS = 19


# --------------------------------------------------------------------------- extraction

class _Stub(nn.Module):
    """Stands in for any light_malib class; nn.Module.__setstate__ restores params/submodules."""
    def __init__(self, *a, **k):
        super().__init__()


def _stub_unpickler_module():
    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("light_malib"):
                return type(name, (_Stub,), {})
            return super().find_class(module, name)
    mod = types.ModuleType("stub_pickle")
    mod.Unpickler = Unpickler
    mod.load = lambda f, **kw: Unpickler(f, **kw).load()
    mod.__name__ = "pickle"
    return mod


def extract(clone: Path) -> None:
    src = clone / "light_malib" / "trained_models" / "gr_football" / "5_vs_5"
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    for name in CHECKPOINTS:
        module = torch.load(src / name / "actor.pt", map_location="cpu", weights_only=False,
                            pickle_module=_stub_unpickler_module())
        sd = {k: v.clone() for k, v in module.state_dict().items()}
        torch.save(sd, WEIGHTS / f"{name}.pt")
        print(name, {k: tuple(v.shape) for k, v in sd.items()})


# --------------------------------------------------------------------------- network

class TeacherNet(nn.Module):
    """Rebuilds their actor from a state_dict: feature norm -> Linear/ReLU stack -> out."""

    def __init__(self, sd: dict):
        super().__init__()
        self.enhanced = "base._feature_norm.layer.weight" in sd
        if self.enhanced:
            d1 = sd["base._feature_norm.layer.weight"].shape[0]
            d2 = sd["base._feature_norm.layer2.weight"].shape[0]
            self.norm1, self.norm2 = nn.LayerNorm(d1), nn.LayerNorm(d2)
            self.norm1.load_state_dict({"weight": sd["base._feature_norm.layer.weight"],
                                        "bias": sd["base._feature_norm.layer.bias"]})
            self.norm2.load_state_dict({"weight": sd["base._feature_norm.layer2.weight"],
                                        "bias": sd["base._feature_norm.layer2.bias"]})
            self.split = d1
        else:
            d = sd["base._feature_norm.weight"].shape[0]
            self.norm1 = nn.LayerNorm(d)
            self.norm1.load_state_dict({"weight": sd["base._feature_norm.weight"],
                                        "bias": sd["base._feature_norm.bias"]})
        idx = sorted({int(k.split(".")[2]) for k in sd if k.startswith("base.net.")})
        layers = []
        for i in idx:
            w, b = sd[f"base.net.{i}.weight"], sd[f"base.net.{i}.bias"]
            lin = nn.Linear(w.shape[1], w.shape[0])
            lin.load_state_dict({"weight": w, "bias": b})
            layers += [lin, nn.ReLU()]
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(sd["out.weight"].shape[1], sd["out.weight"].shape[0])
        self.out.load_state_dict({"weight": sd["out.weight"], "bias": sd["out.bias"]})
        self.in_dim = (self.split + self.norm2.normalized_shape[0]) if self.enhanced else self.norm1.normalized_shape[0]

    def forward(self, x):
        if self.enhanced:
            x = torch.cat([self.norm1(x[..., :self.split]), self.norm2(x[..., self.split:])], -1)
        else:
            x = self.norm1(x)
        return self.out(self.net(x))

    @torch.no_grad()
    def act(self, feats: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(feats, dtype=torch.float32)
        logits = self(x) - 1e10 * (1 - x[..., :N_ACTIONS])
        return logits.argmax(-1).numpy()


# --------------------------------------------------------------------------- teacher over cuju raw

def _import_vendored():
    sys.path.insert(0, str(ROOT))
    from third_party.grf_marl import encoder_basic, encoder_enhanced5, state
    return encoder_basic, encoder_enhanced5, state


class GRFMarlTeacher:
    """Callable over cuju raw tensors [E, A, ...] -> actions [E, A] (CPU). Keeps their per-player
    State (needed for offside features). Call `reset(mask)` for environments that start a new
    episode (with auto_reset=False evaluations it is never needed)."""

    def __init__(self, name: str):
        self.name = name
        sd = torch.load(WEIGHTS / f"{name}.pt", map_location="cpu", weights_only=True)
        self.net = TeacherNet(sd).eval()
        enc_basic, enc_enh, state_mod = _import_vendored()
        # basic encoder: FE_cfg num_players=10 in their desc.pkl (5v5, both teams)
        self.encoder = enc_enh.FeatureEncoder() if self.net.enhanced else enc_basic.FeatureEncoder(num_players=10)
        self._State = state_mod.State
        self.states: list[list] | None = None
        self.last: np.ndarray | None = None

    def reset(self, mask=None):
        if self.states is None:
            return
        for e in range(len(self.states)):
            if mask is None or bool(mask[e]):
                self.states[e] = None

    def set_executed(self, actions) -> None:
        """Tell the teacher which actions were actually played this step (DAgger / masked student),
        so its State history matches the game."""
        self.last = np.asarray(actions.cpu() if hasattr(actions, "cpu") else actions)

    def __call__(self, raw) -> torch.Tensor:
        feats = self.encode(raw)
        actions = self.net.act(feats)
        self.last = actions
        return torch.as_tensor(actions, dtype=torch.long)

    def legal_mask(self, feats: np.ndarray) -> torch.Tensor:
        """Their legal-action mask (first 19 features) -> bool tensor [E, A, 19]."""
        return torch.as_tensor(feats[..., :N_ACTIONS] > 0.5)

    def encode(self, raw) -> np.ndarray:
        """Their features [E, A, D] for the current step (updates the per-player State)."""
        from cup.features import raw_to_dicts
        E, A = raw["active"].shape
        if self.states is None:
            self.states = [None] * E
        n_left = raw["left_team"].shape[2]
        n_right = raw["right_team"].shape[2]
        feats = []
        for e in range(E):
            obs = raw_to_dicts(raw, e, A, n_left, n_right)
            if self.states[e] is None:
                self.states[e] = [self._State(n_player=5) for _ in range(A)]
                for s, o in zip(self.states[e], obs):
                    s.update_obs(o)
            else:
                for k, (s, o) in enumerate(zip(self.states[e], obs)):
                    s.update_action(int(self.last[e, k]))
                    s.update_obs(o)
                    # keep memory bounded: their encoders only look at the latest obs/actions
                    if len(s.obs_list) > 8:
                        del s.obs_list[:-4]
                        del s.action_list[:-4]
            feats.append(np.asarray(self.encoder.encode(self.states[e]), dtype=np.float32))
        return np.stack(feats)


class MaskedStudent:
    """Our student (egocentric MLP) with the GRF_MARL teacher's legal-action mask applied to its
    logits: a diagnostic of how much the missing mask explains the student-teacher gap."""

    def __init__(self, model, teacher: GRFMarlTeacher):
        self.model, self.teacher = model, teacher

    @torch.no_grad()
    def __call__(self, raw) -> torch.Tensor:
        from cup import features
        mask = self.teacher.legal_mask(self.teacher.encode(raw))
        logits = self.model.logits(features.encode(raw))
        a = logits.masked_fill(~mask, -1e9).argmax(-1)
        self.teacher.set_executed(a)
        return a


# --------------------------------------------------------------------------- measurement

def measure(names: list[str], matches: int, seed: int, out: Path) -> None:
    from cup.evaluate import play
    results = {}
    for name in names:
        res = play(GRFMarlTeacher(name), None, matches, seed)
        results[name] = res
        print(f"{name:16s} W/D/L {res['W']:2d}/{res['D']:2d}/{res['L']:2d}  goals {res['goals_for']:3d}-"
              f"{res['goals_against']:3d}  diff/match {res['goal_diff_per_match']:+.2f}  "
              f"pts/match {res['points_per_match']:.2f}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("clone", type=Path)
    m = sub.add_parser("measure")
    m.add_argument("--names", nargs="*", default=CHECKPOINTS)
    m.add_argument("--matches", type=int, default=32)
    m.add_argument("--seed", type=int, default=1000)
    m.add_argument("--out", type=Path, default=ROOT / "runs" / "grf_marl_vs_builtin.json")
    a = p.parse_args()
    if a.cmd == "extract":
        extract(a.clone)
    else:
        measure(a.names, a.matches, a.seed, a.out)


if __name__ == "__main__":
    main()

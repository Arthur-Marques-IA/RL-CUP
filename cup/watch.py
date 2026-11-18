"""Watch matches in the browser (3D, cuju-render).

    python -m cup.watch --line ppo_bc_gkbug              # latest recorded eval match of a line (after git pull)
    python -m cup.watch --checkpoint checkpoints/ppo_bc_gkbug/latest.pt            # live vs builtin
    python -m cup.watch --checkpoint A.pt --opponent checkpoints/x/pool/snap_...pt # live, A vs B
    python -m cup.watch --checkpoint A.pt --opponent grf:GKBug_v1                 # live vs a GRF_MARL policy

Needs `pip install "cuju[render]"` (in requirements.txt). The page opens in the default browser.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

import cuju
from cup import features
from cup.checkpoint import PUBLISHED, RUNS, safe_load
from cup.model import ActorCritic


def _policy(spec: str):
    if spec.startswith("grf:"):
        from cup.grf_marl import GRFMarlTeacher
        return GRFMarlTeacher(spec[4:])
    st = safe_load(Path(spec))
    m = ActorCritic()
    m.load_state_dict(st["model"] if "model" in st else st, strict=False)
    m.eval()
    return lambda raw: m.logits(features.encode(raw)).argmax(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--line")
    p.add_argument("--checkpoint")
    p.add_argument("--opponent", default="builtin")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--matches", type=int, default=1)
    a = p.parse_args()
    import cuju.render as render

    if a.line:
        for base in (RUNS, PUBLISHED):
            f = base / a.line / "latest_match.cjr"
            if f.is_file():
                print(f"opening {f}")
                render.show(str(f), display="browser", stay=True)
                return
        raise SystemExit(f"no latest_match.cjr for line {a.line!r} (has it run an evaluation and been published?)")

    if not a.checkpoint:
        raise SystemExit("--line or --checkpoint is required")
    left = _policy(a.checkpoint)
    right = None if a.opponent == "builtin" else _policy(a.opponent)
    scenario = "5_vs_5_l4_r0" if right is None else "5_vs_5_l4_r4"
    with cuju.BatchEnvironment(scenario, environment_count=1, device="cpu", representation="raw",
                               rewards="scoring", seed=a.seed) as batch:
        names = [{"name": Path(a.checkpoint).parent.name or "left"}, {"name": a.opponent}]
        render.show(batch, title=f"{names[0]['name']} vs {a.opponent}", teams=names, display="browser")
        raw = batch.reset()
        played = 0
        with torch.no_grad():
            while played < a.matches:
                if right is None:
                    act = left(raw)
                else:
                    act = torch.cat([left({k: v[:, :4] for k, v in raw.items()}),
                                     right({k: v[:, 4:] for k, v in raw.items()})], 1)
                raw, _, done, _ = batch.step(act.to(torch.int32))
                played += int(done.sum())
                time.sleep(0.1)  # ~real time (10 steps = 1 s of game)
    print("match over")


if __name__ == "__main__":
    main()

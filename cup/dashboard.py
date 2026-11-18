"""Team dashboard: every line's curves (published by any member + your local runs) in TensorBoard.

    git pull
    python -m cup.dashboard            # rebuilds runs/_dashboard from metrics.jsonl/evals.jsonl
    tensorboard --logdir runs/_dashboard

Each member's training writes metrics.jsonl/evals.jsonl; `cup.checkpoint publish` commits them.
This script turns all of them into TensorBoard event files, so everyone sees all lines without a
cloud service. For a single local run, `tensorboard --logdir runs/<line>/tb` is live.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from cup.checkpoint import PUBLISHED, RUNS


def _rows(path: Path):
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=RUNS / "_dashboard")
    a = p.parse_args()
    from torch.utils.tensorboard import SummaryWriter

    if a.out.exists():
        shutil.rmtree(a.out)
    lines: dict[str, Path] = {}
    for base in (PUBLISHED, RUNS):              # local runs override published ones (newer)
        for d in base.iterdir() if base.is_dir() else []:
            if d.is_dir() and not d.name.startswith("_") and (d / "metrics.jsonl").is_file():
                lines[d.name] = d
    for name, d in sorted(lines.items()):
        w = SummaryWriter(str(a.out / name))
        n = 0
        for prefix, file in (("", "metrics.jsonl"), ("eval/", "evals.jsonl")):
            for r in _rows(d / file):
                step = r.get("env_steps")
                for k, v in r.items():
                    if k in ("env_steps", "time") or not isinstance(v, (int, float)) or not math.isfinite(v):
                        continue
                    w.add_scalar(prefix + k, v, step)
                    n += 1
        w.close()
        print(f"{name:24s} {n:>8} points  ({d})")
    print(f"\ntensorboard --logdir {a.out}")


if __name__ == "__main__":
    main()

"""Local stand-in for arena.check's probe + speed checks (no Docker needed).

    python -m cup.local_check submissions/bc_template

Loads agent.py from the folder in a fresh module (as the arena host does), limits torch/numpy
to one thread, replays probe.pkl and times act(). The official check (Docker image, real
GRF engine, full matches) is still required before submitting.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import importlib.util
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("submission", type=Path)
    args = p.parse_args()
    folder = args.submission.resolve()

    t0 = time.perf_counter()
    spec = importlib.util.spec_from_file_location("agent", folder / "agent.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agent"] = mod
    spec.loader.exec_module(mod)
    agent = mod.Agent(folder)
    load_s = time.perf_counter() - t0

    with open(folder / "probe.pkl", "rb") as f:
        probe = pickle.load(f)
    matched = total = 0
    times = []
    actions = Counter()
    for episode in probe["episodes"]:
        agent.reset()
        for step in episode:
            t = time.perf_counter()
            got = agent.act(step["obs"])
            times.append(time.perf_counter() - t)
            assert len(got) == len(step["obs"]) and all(0 <= int(g) < 19 for g in got)
            matched += sum(int(g == e) for g, e in zip(got, step["actions"]))
            total += len(step["actions"])
            actions.update(int(g) for g in got)
    size_mb = sum(q.stat().st_size for q in folder.rglob("*") if q.is_file()) / 2**20
    rate = matched / total
    top, cnt = actions.most_common(1)[0]
    print(f"load {load_s:.2f} s | size {size_mb:.2f} MB")
    print(f"probe {matched}/{total} reproduced ({rate:.1%}) -> {'PASS' if rate >= 0.9 else 'FAIL'}")
    print(f"act() mean {1000 * np.mean(times[1:]):.2f} ms, max {1000 * np.max(times):.1f} ms (limit mean 20 ms)")
    print(f"most common action {top}: {cnt / sum(actions.values()):.0%} of probe actions")
    sys.exit(0 if rate >= 0.9 else 1)


if __name__ == "__main__":
    main()

"""Evaluate policies in cuju over full 5_vs_5 matches from kick-off (difficulty 0.05/0.05).

    python -m cup.evaluate --left runs/bc_template_s1/model.pt --right builtin --matches 32
    python -m cup.evaluate --left teacher:template --right builtin

A side is: a model.pt path (argmax), `teacher:<name>`, or (right only) `builtin`.
Results go to stdout and, with --json, to a file (W/D/L, goals, expected points).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import cuju
from cup import features
from cup.model import ActorCritic
from cup.teachers import TEACHERS


def load_side(spec: str):
    if spec == "builtin":
        return None
    if spec.startswith("teacher:"):
        return TEACHERS[spec.split(":", 1)[1]]()
    model = ActorCritic()
    model.load_state_dict(torch.load(spec, map_location="cpu", weights_only=True))
    model.eval()

    def policy(raw):
        with torch.no_grad():
            return model.logits(features.encode(raw)).argmax(-1)
    return policy


def play(left, right, matches: int, seed: int, device: str = "cpu") -> dict:
    scenario = "5_vs_5_l4_r0" if right is None else "5_vs_5_l4_r4"
    with cuju.BatchEnvironment(scenario, environment_count=matches, device=device,
                               representation="raw", rewards="scoring", seed=seed,
                               auto_reset=False) as batch:
        raw = batch.reset()
        n_left = 4
        goals = torch.zeros(matches, 2, dtype=torch.int64)
        finished = torch.zeros(matches, dtype=torch.bool)
        while not finished.all():
            if right is None:
                act = left(raw)
            else:
                l_raw = {k: v[:, :n_left] for k, v in raw.items()}
                r_raw = {k: v[:, n_left:] for k, v in raw.items()}
                act = torch.cat([left(l_raw), right(r_raw)], 1)
            raw, reward, done, info = batch.step(act.to(torch.int32))
            sr = info["score_reward"].long()
            live = ~finished
            goals[:, 0] += ((sr == 1) & live).long()
            goals[:, 1] += ((sr == -1) & live).long()
            finished |= done.bool()
    gf, ga = goals[:, 0], goals[:, 1]
    w, d, l = int((gf > ga).sum()), int((gf == ga).sum()), int((gf < ga).sum())
    return {"matches": matches, "W": w, "D": d, "L": l,
            "goals_for": int(gf.sum()), "goals_against": int(ga.sum()),
            "goal_diff_per_match": float((gf - ga).float().mean()),
            "points_per_match": (3 * w + d) / matches}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--left", required=True)
    p.add_argument("--right", default="builtin")
    p.add_argument("--matches", type=int, default=32)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    res = play(load_side(args.left), load_side(args.right), args.matches, args.seed)
    res.update(left=args.left, right=args.right, seed=args.seed)
    print(json.dumps(res, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

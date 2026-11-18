"""Behaviour cloning: collect teacher demonstrations in cuju, train the student, export.

    # one shot (fast teachers, e.g. the scripted template)
    python -m cup.bc run --teacher template --envs 32 --steps 3001 --out runs/bc_template

    # slow teachers (GRF_MARL encoders are pure-Python): collect shards in parallel, then train
    python -m cup.bc collect --teacher grf:GKBug_v1 --envs 12 --seed 1 --out data/gkbug_v1/shard1.pt
    python -m cup.bc train --data data/gkbug_v1 --out runs/bc_gkbug_v1 --publish bc_gkbug_v1

Collection plays `5_vs_5_l4_r0` (opponent = built-in AI, difficulty 0.05/0.05, as in the
tournament). With probability --noise an agent executes a random *movement* action instead,
but the label is always the teacher's. Default 0: with the template teacher, 10% noise turned a
+1.03 goal-diff teacher into a -1.09 student (rare SHOT labels vanish).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

import cuju
from cup import features
from cup.model import N_ACTIONS, ActorCritic
from cup.teachers import make_teacher

SCENARIO = "5_vs_5_l4_r0"


def collect(teacher, steps: int, envs: int, noise: float, seed: int, device: str = "cpu"):
    """Returns features [N, 87] (float16 to save space), labels [N] (uint8) and stats."""
    gen = torch.Generator().manual_seed(seed)
    feats, labels = [], []
    goals_for = goals_against = 0
    with cuju.BatchEnvironment(SCENARIO, environment_count=envs, device=device,
                               representation="raw", rewards="scoring", seed=seed,
                               auto_reset=False) as batch:
        raw = batch.reset()
        for _ in range(steps):
            x = features.encode(raw)
            a = teacher(raw)
            feats.append(x.reshape(-1, x.shape[-1]).cpu().half())
            labels.append(a.reshape(-1).cpu().to(torch.uint8))
            act = a.clone()
            if noise > 0:
                flip = torch.rand(a.shape, generator=gen) < noise
                act[flip] = torch.randint(1, 9, (int(flip.sum()),), generator=gen)
            raw, reward, done, info = batch.step(act.to(torch.int32))
            sr = info["score_reward"]
            goals_for += int((sr == 1).sum())
            goals_against += int((sr == -1).sum())
            if bool(done.all()):
                break
    return torch.cat(feats), torch.cat(labels), {
        "matches": envs, "goals_for": goals_for, "goals_against": goals_against}


def train(model, x, y, epochs: int, batch_size: int, lr: float, log_path: Path, seed: int,
          class_weight_power: float = 0.0):
    """Cross-entropy BC. class_weight_power > 0 up-weights rare actions (w ∝ freq^-power)."""
    gen = torch.Generator().manual_seed(seed)
    n = len(x)
    perm = torch.randperm(n, generator=gen)
    n_val = max(n // 20, 1)
    val, tr = perm[:n_val], perm[n_val:]
    model.set_obs_stats(x[tr])
    freq = torch.bincount(y[tr], minlength=N_ACTIONS).float().clamp_min(1)
    weight = (freq / freq.sum()) ** (-class_weight_power) if class_weight_power > 0 else None
    if weight is not None:
        weight = weight / (weight * freq / freq.sum()).sum()  # keep the mean loss scale
    opt = torch.optim.Adam(model.actor.parameters(), lr=lr)
    acc = 0.0
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_acc_shot"])
        for ep in range(epochs):
            model.train()
            order = tr[torch.randperm(len(tr), generator=gen)]
            total = 0.0
            for i in range(0, len(order), batch_size):
                idx = order[i:i + batch_size]
                loss = torch.nn.functional.cross_entropy(model.logits(x[idx]), y[idx], weight=weight)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(idx)
            model.eval()
            with torch.no_grad():
                logits = model.logits(x[val])
                vl = float(torch.nn.functional.cross_entropy(logits, y[val]))
                pred = logits.argmax(-1)
                acc = float((pred == y[val]).float().mean())
                shot = y[val] == 12
                acc_shot = float((pred[shot] == 12).float().mean()) if shot.any() else float("nan")
            w.writerow([ep, total / len(tr), vl, acc, acc_shot])
            print(f"epoch {ep:3d}  train {total / len(tr):.4f}  val {vl:.4f}  acc {acc:.2%}  "
                  f"shot recall {acc_shot:.2%}", flush=True)
    return acc


def _train_and_save(x, y, args, meta: dict):
    model = ActorCritic(hidden=args.hidden)
    acc = train(model, x, y, args.epochs, args.batch_size, args.lr, args.out / "log.csv", args.seed,
                args.class_weight_power)
    torch.save(model.state_dict(), args.out / "model.pt")
    print(f"val accuracy {acc:.2%}; saved {args.out / 'model.pt'}")
    if args.publish:
        from cup.checkpoint import publish_bc
        publish_bc(model, args.publish, meta={**meta, "val_accuracy": acc})
    if args.submission:
        from cup.export import export_submission
        export_submission(model, args.submission, team=args.team, description=meta.get("description", "BC"))
    return model


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def common_train(sp):
        sp.add_argument("--epochs", type=int, default=20)
        sp.add_argument("--hidden", type=int, default=256, help="must match ppo.hidden of the PPO config")
        sp.add_argument("--batch-size", type=int, default=4096)
        sp.add_argument("--lr", type=float, default=1e-3)
        sp.add_argument("--class-weight-power", type=float, default=0.0)
        sp.add_argument("--seed", type=int, default=1)
        sp.add_argument("--out", type=Path, required=True)
        sp.add_argument("--publish", default=None, help="publish as checkpoints/<name>/latest.pt (PPO init)")
        sp.add_argument("--submission", type=Path, default=None)
        sp.add_argument("--team", default="ufg-rl-cup")

    def common_collect(sp):
        sp.add_argument("--teacher", default="template", help="template | grf:<checkpoint>")
        sp.add_argument("--steps", type=int, default=3001)
        sp.add_argument("--envs", type=int, default=32)
        sp.add_argument("--noise", type=float, default=0.0)

    r = sub.add_parser("run", help="collect + train in one process")
    common_collect(r)
    common_train(r)
    c = sub.add_parser("collect", help="collect one shard")
    common_collect(c)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("train", help="train on all shards in --data")
    t.add_argument("--data", type=Path, required=True)
    common_train(t)
    args = p.parse_args()

    if args.cmd == "collect":
        t0 = time.time()
        x, y, stats = collect(make_teacher(args.teacher), args.steps, args.envs, args.noise, args.seed)
        stats.update(teacher=args.teacher, seed=args.seed, samples=len(y), seconds=round(time.time() - t0, 1))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"x": x, "y": y, "stats": stats}, args.out)
        print(json.dumps(stats))
        return

    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    if args.cmd == "run":
        x, y, stats = collect(make_teacher(args.teacher), args.steps, args.envs, args.noise, args.seed)
        stats = [stats]
        teacher = args.teacher
    else:
        shards = [torch.load(f, weights_only=False) for f in sorted(args.data.glob("*.pt"))]
        x = torch.cat([s["x"] for s in shards])
        y = torch.cat([s["y"] for s in shards])
        stats = [s["stats"] for s in shards]
        teacher = stats[0].get("teacher", "?")
    x, y = x.float(), y.long()
    hist = torch.bincount(y, minlength=N_ACTIONS).tolist()
    summary = {"teacher": teacher, "samples": len(y), "label_histogram": hist, "shards": stats}
    (args.out / "data.json").write_text(json.dumps(summary, indent=2))
    print(f"{len(y):,} samples from {teacher}; label %: {[round(100 * h / len(y), 1) for h in hist]}")
    _train_and_save(x, y, args, {"teacher": teacher, "samples": len(y),
                                 "description": f"BC from {teacher} ({len(y):,} samples), egocentric MLP"})


if __name__ == "__main__":
    main()

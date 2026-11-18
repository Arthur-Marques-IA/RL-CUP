"""Checkpoints: local saves, publishing to git, resuming another member's run.

Two tiers:
  runs/<line>/            local, frequent, gitignored: latest.pt, pool/, metrics.jsonl, tb/, replays/
  checkpoints/<line>/     published, committed to git: latest.pt, pool/, metrics.jsonl, meta.json

    python -m cup.checkpoint publish <line>      # copy runs/<line> -> checkpoints/<line>, commit, push
    python -m cup.checkpoint status              # list published lines (steps, who, when)

Rules (see TREINO.md):
  * one line, one owner at a time. Binary files cannot be merged: to branch off, resume under a
    NEW line name (python -m cup.train --resume <old> --line <new>).
  * publishing never force-pushes: it pulls with rebase first and only stages checkpoints/<line>.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
PUBLISHED = ROOT / "checkpoints"


def atomic_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def safe_load(path: Path):
    path = Path(path)
    head = path.read_bytes()[:64]
    if head.startswith(b"version https://git-lfs"):
        raise RuntimeError(f"{path} is a Git LFS pointer, not a checkpoint: run `git lfs pull`")
    return torch.load(path, map_location="cpu", weights_only=False)


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def find_latest(line: str) -> Path | None:
    """The newer of runs/<line>/latest.pt (local) and checkpoints/<line>/latest.pt (published)."""
    cands = []
    for base in (RUNS, PUBLISHED):
        p = base / line / "latest.pt"
        if p.is_file():
            try:
                steps = int(safe_load(p)["env_steps"])
            except Exception:
                continue
            cands.append((steps, base == RUNS, p))
    if not cands:
        return None
    return max(cands)[2]


def _meta(state: dict) -> dict:
    return {
        "line": state["config"]["line"],
        "env_steps": int(state["env_steps"]),
        "updates": int(state["updates"]),
        "member": os.environ.get("CUP_MEMBER", getpass.getuser()),
        "host": platform.node(),
        "code_sha": git_sha(),
        "parent": state.get("parent"),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_eval": state.get("last_eval"),
    }


def _git(*args, check=True):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=check)


def publish(line: str, push: bool = True, message: str | None = None) -> bool:
    src = RUNS / line
    if not (src / "latest.pt").is_file():
        raise FileNotFoundError(f"{src / 'latest.pt'} not found (nothing trained locally for {line!r})")
    state = safe_load(src / "latest.pt")
    dst = PUBLISHED / line
    if push:
        r = _git("pull", "--rebase", "--autostash", check=False)
        if r.returncode != 0:
            print("git pull --rebase failed; publish aborted:\n" + r.stderr)
            return False
        remote = dst / "meta.json"
        if remote.is_file():
            m = json.loads(remote.read_text())
            if m.get("env_steps", 0) > int(state["env_steps"]):
                print(f"published {line} is AHEAD ({m['env_steps']:,} > {int(state['env_steps']):,} steps, "
                      f"by {m.get('member')}). Not overwriting: resume from it or use a new line name.")
                return False
    (dst / "pool").mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "latest.pt", dst / "latest.pt")
    for f in (src / "pool").glob("*.pt"):
        if not (dst / "pool" / f.name).exists():
            shutil.copy2(f, dst / "pool" / f.name)
    for f in (dst / "pool").glob("*.pt"):           # mirror pool removals
        if not (src / "pool" / f.name).exists():
            f.unlink()
    for name in ("metrics.jsonl", "evals.jsonl", "config.yaml", "latest_match.cjr"):
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)
    meta = _meta(state)
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"published {line}: {meta['env_steps']:,} env steps -> {dst}")
    if not push:
        return True
    _git("add", "-A", str(dst.relative_to(ROOT)))
    msg = message or f"checkpoint {line}: {meta['env_steps']:,} env steps ({meta['member']}@{meta['host']})"
    r = _git("commit", "-m", msg, check=False)
    if r.returncode != 0 and "nothing to commit" in (r.stdout + r.stderr):
        print("nothing new to commit")
        return True
    r = _git("push", check=False)
    if r.returncode != 0:
        print("git push failed (commit kept locally; run `git pull --rebase && git push`):\n" + r.stderr)
        return False
    print("pushed")
    return True


def publish_bc(model, line: str, meta: dict) -> None:
    """Publish a BC-trained model as the init of PPO lines (checkpoints/<line>/latest.pt)."""
    dst = PUBLISHED / line
    dst.mkdir(parents=True, exist_ok=True)
    atomic_save({"model": model.state_dict(), "env_steps": 0, "updates": 0, "kind": "bc",
                 "config": {"line": line}, "bc_meta": meta}, dst / "latest.pt")
    (dst / "meta.json").write_text(json.dumps({"line": line, "kind": "bc", "env_steps": 0,
                                               "code_sha": git_sha(), **meta}, indent=2))
    print(f"BC init published to {dst / 'latest.pt'} (commit it with git)")


def status() -> None:
    for meta in sorted(PUBLISHED.glob("*/meta.json")):
        m = json.loads(meta.read_text())
        ev = m.get("last_eval") or {}
        extra = f"  eval vs builtin: diff {ev.get('goal_diff_per_match', float('nan')):+.2f}" if ev else ""
        print(f"{m['line']:24s} {m.get('env_steps', 0):>14,} steps  {m.get('member', '?')}@{m.get('host', '?')}  "
              f"{m.get('time', '')}{extra}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pub = sub.add_parser("publish")
    pub.add_argument("line")
    pub.add_argument("--no-push", action="store_true")
    sub.add_parser("status")
    a = p.parse_args()
    if a.cmd == "publish":
        publish(a.line, push=not a.no_push)
    else:
        status()


if __name__ == "__main__":
    main()

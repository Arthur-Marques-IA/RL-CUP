"""Training configuration: a YAML file -> nested dataclasses.

Every schedule is keyed on ENVIRONMENT STEPS (not updates), so runs on machines with different
env counts follow the same curriculum. See configs/*.yaml for documented examples.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class EnvCfg:
    num_envs: int = 64
    device: str = "cpu"                # "cpu" or "cuda"
    threads: int = 0                   # cuju CPU threads (0 = all cores)
    builtin_fraction: float = 1.0      # share of envs vs the built-in AI (5_vs_5_l4_r0); rest vs snapshots (l4_r4)
    stagger: bool = True               # desynchronise matches at the start (random first-episode cut)


@dataclass
class PPOCfg:
    rollout: int = 128
    gamma: float = 0.995
    gae_lambda: float = 0.95
    lr: float = 3e-4
    lr_final: float = 3e-4             # linear in env steps until total_steps
    epochs: int = 4
    minibatches: int = 4
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.003
    max_grad_norm: float = 0.5
    hidden: int = 256


@dataclass
class RewardCfg:
    # r = scoring + w(t) * (gamma * Phi(s') - Phi(s));  Phi = ball_x * alpha + possession * beta
    alpha: float = 0.2                 # ball progress potential (ball x in [-1, 1])
    beta: float = 0.05                 # possession potential (+1 ours / -1 theirs, by last touch)
    shaping_start: int = 0             # env steps where w starts decaying from 1
    shaping_end: int = 50_000_000      # env steps where w reaches shaping_final
    shaping_final: float = 0.0


@dataclass
class InitCfg:
    checkpoint: str | None = None      # e.g. checkpoints/bc_gkbug_v1/latest.pt (BC init)
    kl_coef: float = 0.0               # KL(pi || pi_init) penalty, decays linearly to 0
    kl_end: int = 20_000_000           # env steps where the KL penalty reaches 0
    critic_warmup: int = 0             # env steps with the actor frozen (value head learns first)


@dataclass
class LeagueCfg:
    refresh_latest: int = 2_000_000    # env steps between copies of the learner into the "latest" opponent
    snapshot_every: int = 10_000_000   # env steps between snapshots added to the pool
    pool_size: int = 10                # pool members kept (oldest dropped)
    p_latest: float = 0.6              # P(opponent = latest copy) at each new match; else uniform over pool
    seed_pool: list[str] = field(default_factory=list)  # extra frozen opponents (model .pt paths)


@dataclass
class LogCfg:
    eval_every: int = 5_000_000        # env steps between evaluations vs builtin (full matches)
    eval_matches: int = 16
    save_replay: bool = True           # GRF .dump of one eval match (view with cuju-render / GRF replay)
    checkpoint_every: int = 2_000_000  # env steps between local checkpoints (runs/<line>/latest.pt)
    publish_every_minutes: float = 0   # >0: auto `git push` of checkpoints/<line>/ every N minutes


@dataclass
class TrainCfg:
    line: str = "default"              # experiment line = checkpoint folder name
    seed: int = 1
    total_steps: int = 100_000_000
    env: EnvCfg = field(default_factory=EnvCfg)
    ppo: PPOCfg = field(default_factory=PPOCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    init: InitCfg = field(default_factory=InitCfg)
    league: LeagueCfg = field(default_factory=LeagueCfg)
    log: LogCfg = field(default_factory=LogCfg)

    # fields that may differ when another machine resumes a line
    RESUMABLE_OVERRIDES = ("env.num_envs", "env.device", "env.threads", "log.publish_every_minutes",
                           "total_steps")


def _build(cls, data: dict):
    kwargs = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for k, v in (data or {}).items():
        if k not in fields:
            raise ValueError(f"unknown config key {cls.__name__}.{k}")
        ftype = fields[k].type
        sub = {"EnvCfg": EnvCfg, "PPOCfg": PPOCfg, "RewardCfg": RewardCfg, "InitCfg": InitCfg,
               "LeagueCfg": LeagueCfg, "LogCfg": LogCfg}.get(ftype if isinstance(ftype, str) else ftype.__name__)
        kwargs[k] = _build(sub, v) if sub else v
    return cls(**kwargs)


def load_config(path: str | Path, overrides: list[str] | None = None) -> TrainCfg:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        cur = data
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = yaml.safe_load(val)
    return _build(TrainCfg, data)


def to_dict(cfg: TrainCfg) -> dict:
    return dataclasses.asdict(cfg)


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def drift(saved: dict, current: dict) -> list[str]:
    """Config keys that differ (ignoring the ones another machine may change)."""
    a, b = flatten(saved), flatten(current)
    return [k for k in sorted(set(a) | set(b))
            if a.get(k) != b.get(k) and k not in TrainCfg.RESUMABLE_OVERRIDES]

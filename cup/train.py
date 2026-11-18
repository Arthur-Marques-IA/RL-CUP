"""PPO training on cuju with a shared 4-player policy, potential shaping and a snapshot league.

    python -m cup.train --config configs/ppo_bc_gkbug.yaml            # new line
    python -m cup.train --resume ppo_bc_gkbug                         # continue (local or published)
    python -m cup.train --resume ppo_bc_gkbug --line ppo_bc_gkbug_b   # branch off into a new line
    python -m cup.train --config configs/smoke.yaml                   # 2-minute CPU smoke test

Outputs (runs/<line>/): latest.pt, pool/*.pt, metrics.jsonl, evals.jsonl, tb/ (TensorBoard),
replays/*.dump, config.yaml. Publish to git with `python -m cup.checkpoint publish <line>`
(or log.publish_every_minutes in the config).

Design notes (see RECOMPENSA.md / LIGAS.md / TREINO.md):
  * observation = egocentric features incl. steps_left and score -> the match end is TERMINAL.
  * learner = the 4 left agents (slots 0..3) of every env; envs are either vs the built-in AI
    (bundle 5_vs_5_l4_r0) or vs a frozen opponent (5_vs_5_l4_r4, slots 4..7).
  * opponent per pool env, drawn at each new match: the "latest" copy of the learner with prob
    p_latest, else uniform over the snapshot pool (+ seed_pool). Opponents play argmax.
  * all schedules are in env steps.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import torch
import yaml

import cuju
from cup import features
from cup.checkpoint import RUNS, atomic_save, find_latest, publish, safe_load
from cup.config import TrainCfg, drift, load_config, to_dict
from cup.model import N_ACTIONS, ActorCritic
from cup.rewards import PotentialShaping

L = 4  # learner agents per env (left slots 0..3)


def linear(t, t0, t1, v0, v1):
    if t1 <= t0:
        return v1 if t >= t1 else v0
    a = min(max((t - t0) / (t1 - t0), 0.0), 1.0)
    return v0 + a * (v1 - v0)


class Logger:
    def __init__(self, out: Path):
        self.f = open(out / "metrics.jsonl", "a", encoding="utf-8")
        self.fe = open(out / "evals.jsonl", "a", encoding="utf-8")
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(str(out / "tb"))
        except Exception:
            self.tb = None
            print("tensorboard not installed: only metrics.jsonl will be written")

    def log(self, step: int, d: dict, eval_=False):
        d = {"env_steps": step, "time": round(time.time(), 1), **d}
        (self.fe if eval_ else self.f).write(json.dumps(d) + "\n")
        (self.fe if eval_ else self.f).flush()
        if self.tb:
            prefix = "eval/" if eval_ else ""
            for k, v in d.items():
                if isinstance(v, (int, float)) and k not in ("env_steps", "time") and math.isfinite(v):
                    self.tb.add_scalar(prefix + k, v, step)
            self.tb.flush()


# --------------------------------------------------------------------------- evaluation

def _recorder(batch, title: str):
    """cuju-render recorder of env 0 (3D replay in the browser), or None if not installed."""
    try:
        import cuju.render as render
    except Exception:
        return None
    return render.record(batch, 0, title=title, teams=[{"name": "UFG (nós)"}, {"name": "builtin 0.05"}])


@torch.no_grad()
def evaluate(model: ActorCritic, matches: int, seed: int, replay_dir: Path | None, title: str = "eval") -> dict:
    """Full matches from kick-off vs the built-in AI, argmax policy, on CPU. With replay_dir, env 0
    is saved as replay_dir/match.cjr (cuju-render) or, without cuju-render, as a GRF .dump."""
    m = copy.deepcopy(model).cpu().eval()
    with cuju.BatchEnvironment("5_vs_5_l4_r0", environment_count=matches, device="cpu",
                               representation="raw", rewards="scoring", seed=seed,
                               auto_reset=False, with_raw=replay_dir is not None) as batch:
        writer = rec = None
        if replay_dir is not None:
            replay_dir.mkdir(parents=True, exist_ok=True)
            rec = _recorder(batch, title)
            if rec is None:
                from cuju import dumps as cuju_dumps
                writer = cuju_dumps.DumpWriter(batch, str(replay_dir), environments=[0])
        raw = batch.reset()
        gf = torch.zeros(matches, dtype=torch.long)
        ga = torch.zeros(matches, dtype=torch.long)
        fin = torch.zeros(matches, dtype=torch.bool)
        hist = torch.zeros(N_ACTIONS)
        while not fin.all():
            a = m.logits(features.encode(raw)).argmax(-1)
            hist += torch.bincount(a[~fin].flatten(), minlength=N_ACTIONS)
            raw, _, done, info = batch.step(a.to(torch.int32))
            if writer is not None:
                writer.record(a.to(torch.int32))
            sr = info["score_reward"].long()
            gf += ((sr == 1) & ~fin).long()
            ga += ((sr == -1) & ~fin).long()
            fin |= done.bool()
        if writer is not None:
            writer.close()
        if rec is not None:
            rec.close()
            if rec.saved:
                import shutil
                shutil.copy2(rec.saved[-1], replay_dir / "match.cjr")
    w, d, l = int((gf > ga).sum()), int((gf == ga).sum()), int((gf < ga).sum())
    top = hist / hist.sum()
    return {"W": w, "D": d, "L": l, "goals_for": int(gf.sum()), "goals_against": int(ga.sum()),
            "goal_diff_per_match": float((gf - ga).float().mean()),
            "points_per_match": (3 * w + d) / matches,
            "top_action_share": float(top.max()), "shot_share": float(top[12])}


# --------------------------------------------------------------------------- trainer

class Trainer:
    def __init__(self, cfg: TrainCfg, resume_state: dict | None = None, parent: str | None = None):
        self.cfg = cfg
        self.parent = parent
        torch.manual_seed(cfg.seed + (resume_state or {}).get("updates", 0))
        self.dev = torch.device(cfg.env.device)
        self.out = RUNS / cfg.line
        (self.out / "pool").mkdir(parents=True, exist_ok=True)
        (self.out / "config.yaml").write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))

        # ---- model / optimiser / reference (BC) policy
        self.model = ActorCritic(hidden=cfg.ppo.hidden).to(self.dev)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.ppo.lr, eps=1e-5)
        self.ref = None
        self.env_steps = 0
        self.updates = 0
        self.obs_stats_ready = False
        self.pool_files: list[str] = []
        self.last_eval = None
        if resume_state is not None:
            self.model.load_state_dict(resume_state["model"])
            if "optimizer" in resume_state:
                self.opt.load_state_dict(resume_state["optimizer"])
            self.env_steps = int(resume_state["env_steps"])
            self.updates = int(resume_state["updates"])
            self.obs_stats_ready = True
            self.pool_files = list(resume_state.get("pool", []))
            self.last_eval = resume_state.get("last_eval")
            self.parent = resume_state.get("parent", parent)
        elif cfg.init.checkpoint:
            init = safe_load(Path(cfg.init.checkpoint))
            self.model.load_state_dict(init["model"], strict=False)
            self.obs_stats_ready = True
            self.parent = cfg.init.checkpoint
            print(f"initialised from {cfg.init.checkpoint}")
        if cfg.init.checkpoint and cfg.init.kl_coef > 0:
            init = safe_load(Path(cfg.init.checkpoint))
            self.ref = ActorCritic(hidden=cfg.ppo.hidden).to(self.dev)
            self.ref.load_state_dict(init["model"], strict=False)
            self.ref.requires_grad_(False).eval()

        # ---- environments
        E = cfg.env.num_envs
        self.n_builtin = int(round(E * cfg.env.builtin_fraction))
        self.n_pool = E - self.n_builtin
        bundles, assignment = ["5_vs_5_l4_r0"], None
        if self.n_pool > 0:
            bundles = ["5_vs_5_l4_r0", "5_vs_5_l4_r4"] if self.n_builtin > 0 else ["5_vs_5_l4_r4"]
            if self.n_builtin > 0:
                assignment = torch.cat([torch.zeros(self.n_builtin, dtype=torch.long),
                                        torch.ones(self.n_pool, dtype=torch.long)])
        kw = dict(environment_count=E, device=cfg.env.device, representation="raw",
                  rewards="scoring", seed=cfg.seed * 1000 + self.updates, threads=cfg.env.threads)
        if assignment is not None:
            kw["assignment"] = assignment
        self.batch = cuju.BatchEnvironment(bundles if len(bundles) > 1 else bundles[0], **kw)
        self.agents = self.batch.agents

        # ---- league
        self.latest = copy.deepcopy(self.model).requires_grad_(False).eval()
        self.pool: list[ActorCritic] = []
        for f in self.pool_files:
            self._add_pool_member(self.out / "pool" / f, register=False)
        for f in cfg.league.seed_pool:
            self._add_pool_member(Path(f), register=False)
        self.opp_idx = torch.zeros(max(self.n_pool, 1), dtype=torch.long, device=self.dev)

    # ---------------------------------------------------------------- league helpers
    def _add_pool_member(self, path: Path, register=True):
        st = safe_load(path)
        m = ActorCritic(hidden=self.cfg.ppo.hidden).to(self.dev)
        m.load_state_dict(st["model"] if "model" in st else st, strict=False)
        m.requires_grad_(False).eval()
        self.pool.append(m)
        if register:
            self.pool_files.append(path.name)

    def _snapshot(self):
        name = f"snap_{self.env_steps:012d}.pt"
        atomic_save({"model": self.model.state_dict(), "env_steps": self.env_steps}, self.out / "pool" / name)
        self._add_pool_member(self.out / "pool" / name)
        n_seed = len(self.cfg.league.seed_pool)
        while len(self.pool_files) > self.cfg.league.pool_size:
            old = self.pool_files.pop(0)
            self.pool.pop(0)  # own snapshots come first in self.pool
            (self.out / "pool" / old).unlink(missing_ok=True)
        assert len(self.pool) == len(self.pool_files) + n_seed

    def _sample_opponents(self, n: int) -> torch.Tensor:
        """0 = latest copy, 1..P = pool members."""
        P = len(self.pool)
        if P == 0:
            return torch.zeros(n, dtype=torch.long, device=self.dev)
        latest = torch.rand(n, device=self.dev) < self.cfg.league.p_latest
        other = torch.randint(1, P + 1, (n,), device=self.dev)
        return torch.where(latest, torch.zeros_like(other), other)

    @torch.no_grad()
    def _opponent_actions(self, x_right: torch.Tensor) -> torch.Tensor:
        models = [self.latest] + self.pool
        logits = torch.stack([m.logits(x_right) for m in models])         # [P+1, n_pool, 4, 19]
        idx = self.opp_idx[None, :, None, None].expand(1, -1, x_right.shape[1], N_ACTIONS)
        return torch.gather(logits, 0, idx).squeeze(0).argmax(-1)

    # ---------------------------------------------------------------- state
    def state(self) -> dict:
        return {"model": self.model.state_dict(), "optimizer": self.opt.state_dict(),
                "env_steps": self.env_steps, "updates": self.updates, "config": to_dict(self.cfg),
                "pool": self.pool_files, "parent": self.parent, "last_eval": self.last_eval}

    def save(self):
        atomic_save(self.state(), self.out / "latest.pt")

    # ---------------------------------------------------------------- main loop
    def run(self):
        cfg, dev = self.cfg, self.dev
        E, T, A = cfg.env.num_envs, cfg.ppo.rollout, self.agents
        log = Logger(self.out)
        shaping = PotentialShaping(E, cfg.reward.alpha, cfg.reward.beta, cfg.ppo.gamma, dev)

        obs = torch.zeros((T, E, L, features.FEATURE_DIM), device=dev)
        actions = torch.zeros((T, E, L), dtype=torch.long, device=dev)
        logps = torch.zeros((T, E, L), device=dev)
        values = torch.zeros((T, E, L), device=dev)
        rewards = torch.zeros((T, E, L), device=dev)
        dones = torch.zeros((T, E), device=dev)
        act_full = torch.zeros((E, A), dtype=torch.int32, device=dev)

        raw = self.batch.reset()
        shaping.reset(raw)
        self.opp_idx = self._sample_opponents(max(self.n_pool, 1))
        goals_l = torch.zeros(E, device=dev)
        goals_r = torch.zeros(E, device=dev)
        do_stagger = cfg.env.stagger and self.updates == 0
        stagger_at = (torch.randint(1, 3001, (E,), device=dev) if do_stagger
                      else torch.full((E,), -1, device=dev))
        # the first (cut) match of each env is not counted; every later match starts at kick-off
        valid_ep = torch.full((E,), not do_stagger, dtype=torch.bool, device=dev)
        global_t = 0
        next_eval = (self.env_steps // cfg.log.eval_every + 1) * cfg.log.eval_every
        next_ckpt = self.env_steps + cfg.log.checkpoint_every
        next_refresh = self.env_steps + cfg.league.refresh_latest
        next_snap = (self.env_steps // cfg.league.snapshot_every + 1) * cfg.league.snapshot_every
        last_publish = time.time()
        builtin_env = torch.arange(E, device=dev) < self.n_builtin

        print(f"[{cfg.line}] {E} envs ({self.n_builtin} vs builtin, {self.n_pool} vs league) on {dev}; "
              f"start at {self.env_steps:,} env steps; target {cfg.total_steps:,}")
        while self.env_steps < cfg.total_steps:
            t0 = time.time()
            frac_lr = linear(self.env_steps, 0, cfg.total_steps, cfg.ppo.lr, cfg.ppo.lr_final)
            for g in self.opt.param_groups:
                g["lr"] = frac_lr
            w_shape = linear(self.env_steps, cfg.reward.shaping_start, cfg.reward.shaping_end, 1.0,
                             cfg.reward.shaping_final)
            kl_coef = linear(self.env_steps, 0, cfg.init.kl_end, cfg.init.kl_coef, 0.0) if self.ref else 0.0

            st = {k: torch.zeros((), device=dev) for k in
                  ("ep_b", "w_b", "d_b", "l_b", "gf_b", "ga_b", "ep_p", "w_p", "d_p", "l_p", "gf_p", "ga_p",
                   "shaping", "score", "own_ball")}
            act_hist = torch.zeros(N_ACTIONS, device=dev)

            # ---------------- rollout
            for t in range(T):
                x = features.encode(raw)                          # [E, A, 87]
                xl = x[:, :L]
                with torch.no_grad():
                    logits = self.model.logits(xl)
                    dist = torch.distributions.Categorical(logits=logits)
                    a = dist.sample()
                    values[t] = self.model.value(xl)
                    logps[t] = dist.log_prob(a)
                obs[t] = xl
                actions[t] = a
                act_full[:, :L] = a.to(torch.int32)
                if self.n_pool > 0:
                    act_full[self.n_builtin:, L:2 * L] = self._opponent_actions(
                        x[self.n_builtin:, L:2 * L]).to(torch.int32)
                act_hist += torch.bincount(a.flatten(), minlength=N_ACTIONS)

                raw, rew, done, info = self.batch.step(act_full)
                score_r = rew[:, :L].float().clone()
                d = done.bool().clone()
                sr = info["score_reward"].float().clone()

                # stagger: cut each env's first match at a random step (counts as an end)
                if global_t < 3001 and do_stagger:
                    cut = stagger_at == global_t
                    self.batch.reset(mask=cut)
                    d = d | cut
                global_t += 1

                shaped = shaping.step(raw, d) * w_shape
                rewards[t] = score_r + shaped[:, None]
                dones[t] = d.float()
                st["shaping"] += shaped.sum()
                st["score"] += score_r[:, 0].sum()
                st["own_ball"] += (raw["ball_owned_team"][:, 0] == 0).float().sum()

                goals_l += (sr == 1).float()
                goals_r += (sr == -1).float()
                ended = d & valid_ep
                win, draw, loss = ended & (goals_l > goals_r), ended & (goals_l == goals_r), ended & (goals_l < goals_r)
                for suffix, sel in (("b", builtin_env), ("p", ~builtin_env)):
                    st["ep_" + suffix] += (ended & sel).float().sum()
                    st["w_" + suffix] += (win & sel).float().sum()
                    st["d_" + suffix] += (draw & sel).float().sum()
                    st["l_" + suffix] += (loss & sel).float().sum()
                    st["gf_" + suffix] += (goals_l * (ended & sel)).sum()
                    st["ga_" + suffix] += (goals_r * (ended & sel)).sum()
                goals_l = torch.where(d, torch.zeros_like(goals_l), goals_l)
                goals_r = torch.where(d, torch.zeros_like(goals_r), goals_r)
                valid_ep |= d   # the match that starts now is a full one
                if self.n_pool > 0:
                    new = self._sample_opponents(self.n_pool)
                    self.opp_idx = torch.where(d[self.n_builtin:], new, self.opp_idx)

            # ---------------- obs stats (scratch runs only): frozen after the first rollout
            if not self.obs_stats_ready:
                self.model.set_obs_stats(obs.reshape(-1, features.FEATURE_DIM))
                self.obs_stats_ready = True

            # ---------------- GAE (terminal at match end)
            with torch.no_grad():
                next_value = self.model.value(features.encode(raw)[:, :L])
                adv = torch.zeros_like(rewards)
                last = torch.zeros_like(next_value)
                for t in reversed(range(T)):
                    nv = next_value if t == T - 1 else values[t + 1]
                    nonterm = (1.0 - dones[t])[:, None]
                    delta = rewards[t] + cfg.ppo.gamma * nonterm * nv - values[t]
                    last = delta + cfg.ppo.gamma * cfg.ppo.gae_lambda * nonterm * last
                    adv[t] = last
                ret = adv + values

            # ---------------- PPO update
            b_obs = obs.reshape(-1, features.FEATURE_DIM)
            b_act, b_logp, b_adv, b_ret, b_val = (z.reshape(-1) for z in (actions, logps, adv, ret, values))
            n = b_obs.shape[0]
            mb = n // cfg.ppo.minibatches
            warmup = self.env_steps < cfg.init.critic_warmup
            stats = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "kl_init": 0.0, "clipfrac": 0.0}
            count = 0
            for _ in range(cfg.ppo.epochs):
                perm = torch.randperm(n, device=dev)
                for i in range(cfg.ppo.minibatches):
                    idx = perm[i * mb:(i + 1) * mb]
                    o = b_obs[idx]
                    logits = self.model.logits(o)
                    logp_all = torch.log_softmax(logits, -1)
                    logp = logp_all.gather(1, b_act[idx, None]).squeeze(1)
                    ratio = (logp - b_logp[idx]).exp()
                    a_n = b_adv[idx]
                    a_n = (a_n - a_n.mean()) / (a_n.std() + 1e-8)
                    pg = torch.max(-a_n * ratio, -a_n * ratio.clamp(1 - cfg.ppo.clip, 1 + cfg.ppo.clip)).mean()
                    v = self.model.value(o)
                    v_clip = b_val[idx] + (v - b_val[idx]).clamp(-cfg.ppo.clip, cfg.ppo.clip)
                    v_loss = 0.5 * torch.max((v - b_ret[idx]) ** 2, (v_clip - b_ret[idx]) ** 2).mean()
                    probs = logp_all.exp()
                    entropy = -(probs * logp_all).sum(-1).mean()
                    kl_init = torch.zeros((), device=dev)
                    if self.ref is not None and kl_coef > 0:
                        with torch.no_grad():
                            ref_logp = torch.log_softmax(self.ref.logits(o), -1)
                        kl_init = (probs * (logp_all - ref_logp)).sum(-1).mean()
                    if warmup:
                        loss = cfg.ppo.value_coef * v_loss
                    else:
                        loss = pg + cfg.ppo.value_coef * v_loss - cfg.ppo.entropy_coef * entropy + kl_coef * kl_init
                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.ppo.max_grad_norm)
                    self.opt.step()
                    with torch.no_grad():
                        stats["pg_loss"] += float(pg)
                        stats["v_loss"] += float(v_loss)
                        stats["entropy"] += float(entropy)
                        stats["approx_kl"] += float(((ratio - 1) - (logp - b_logp[idx])).mean())
                        stats["kl_init"] += float(kl_init)
                        stats["clipfrac"] += float(((ratio - 1).abs() > cfg.ppo.clip).float().mean())
                    count += 1
            stats = {k: v / count for k, v in stats.items()}

            self.updates += 1
            self.env_steps += E * T
            elapsed = time.time() - t0
            s = {k: float(v) for k, v in st.items()}
            hist = (act_hist / act_hist.sum()).cpu()
            row = {
                "updates": self.updates, "sps": round(E * T / elapsed),
                "lr": frac_lr, "shaping_w": w_shape, "kl_coef": kl_coef, "critic_warmup": int(warmup),
                **stats,
                "reward_shaping_per_step": s["shaping"] / (E * T), "reward_score_per_step": s["score"] / (E * T),
                "possession_ours": s["own_ball"] / (E * T),
                "top_action_share": float(hist.max()), "shot_share": float(hist[12]),
                "pool_size": len(self.pool),
            }
            for suffix, name in (("b", "builtin"), ("p", "league")):
                if s["ep_" + suffix] > 0:
                    ne = s["ep_" + suffix]
                    row[f"{name}/matches"] = ne
                    row[f"{name}/win_rate"] = s["w_" + suffix] / ne
                    row[f"{name}/goal_diff"] = (s["gf_" + suffix] - s["ga_" + suffix]) / ne
                    row[f"{name}/points"] = (3 * s["w_" + suffix] + s["d_" + suffix]) / ne
            log.log(self.env_steps, row)
            print(f"[{cfg.line}] upd {self.updates} steps {self.env_steps:,} sps {row['sps']:,} "
                  f"ent {stats['entropy']:.2f} vloss {stats['v_loss']:.4f} kl_init {stats['kl_init']:.3f} "
                  + " ".join(f"{k}={row[k]:+.2f}" for k in ("builtin/goal_diff", "league/goal_diff") if k in row),
                  flush=True)

            # ---------------- league bookkeeping
            if self.env_steps >= next_refresh:
                self.latest.load_state_dict(self.model.state_dict())
                next_refresh = self.env_steps + cfg.league.refresh_latest
            if self.env_steps >= next_snap:
                self._snapshot()
                next_snap += cfg.league.snapshot_every

            # ---------------- evaluation
            if self.env_steps >= next_eval:
                rd = self.out / "replays" / f"step_{self.env_steps:012d}" if cfg.log.save_replay else None
                ev = evaluate(self.model, cfg.log.eval_matches, seed=10_000 + self.updates, replay_dir=rd,
                              title=f"{cfg.line} @ {self.env_steps:,} steps vs builtin")
                if rd is not None and (rd / "match.cjr").is_file():
                    import shutil
                    shutil.copy2(rd / "match.cjr", self.out / "latest_match.cjr")
                self.last_eval = ev
                log.log(self.env_steps, ev, eval_=True)
                print(f"[{cfg.line}] EVAL vs builtin: {ev['W']}/{ev['D']}/{ev['L']}  "
                      f"diff {ev['goal_diff_per_match']:+.2f}  pts {ev['points_per_match']:.2f}", flush=True)
                next_eval += cfg.log.eval_every

            # ---------------- checkpoint / publish
            if self.env_steps >= next_ckpt:
                self.save()
                next_ckpt = self.env_steps + cfg.log.checkpoint_every
            if cfg.log.publish_every_minutes > 0 and time.time() - last_publish > 60 * cfg.log.publish_every_minutes:
                self.save()
                publish(cfg.line)
                last_publish = time.time()

        self.save()
        print(f"[{cfg.line}] done: {self.env_steps:,} env steps; saved {self.out / 'latest.pt'}")
        if cfg.log.publish_every_minutes > 0:
            publish(cfg.line)
        self.batch.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, help="YAML config for a new line")
    p.add_argument("--resume", help="line to continue (newest of runs/<line> and checkpoints/<line>)")
    p.add_argument("--line", help="new line name (branch off when resuming)")
    p.add_argument("--set", nargs="*", default=[], metavar="key=value",
                   help="overrides, e.g. env.num_envs=2048 env.device=cuda total_steps=3e8")
    p.add_argument("--force", action="store_true", help="resume even if the config drifted")
    a = p.parse_args()

    if a.resume:
        path = find_latest(a.resume)
        if path is None:
            raise SystemExit(f"no checkpoint for line {a.resume!r} in runs/ or checkpoints/")
        state = safe_load(path)
        saved = state["config"]
        tmp = RUNS / "_resume_cfg.yaml"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(yaml.safe_dump(saved))
        cfg = load_config(a.config or tmp, a.set)
        if a.config:
            bad = drift(saved, to_dict(cfg))
            if bad and not a.force:
                raise SystemExit(f"config drift on resume: {bad} (use --force or a new --line)")
        parent = None
        if a.line and a.line != cfg.line:
            parent = f"{cfg.line}@{int(state['env_steps'])}"
            cfg.line = a.line
        # pool snapshots live next to the checkpoint they came from
        src_pool = path.parent / "pool"
        dst_pool = RUNS / cfg.line / "pool"
        if src_pool.resolve() != dst_pool.resolve() and src_pool.is_dir():
            dst_pool.mkdir(parents=True, exist_ok=True)
            for f in src_pool.glob("*.pt"):
                if not (dst_pool / f.name).exists():
                    import shutil
                    shutil.copy2(f, dst_pool / f.name)
        # keep the curves continuous across machines: start from the published logs
        import shutil
        for name in ("metrics.jsonl", "evals.jsonl", "latest_match.cjr"):
            src, dst = path.parent / name, RUNS / cfg.line / name
            if src.is_file() and src.resolve() != dst.resolve() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        print(f"resuming {a.resume} from {path} ({int(state['env_steps']):,} env steps)")
        if parent:
            state["parent"] = parent
        Trainer(cfg, resume_state=state, parent=parent).run()
    else:
        if not a.config:
            raise SystemExit("--config or --resume is required")
        cfg = load_config(a.config, a.set)
        if a.line:
            cfg.line = a.line
        if find_latest(cfg.line) is not None:
            raise SystemExit(f"line {cfg.line!r} already has a checkpoint: use --resume {cfg.line} "
                             f"or choose another --line")
        Trainer(cfg).run()


if __name__ == "__main__":
    main()

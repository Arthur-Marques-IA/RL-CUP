"""Check a machine before training and suggest settings.

    python -m cup.doctor            # versions, cuju self-check, CPU/GPU throughput, suggested num_envs
    python -m cup.doctor --quick    # skip `cuju check`

Throughput is measured for the real training step: cuju raw observations + our feature encoder +
the policy forward, with 4 agents per env.
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time

import torch

import cuju
from cup import features
from cup.model import ActorCritic


def bench(device: str, envs: int, steps: int = 200) -> float:
    model = ActorCritic().to(device).eval()
    with cuju.BatchEnvironment("5_vs_5_l4_r0", environment_count=envs, device=device,
                               representation="raw", rewards="scoring", seed=0) as b:
        raw = b.reset()
        for i in range(steps + 20):
            if i == 20:
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
            with torch.no_grad():
                a = torch.distributions.Categorical(logits=model.logits(features.encode(raw))).sample()
            raw, *_ = b.step(a.to(torch.int32))
        if device == "cuda":
            torch.cuda.synchronize()
        return envs * steps / (time.perf_counter() - t0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    print(f"python {sys.version.split()[0]} | {platform.platform()} | torch {torch.__version__} | cuju {cuju.__version__}")
    print(f"CPU threads: {torch.get_num_threads()} (torch), cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, compute capability {props.major}.{props.minor}, {props.total_memory / 2**30:.1f} GiB")
        if (props.major, props.minor) not in ((7, 5), (8, 0), (8, 6), (8, 9)):
            print("  ! the cuju wheel has no kernels for this GPU (e.g. RTX 50 = 12.0): build cuju from source "
                  "(TREINO.md, 'GPU: build do cuju'), or train with env.device=cpu")
    if not a.quick:
        print("\n$ cuju check")
        subprocess.run([sys.executable, "-m", "cuju.check"], check=False)

    print("\nthroughput (env steps/s, 4 agents/env, features + policy included):")
    best = {}
    for envs in (16, 64, 256):
        try:
            sps = bench("cpu", envs)
            best["cpu"] = max(best.get("cpu", (0, 0)), (sps, envs))
            print(f"  cpu  {envs:6d} envs: {sps:10,.0f}")
        except Exception as e:
            print(f"  cpu  {envs:6d} envs: failed ({e})")
    if torch.cuda.is_available():
        for envs in (1024, 4096, 8192):
            try:
                sps = bench("cuda", envs)
                best["cuda"] = max(best.get("cuda", (0, 0)), (sps, envs))
                print(f"  cuda {envs:6d} envs: {sps:10,.0f}")
            except Exception as e:
                print(f"  cuda {envs:6d} envs: failed ({type(e).__name__}: {str(e)[:160]})")
                break
    if best:
        dev, (sps, envs) = max(best.items(), key=lambda kv: kv[1][0])
        print(f"\nsuggested: --set env.device={dev} env.num_envs={envs}   (~{sps / 1e6 * 3600:.0f} M env steps/hour "
              f"before PPO updates)")


if __name__ == "__main__":
    main()

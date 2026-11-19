#!/usr/bin/env bash
# DAgger rounds for the GKBug_v1 clone. Round 0 = plain BC data (data/gkbug_v1) + runs/bc_gkbug_v1_cw0.
# Each round: the current student plays (teacher acts with prob BETA), the teacher labels every state,
# the student is retrained on ALL data so far, then evaluated vs builtin (32 matches).
#   bash scripts/dagger.sh            (CPU-bound; ~25 min per round on 8 cores)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}
[ -x "$PY" ] || PY=.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
BETAS=(0.5 0.25 0.0)
prev=runs/bc_gkbug_v1_cw0/model.pt
data=(data/gkbug_v1)
for r in 1 2 3; do
  beta=${BETAS[$((r-1))]}
  out=data/gkbug_v1_dagger$r
  mkdir -p "$out" runs/logs
  echo "== round $r: student $prev, beta $beta"
  for s in 1 2 3 4 5 6 7; do
    $PY -m cup.bc collect --teacher grf:GKBug_v1 --envs 14 --seed $((100*r+s)) --student "$prev" --beta "$beta" \
        --out "$out/shard$s.pt" > runs/logs/dagger${r}_collect$s.log 2>&1 &
  done
  wait
  data+=("$out")
  $PY -m cup.bc train --data "${data[@]}" --epochs 20 --out runs/bc_gkbug_v1_dagger$r > runs/logs/dagger${r}_train.log 2>&1
  $PY -m cup.evaluate --left runs/bc_gkbug_v1_dagger$r/model.pt --matches 32 \
      --json runs/bc_gkbug_v1_dagger$r/eval_builtin.json > /dev/null 2>&1
  python -c "import json;d=json.load(open('runs/bc_gkbug_v1_dagger$r/eval_builtin.json'));print('round $r eval:',{k:d[k] for k in ('W','D','L','goal_diff_per_match','points_per_match')})"
  grep "val accuracy" runs/logs/dagger${r}_train.log
  prev=runs/bc_gkbug_v1_dagger$r/model.pt
done

#!/bin/bash
# 5 dataset 의 prebake 에 a_2 cache 기반 TTC brake floor 적용 (serial)
set -e
PY=/home/ailab-12/miniforge3/envs/ml_con/bin/python
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
ROOT=/home/ailab-12/git/scc_controller_imitator
LOG=/tmp/a2_inlane_aug.log
cd "$ROOT"

DATASETS=(
  "sangam_lead_0520|f1_prebake_iy_sangam_lead_0520_v1_brakeaug_prev20.npz|a2_out_cache_sangam_lead_0520_prev20_fixed.npz"
  "sangam_lead_0521|f1_prebake_iy_sangam_lead_0521_v1_brakeaug_prev20.npz|a2_out_cache_sangam_lead_0521_prev20_fixed.npz"
  "sangam_pertO_v2|f1_prebake_iy_sangam_pertO_v2_prev20.npz|a2_out_cache_sangam_pertO_v2_prev20_fixed.npz"
  "sangam_pertX_v2|f1_prebake_iy_sangam_pertX_v2_prev20.npz|a2_out_cache_sangam_pertX_v2_prev20_fixed.npz"
  "DriverB_new|f1_prebake_iy_DriverB_new_logged_fixed_prev20.npz|a2_out_cache_DriverB_new_prev20_fixed.npz"
)

echo "[start] $(date)" | tee -a "$LOG"
for ds in "${DATASETS[@]}"; do
  IFS='|' read -r name pb a2 <<< "$ds"
  OUT="$D/prebake/f1_prebake_iy_${name}_a2inlane_prev20.npz"
  if [ -f "$OUT" ]; then
    echo "[skip] $OUT 이미 존재" | tee -a "$LOG"
    continue
  fi
  echo "[run] $name" | tee -a "$LOG"
  $PY -u scripts/augment_prebake_a2_inlane.py \
    --prebake "$D/prebake/$pb" \
    --a2-cache "$D/cache/$a2" \
    --out "$OUT" 2>&1 | tail -15 | tee -a "$LOG"
done
echo "[done] $(date)" | tee -a "$LOG"

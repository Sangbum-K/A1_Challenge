#!/bin/bash
# 6 dataset 의 i_2 cache 빌드 (a_2 v5 fc_proj+head 통과) + AUC 검증
set -e
PY=/home/ailab-12/miniforge3/envs/ml_con/bin/python
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
ROOT=/home/ailab-12/git/scc_controller_imitator
LOG=/tmp/i2_build.log
CKPT=$ROOT/data/ckpts/a2_v5_d128_best_loss.pt
cd "$ROOT"

DATASETS=(
  "sangam_lead_0520|i1_tok_cache_sangam_lead_0520_prev20.npz|f1_prebake_iy_sangam_lead_0520_v1_brakeaug_prev20.npz"
  "sangam_lead_0521|i1_tok_cache_sangam_lead_0521_prev20.npz|f1_prebake_iy_sangam_lead_0521_v1_brakeaug_prev20.npz"
  "sangam_pertO_v2|i1_tok_cache_sangam_pertO_v2_prev20.npz|f1_prebake_iy_sangam_pertO_v2_prev20.npz"
  "sangam_pertX_v2|i1_tok_cache_sangam_pertX_v2_prev20.npz|f1_prebake_iy_sangam_pertX_v2_prev20.npz"
  "DriverB_new|i1_tok_cache_DriverB_new_prev20.npz|f1_prebake_iy_DriverB_new_logged_fixed_prev20.npz"
  "spd_v2|i1_tok_cache_spd_v2_prev20.npz|f1_prebake_iy_spd_streaming_brakeaug_prev20.npz"
)
echo "[start] $(date)" | tee -a "$LOG"
for ds in "${DATASETS[@]}"; do
  IFS='|' read -r name i1 pb <<< "$ds"
  OUT="$D/cache/i2_mem_cache_${name}_prev20.npz"
  if [ -f "$OUT" ]; then echo "[skip] $OUT" | tee -a "$LOG"; continue; fi
  echo "[build] $name" | tee -a "$LOG"
  $PY -u scripts/build_i2_cache.py \
    --i1-cache "$D/cache/$i1" \
    --prebake "$D/prebake/$pb" \
    --out "$OUT" \
    --a2-ckpt "$CKPT" 2>&1 | tail -10 | tee -a "$LOG"
done
echo "[i2 build done] $(date)" | tee -a "$LOG"

# AUC 검증
echo "[eval] AUC of a_2 v5 (per dataset)" | tee -a "$LOG"
$PY -u /tmp/measure_auc_v5.py 2>&1 | tee -a "$LOG"
echo "[done] $(date)" | tee -a "$LOG"

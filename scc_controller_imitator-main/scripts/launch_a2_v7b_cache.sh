#!/bin/bash
# a_2 v7b — v7 의 sangam_lead obj rec 3% 문제 fix
#   원인: focal_alpha=0.5 (= no weighting, BCE 동등)
#         sangam_lead_0521 GT pos rate 1.9% → negative loss 지배 → model "다 negative" 학습
#   fix:  focal_alpha=0.75 (positive upweight 3× negative)
#         v7 ckpt resume, +10 ep, LR 살아있게 1e-4
set -e
LOG=/tmp/a2_v7b_cache.log
CKPT_INIT=/home/ailab-12/git/scc_controller_imitator/data/ckpts/a2_v7_d128_regfix_best_loss.pt
source /home/ailab-12/miniforge3/etc/profile.d/conda.sh
conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data

nohup python -u scripts/train_a2_v5_from_cache.py \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0520_prev20.npz" \
  --aux-dir "$D/aux_targets_sangam_lead_0520" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0521_prev20.npz" \
  --aux-dir "$D/aux_targets_sangam_lead_0521" \
  --i1-cache "$D/cache/i1_tok_cache_DriverB_new_prev20.npz" \
  --aux-dir "$D/aux_targets_DriverB_new" \
  --i1-cache "$D/cache/i1_tok_cache_spd_v2_prev20.npz" \
  --aux-dir "$D/aux_targets_spd_fx640" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_pertO_v2_prev20.npz" \
  --aux-dir "$D/aux_targets_sangam_fx640" --run-filter "" --run-filter "" --run-filter "" --run-filter "" --run-filter "pertO" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_pertX_v2_prev20.npz" \
  --aux-dir "$D/aux_targets_sangam_fx640" \
  --resume-from "$CKPT_INIT" \
  --d-out 128 \
  --epochs 10 \
  --batch-size 256 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --workers 4 \
  --val-ratio 0.05 \
  --w-obj 1.0 --w-reg 1.0 \
  --use-focal --focal-alpha 0.75 --focal-gamma 2.0 \
  --scale-v 20.0 --scale-dist 40.0 \
  --fx 640 \
  --name v7b_d128_focal_alpha075 \
  --out-dir outputs_multi/a2 \
  > "$LOG" 2>&1 &
PID=$!
echo "a_2 v7b pid=$PID log=$LOG"
echo "$PID" > /tmp/a2_v7b.pid

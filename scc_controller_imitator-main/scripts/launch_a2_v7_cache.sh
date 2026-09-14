#!/bin/bash
# a_2 v7 — v5/v6 의 regression loss 버그 FIX 후 retrain
#   v5/v6 bug:  loss_reg = L1(pred[1:], target[1:])  ← target normalize 없음 + obj_mask 없음
#                → 90% 토큰 (obj=0, reg=0) 영향으로 model 이 모든 곳에서 0 출력 학습됨
#                → vL_rel pred std=0, closing 감지 recall 0%
#   v7 fix:    target / scales 적용 + obj_mask 만 loss 에 포함 (train_a2_head.py 와 동일)
#   동일 dataset, 동일 backbone (a_2 v4 resume), focal loss 사용
set -e
LOG=/tmp/a2_v7_cache.log
CKPT_INIT=/home/ailab-12/git/scc_controller_imitator/data/ckpts/a2_fx640_v4_best_loss.pt
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
  --epochs 20 \
  --batch-size 256 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --workers 4 \
  --val-ratio 0.05 \
  --w-obj 1.0 --w-reg 1.0 \
  --use-focal --focal-alpha 0.5 --focal-gamma 2.0 \
  --scale-v 20.0 --scale-dist 40.0 \
  --fx 640 \
  --name v7_d128_focal_regfix \
  --out-dir outputs_multi/a2 \
  > "$LOG" 2>&1 &
PID=$!
echo "a_2 v7 pid=$PID log=$LOG"
echo "$PID" > /tmp/a2_v7_cache.pid

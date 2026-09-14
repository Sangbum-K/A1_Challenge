#!/bin/bash
# a_2 v5 — i_1 cache 만 사용한 빠른 학습 (raw image load 없음)
# encoder + fusion frozen (v4 ckpt), fc_proj 256→128 + new AuxHead 128→7 만 학습
set -e
LOG=/tmp/a2_v5_cache.log
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
  --epochs 10 \
  --batch-size 256 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --workers 4 \
  --val-ratio 0.05 \
  --w-obj 1.0 --w-reg 1.0 \
  --scale-v 20.0 --scale-dist 40.0 \
  --fx 640 \
  --name v5_d128_cache \
  --out-dir outputs_multi/a2 \
  > "$LOG" 2>&1 &
PID=$!
echo "a_2 v5 cache pid=$PID log=$LOG"
echo "$PID" > /tmp/a2_v5_cache.pid

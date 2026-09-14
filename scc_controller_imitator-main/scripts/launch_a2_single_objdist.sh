#!/bin/bash
# a_2 single-frame + obj·dist-only (2026-05-26).
#   v7b 와 동일 datasets/split (seed=0) 로 공정 비교.
#   encoder.proj init = v7b ckpt, sa+fc_proj+head 는 from scratch.
#   obj focal α=0.75 (v7b 동등), reg = masked L1(dist) scale=40, +LDS.
set -e
LOG=/tmp/a2_single_objdist.log
CKPT_INIT=/home/ailab-12/git/scc_controller_imitator/data/ckpts/a2_v7b_d128_alpha075_best_loss.pt
source /home/ailab-12/miniforge3/etc/profile.d/conda.sh
conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data

nohup python -u scripts/train_a2_single_objdist.py \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0520_prev20.npz" \
  --aux-dir "$D/aux_targets_sangam_lead_0520" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0521_prev20.npz" \
  --aux-dir "$D/aux_targets_sangam_lead_0521" \
  --i1-cache "$D/cache/i1_tok_cache_DriverB_new_prev20.npz" \
  --aux-dir "$D/aux_targets_DriverB_new" \
  --i1-cache "$D/cache/i1_tok_cache_spd_v2_prev20.npz" \
  --aux-dir "$D/aux_targets_spd_fx640" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_pertO_v2_prev20.npz" \
  --aux-dir "$D/aux_targets_sangam_fx640" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_pertX_v2_prev20.npz" \
  --aux-dir "$D/aux_targets_sangam_fx640" \
  --run-filter "" --run-filter "" --run-filter "" --run-filter "" --run-filter "pertO" \
  --resume-from "$CKPT_INIT" \
  --d-out 128 --n-ch 2 --n-sa 2 \
  --epochs 12 \
  --batch-size 256 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --workers 4 \
  --val-ratio 0.05 \
  --w-obj 1.0 --w-reg 1.0 \
  --use-focal --focal-alpha 0.75 --focal-gamma 2.0 \
  --scale-dist 40.0 \
  --lds \
  --name a2_single_objdist_v1 \
  --out-dir outputs_multi \
  > "$LOG" 2>&1 &
PID=$!
echo "a_2 single-objdist pid=$PID log=$LOG"
echo "$PID" > /tmp/a2_single_objdist.pid

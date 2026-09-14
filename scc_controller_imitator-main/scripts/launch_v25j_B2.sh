#!/bin/bash
# v25j-B2: Phase 2b — B 모델 variant 2 (close_weight 7, traj_zero 25)
#   B1 보다 더 강한 close-lead 가중 + 더 긴 lookahead (2.5s)
set -e
LOG=/tmp/v25j_B2.log
CKPT_INIT=/home/ailab-12/git/scc_controller_imitator/data/ckpts/iy_v25h_B2_best_loss.pt
source /home/ailab-12/miniforge3/etc/profile.d/conda.sh
conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
CI=/home/ailab-12/AILabDatasetLocal/02_Custom_Dataset/35_e2e_morai/controller_imitator

nohup python -u scripts/train_iy.py \
  --resume-from "$CKPT_INIT" \
  --prebake "$D/prebake/f1_prebake_iy_sangam_lead_0520_v1_brakeaug_prev20.npz" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0520_prev20.npz" \
  --i2-cache "$D/cache/i2_mem_cache_sangam_lead_0520_v7b_prev20.npz" \
  --prebake-extra \
    "$D/prebake/f1_prebake_iy_sangam_lead_0521_v1_brakeaug_prev20.npz" \
    "$D/prebake/f1_prebake_iy_sangam_pertO_v2_prev20.npz" \
    "$D/prebake/f1_prebake_iy_sangam_pertX_v2_prev20.npz" \
    "$D/prebake/f1_prebake_iy_DriverB_new_logged_fixed_prev20.npz" \
    "$D/prebake/f1_prebake_iy_spd_streaming_brakeaug_prev20.npz" \
  --i1-cache-extra \
    "$D/cache/i1_tok_cache_sangam_lead_0521_prev20.npz" \
    "$D/cache/i1_tok_cache_sangam_pertO_v2_prev20.npz" \
    "$D/cache/i1_tok_cache_sangam_pertX_v2_prev20.npz" \
    "$D/cache/i1_tok_cache_DriverB_new_prev20.npz" \
    "$D/cache/i1_tok_cache_spd_v2_prev20.npz" \
  --i2-cache-extra \
    "$D/cache/i2_mem_cache_sangam_lead_0521_v7b_prev20.npz" \
    "$D/cache/i2_mem_cache_sangam_pertO_v2_v7b_prev20.npz" \
    "$D/cache/i2_mem_cache_sangam_pertX_v2_v7b_prev20.npz" \
    "$D/cache/i2_mem_cache_DriverB_new_v7b_prev20.npz" \
    "$D/cache/i2_mem_cache_spd_v2_v7b_prev20.npz" \
  --prebake-extra-limits 0 500000 500000 0 0 \
  --ci-aug-dirs "$CI/controller_imitator_data_multi_valid_v0237" \
  --ci-aug-limit 100000 \
  --main-no-image --side-cross-attn --side-num-layers 4 --side-dim-feedforward 512 \
  --target-first-only --traj-zero-after 25 \
  --phase1-epochs 30 \
  --main-lr-phase2 1.5e-5 --side-lr 3e-4 \
  --lr 3e-4 --lr-prof 1.5e-5 \
  --close-thr 30.0 --close-weight 7.0 \
  --fx 640 \
  --epochs 30 --batch-size 256 --workers 6 --val-ratio 0.05 \
  --name v25j_B2_v7b_close7_traj25 \
  --out-dir outputs_multi/iy \
  > "$LOG" 2>&1 &
echo $! > /tmp/v25j_B2.pid
echo "v25j-B2 pid=$(cat /tmp/v25j_B2.pid) log=$LOG"

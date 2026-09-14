#!/bin/bash
# v25h-A: 버전 A — 정통 (main image-aware self-attn, side 없음)
#   main = traj + ego + image_mem (i_2 cache) concat → self-attn (controller_imitator + image 확장)
#   side = 없음
#   init = v25c best (text-only baseline) — image_mem 추가 학습으로 4-way self-attn 형성
set -e
LOG=/tmp/v25h_A.log
CKPT_INIT=/home/ailab-12/git/scc_controller_imitator/data/ckpts/iy_v25c_best_loss.pt
source /home/ailab-12/miniforge3/etc/profile.d/conda.sh
conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
CI=/home/ailab-12/AILabDatasetLocal/02_Custom_Dataset/35_e2e_morai/controller_imitator

nohup python -u scripts/train_iy.py \
  --resume-from "$CKPT_INIT" \
  --prebake "$D/prebake/f1_prebake_iy_sangam_lead_0520_v1_brakeaug_prev20.npz" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0520_prev20.npz" \
  --i2-cache "$D/cache/i2_mem_cache_sangam_lead_0520_v6_prev20.npz" \
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
    "$D/cache/i2_mem_cache_sangam_lead_0521_v6_prev20.npz" \
    "$D/cache/i2_mem_cache_sangam_pertO_v2_v6_prev20.npz" \
    "$D/cache/i2_mem_cache_sangam_pertX_v2_v6_prev20.npz" \
    "$D/cache/i2_mem_cache_DriverB_new_v6_prev20.npz" \
    "$D/cache/i2_mem_cache_spd_v2_v6_prev20.npz" \
  --prebake-extra-limits 0 500000 500000 0 0 \
  --ci-aug-dirs "$CI/controller_imitator_data_multi_valid_v0237" \
  --ci-aug-limit 100000 \
  --lr 3e-4 --lr-prof 3e-5 \
  --close-thr 30.0 --close-weight 3.0 \
  --fx 640 \
  --epochs 30 --batch-size 256 --workers 6 --val-ratio 0.05 \
  --wandb --wandb-run v25h_A_main_image_aware \
  --name v25h_A_main_image_aware \
  --out-dir outputs_multi/iy \
  > "$LOG" 2>&1 &
PID=$!
echo "v25h-A pid=$PID log=$LOG"
echo $PID > /tmp/v25h_A.pid

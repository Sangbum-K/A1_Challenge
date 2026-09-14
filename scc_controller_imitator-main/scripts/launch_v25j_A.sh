#!/bin/bash
# v25j-A: Phase 1 — vL_rel reg fix 효과 검증 (A 모델 빠른 학습)
#   v25h-A 와 동일 config, but i_2 cache 만 v7b 로 교체 (a_2 v7b: reg fix + focal α=0.75)
#   목적: a_2 cache 차이로 val 이 개선되는지 (= reg fix 가 학습에 효과 있나) 검증
#   비교 baseline: v25h-A best 0.1649 (v6 cache), v25i-A best 0.1437 (v6 + obj_gate)
set -e
LOG=/tmp/v25j_A.log
CKPT_INIT=/home/ailab-12/git/scc_controller_imitator/data/ckpts/iy_v25h_A_best_loss.pt
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
  --lr 3e-4 --lr-prof 3e-5 \
  --close-thr 30.0 --close-weight 3.0 \
  --ttc-augment \
  --ttc-augment-steps 10 \
  --ttc-augment-max-brake 20.0 \
  --ttc-augment-dist-max 50.0 \
  --ttc-augment-ttc-max 4.0 \
  --ttc-augment-safety-margin 2.0 \
  --fx 640 \
  --epochs 20 --batch-size 256 --workers 6 --val-ratio 0.05 \
  --name v25j_A_v7b_cache \
  --out-dir outputs_multi/iy \
  > "$LOG" 2>&1 &
echo $! > /tmp/v25j_A.pid
echo "v25j-A pid=$(cat /tmp/v25j_A.pid) log=$LOG"

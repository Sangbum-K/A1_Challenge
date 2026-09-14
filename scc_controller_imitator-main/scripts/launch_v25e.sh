#!/bin/bash
# v25e: f_2 (Side SCC) 본격 학습 — ACC lead car 데이터 다수 + f_1 long freeze
#
# 변경점 (v25d → v25e):
#   - 데이터: sangam_lead_0520/0521 + pertO/X + DriverB_new + spd_v2 (6 prebake, 모두 prev20)
#     기존 v25d 의 ci_aug v0237/v0301/v0302 (waypoint only) 는 ACC 학습에 부적합
#     → ci_aug 는 limit 100k 로 축소 (α gate "lead 없으면 비활성" 학습용)
#   - phase 구성: phase1 5 → 45 ep (f_1 frozen), phase2 35 → 5 ep (joint fine-polish)
#     User 요구: "main 은 훨씬 많이 freez하고 훈련" + "앞차 인지 없으면 side 동작 안 함"
#   - cache 통일: 모두 _prev20 변종 (PREV_DTICK=20 align)
#
# init = v25c best (f_1 waypoint baseline). f_2 (side_cross_attn + a_2 feat + TTC) 학습.

set -e
LOG=/tmp/v25e.log
CKPT_INIT=/home/ailab-12/git/scc_controller_imitator/data/ckpts/iy_v25c_best_loss.pt
[ -f "$CKPT_INIT" ] || { echo "ERR: $CKPT_INIT 없음"; exit 1; }

source /home/ailab-12/miniforge3/etc/profile.d/conda.sh
conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator

D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
CI=/home/ailab-12/AILabDatasetLocal/02_Custom_Dataset/35_e2e_morai/controller_imitator

nohup python -u scripts/train_iy.py \
  --resume-from "$CKPT_INIT" \
  --prebake     "$D/prebake/f1_prebake_iy_sangam_lead_0520_v1_brakeaug_prev20.npz" \
  --i1-cache    "$D/cache/i1_tok_cache_sangam_lead_0520_prev20.npz" \
  --a2-cache    "$D/cache/a2_out_cache_sangam_lead_0520_prev20.npz" \
  --prebake-extra \
    "$D/prebake/f1_prebake_iy_sangam_lead_0521_v1_brakeaug_prev20.npz" \
    "$D/prebake/f1_prebake_iy_sangam_pertO_v2_prev20.npz" \
    "$D/prebake/f1_prebake_iy_sangam_pertX_v2_prev20.npz" \
    "$D/prebake/f1_prebake_iy_DriverB_new_logged_clipped_prev20.npz" \
    "$D/prebake/f1_prebake_iy_spd_streaming_brakeaug_prev20.npz" \
  --i1-cache-extra \
    "$D/cache/i1_tok_cache_sangam_lead_0521_prev20.npz" \
    "$D/cache/i1_tok_cache_sangam_pertO_v2_prev20.npz" \
    "$D/cache/i1_tok_cache_sangam_pertX_v2_prev20.npz" \
    "$D/cache/i1_tok_cache_DriverB_new_prev20.npz" \
    "$D/cache/i1_tok_cache_spd_v2_prev20.npz" \
  --a2-cache-extra \
    "$D/cache/a2_out_cache_sangam_lead_0521_prev20.npz" \
    "$D/cache/a2_out_cache_sangam_pertO_v2_prev20.npz" \
    "$D/cache/a2_out_cache_sangam_pertX_v2_prev20.npz" \
    "$D/cache/a2_out_cache_DriverB_new_prev20.npz" \
    "$D/cache/a2_out_cache_spd_v2_prev20.npz" \
  --ci-aug-dirs "$CI/controller_imitator_data_multi_valid_v0237" \
  --ci-aug-limit 100000 \
  --main-no-image \
  --side-cross-attn \
  --use-a2-feat \
  --ttc-augment \
  --ttc-augment-steps 10 \
  --ttc-augment-max-brake 20.0 \
  --ttc-augment-dist-max 50.0 \
  --ttc-augment-ttc-max 4.0 \
  --ttc-augment-safety-margin 2.0 \
  --under-brake-weight 1.0 \
  --under-brake-thr -2.0 \
  --traj-zero-after 15 \
  --phase1-epochs 45 \
  --main-lr-phase2 1.5e-5 \
  --side-lr 3e-4 \
  --lr 3e-4 --lr-prof 1.5e-5 \
  --close-thr 30.0 --close-weight 3.0 \
  --fx 640 \
  --epochs 50 --batch-size 512 --workers 6 --val-ratio 0.05 \
  --wandb --wandb-run v25e_side_a2_ttc_long_freeze_multi_acc \
  --name v25e_side_a2_ttc_long_freeze_multi_acc \
  --out-dir /home/ailab-12/git/scc_controller_imitator/outputs_multi/iy \
  > "$LOG" 2>&1 &

PID=$!
echo "v25e launched pid=$PID log=$LOG"
echo "$PID" > /tmp/v25e.pid

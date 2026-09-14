#!/bin/bash
# v25f: ACC 데이터 정제 후 f_2 (Side SCC) 학습
#
# 변경점 (v25e → v25f):
#   - DriverB prebake fixed (이전 target=0 버그 → 재생성)
#   - pertO/pertX 각 limit 500k (전체 3M+ 의 1/6, perturbed waypoint dominate 방지)
#   - sangam_lead 0520/0521 + spd + DriverB 는 전체 사용 (ACC 핵심 데이터)
#   - per-prebake limit (--prebake-extra-limits) 적용
#
# 데이터 (예상):
#   main: sangam_lead_0520 72k (전체)
#   extra: sangam_lead_0521 97k, pertO 500k, pertX 500k, DriverB 135k (fixed), spd 339k
#   ci_aug: 100k
#   total = ~1.74M

set -e
LOG=/tmp/v25f.log
CKPT_INIT=/home/ailab-12/git/scc_controller_imitator/data/ckpts/iy_v25c_best_loss.pt
DRIVERB_FIXED=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/prebake/f1_prebake_iy_DriverB_new_logged_fixed_prev20.npz

[ -f "$CKPT_INIT" ] || { echo "ERR: $CKPT_INIT 없음"; exit 1; }
[ -f "$DRIVERB_FIXED" ] || { echo "ERR: DriverB fixed prebake 아직 생성 안 됨 ($DRIVERB_FIXED). DriverB prebake 완료 대기 필요"; exit 1; }

source /home/ailab-12/miniforge3/etc/profile.d/conda.sh
conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator

D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
CI=/home/ailab-12/AILabDatasetLocal/02_Custom_Dataset/35_e2e_morai/controller_imitator

nohup python -u scripts/train_iy.py \
  --resume-from "$CKPT_INIT" \
  --prebake     "$D/prebake/f1_prebake_iy_sangam_lead_0520_v1_brakeaug_prev20.npz" \
  --i1-cache    "$D/cache/i1_tok_cache_sangam_lead_0520_prev20.npz" \
  --a2-cache    "$D/cache/a2_out_cache_sangam_lead_0520_prev20_fixed.npz" \
  --prebake-extra \
    "$D/prebake/f1_prebake_iy_sangam_lead_0521_v1_brakeaug_prev20.npz" \
    "$D/prebake/f1_prebake_iy_sangam_pertO_v2_prev20.npz" \
    "$D/prebake/f1_prebake_iy_sangam_pertX_v2_prev20.npz" \
    "$DRIVERB_FIXED" \
    "$D/prebake/f1_prebake_iy_spd_streaming_brakeaug_prev20.npz" \
  --i1-cache-extra \
    "$D/cache/i1_tok_cache_sangam_lead_0521_prev20.npz" \
    "$D/cache/i1_tok_cache_sangam_pertO_v2_prev20.npz" \
    "$D/cache/i1_tok_cache_sangam_pertX_v2_prev20.npz" \
    "$D/cache/i1_tok_cache_DriverB_new_prev20.npz" \
    "$D/cache/i1_tok_cache_spd_v2_prev20.npz" \
  --a2-cache-extra \
    "$D/cache/a2_out_cache_sangam_lead_0521_prev20_fixed.npz" \
    "$D/cache/a2_out_cache_sangam_pertO_v2_prev20_fixed.npz" \
    "$D/cache/a2_out_cache_sangam_pertX_v2_prev20_fixed.npz" \
    "$D/cache/a2_out_cache_DriverB_new_prev20_fixed.npz" \
    "$D/cache/a2_out_cache_spd_v2_prev20_fixed.npz" \
  --prebake-extra-limits 0 500000 500000 0 0 \
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
  --wandb --wandb-run v25f_side_acc_pertlimit_driverb_fixed \
  --name v25f_side_acc_pertlimit_driverb_fixed \
  --out-dir /home/ailab-12/git/scc_controller_imitator/outputs_multi/iy \
  > "$LOG" 2>&1 &

PID=$!
echo "v25f launched pid=$PID log=$LOG"
echo "$PID" > /tmp/v25f.pid

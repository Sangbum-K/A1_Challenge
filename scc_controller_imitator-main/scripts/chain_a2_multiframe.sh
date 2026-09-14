#!/bin/bash
# a_2 multi-frame 실험 순차 체인 (GPU 동시 점유 회피).
#   1) P1 2f yaw ON   (a2_2f_yaw)        — OFF 는 이미 별도 실행
#   2) P2 3f yaw OFF  (a2_multiframe_3f)
#   3) P2 4f yaw OFF  (a2_multiframe_4f)
# 각 학습은 train_a2_multiframe.py 를 foreground 로 호출 (체인 자체가 nohup background).
set -e
source /home/ailab-12/miniforge3/etc/profile.d/conda.sh
conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
P=$D/prebake
CKPT_INIT=data/ckpts/a2_v7b_d128_alpha075_best_loss.pt

run_one () {
  NAME=$1; NF=$2; YAW=$3
  echo "=== [chain] start $NAME (nf=$NF yaw=$YAW) $(date) ==="
  python -u scripts/train_a2_multiframe.py \
    --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0520_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0520" \
    --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0521_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0521" \
    --i1-cache "$D/cache/i1_tok_cache_DriverB_new_prev20.npz" --aux-dir "$D/aux_targets_DriverB_new" \
    --i1-cache "$D/cache/i1_tok_cache_spd_v2_prev20.npz" --aux-dir "$D/aux_targets_spd_nolead_0520" \
    --i1-cache "$D/cache/i1_tok_cache_sangam_pertO_v2_prev20.npz" --aux-dir "$D/aux_targets_sangam_fx640" \
    --i1-cache "$D/cache/i1_tok_cache_sangam_pertX_v2_prev20.npz" --aux-dir "$D/aux_targets_sangam_fx640" \
    --run-filter "" --run-filter "" --run-filter "" --run-filter "" --run-filter "pertO" --run-filter "pertX" \
    --prebake "$P/f1_prebake_iy_sangam_lead_0520_a2inlane_prev20.npz" \
    --prebake "$P/f1_prebake_iy_sangam_lead_0521_a2inlane_prev20.npz" \
    --prebake "$P/f1_prebake_iy_DriverB_new_a2inlane_prev20.npz" \
    --prebake "$P/f1_prebake_iy_sangam_pertO_v2_a2inlane_prev20.npz" \
    --prebake "$P/f1_prebake_iy_sangam_pertX_v2_a2inlane_prev20.npz" \
    --n-frames "$NF" --yaw-shift "$YAW" \
    --resume-from "$CKPT_INIT" \
    --d-out 128 --epochs 12 --batch-size 192 --lr 3e-4 --weight-decay 1e-4 \
    --workers 4 --val-ratio 0.05 \
    --w-obj 1.0 --w-reg 1.0 --use-focal --focal-alpha 0.75 --focal-gamma 2.0 \
    --scale-v 20.0 --scale-dist 40.0 --lds --fx 640 \
    --name "$NAME" --out-dir outputs_multi
  echo "=== [chain] done $NAME $(date) ==="
}

run_one a2_2f_yaw 2 on
run_one a2_multiframe_3f 3 off
run_one a2_multiframe_4f 4 off
echo "=== [chain] ALL DONE $(date) ==="

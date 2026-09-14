#!/bin/bash
# a_2 multi-frame / yaw_shift-ablation 학습 (2026-05-26 plan P1+P2).
#   v7b 와 동일 datasets/split(seed=0)/recipe 로 공정 비교.
#   사용: bash launch_a2_multiframe.sh <name> <n_frames> <yaw on|off>
#   예:   bash launch_a2_multiframe.sh a2_2f_noyaw   2 off   # P1 OFF
#         bash launch_a2_multiframe.sh a2_2f_yaw     2 on    # P1 ON
#         bash launch_a2_multiframe.sh a2_multiframe_3f 3 <yaw>  # P2
#         bash launch_a2_multiframe.sh a2_multiframe_4f 4 <yaw>  # P2
set -e
NAME=${1:?name}; NF=${2:?n_frames}; YAW=${3:?on|off}
LOG=/tmp/${NAME}.log
CKPT_INIT=/home/ailab-12/git/scc_controller_imitator/data/ckpts/a2_v7b_d128_alpha075_best_loss.pt
source /home/ailab-12/miniforge3/etc/profile.d/conda.sh
conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
P=$D/prebake

nohup python -u scripts/train_a2_multiframe.py \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0520_prev20.npz" \
  --aux-dir "$D/aux_targets_sangam_lead_0520" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0521_prev20.npz" \
  --aux-dir "$D/aux_targets_sangam_lead_0521" \
  --i1-cache "$D/cache/i1_tok_cache_DriverB_new_prev20.npz" \
  --aux-dir "$D/aux_targets_DriverB_new" \
  --i1-cache "$D/cache/i1_tok_cache_spd_v2_prev20.npz" \
  --aux-dir "$D/aux_targets_spd_nolead_0520" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_pertO_v2_prev20.npz" \
  --aux-dir "$D/aux_targets_sangam_fx640" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_pertX_v2_prev20.npz" \
  --aux-dir "$D/aux_targets_sangam_fx640" \
  --run-filter "" --run-filter "" --run-filter "" --run-filter "" --run-filter "pertO" --run-filter "pertX" \
  --prebake "$P/f1_prebake_iy_sangam_lead_0520_a2inlane_prev20.npz" \
  --prebake "$P/f1_prebake_iy_sangam_lead_0521_a2inlane_prev20.npz" \
  --prebake "$P/f1_prebake_iy_DriverB_new_a2inlane_prev20.npz" \
  --prebake "$P/f1_prebake_iy_sangam_pertO_v2_a2inlane_prev20.npz" \
  --prebake "$P/f1_prebake_iy_sangam_pertX_v2_a2inlane_prev20.npz" \
  --n-frames "$NF" --yaw-shift "$YAW" \
  --resume-from "$CKPT_INIT" \
  --d-out 128 \
  --epochs 12 \
  --batch-size 192 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --workers 4 \
  --val-ratio 0.05 \
  --w-obj 1.0 --w-reg 1.0 \
  --use-focal --focal-alpha 0.75 --focal-gamma 2.0 \
  --scale-v 20.0 --scale-dist 40.0 \
  --lds \
  --fx 640 \
  --name "$NAME" \
  --out-dir outputs_multi \
  > "$LOG" 2>&1 &
PID=$!
echo "a_2 multiframe '$NAME' (nf=$NF yaw=$YAW) pid=$PID log=$LOG"
echo "$PID" > /tmp/${NAME}.pid

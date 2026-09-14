#!/usr/bin/env bash
# 배포 모델 재현: prof635 + distbrake_v1 (= data/ckpts/prof635_side_distbrake_v1.pt).
#   distbrake 라벨(거리-onset 곡선, closing 비의존) + delta_mode=brake_only (delta≤0, 제동만).
#   train default 는 --pedal-suffix ttc / signed 가 아님에 주의 — 아래 두 인자 반드시 명시.
# 5 datasets default (sangam_lead_0520/0521 + pertO_v2 + pertX_v2 + DriverB_new, ~975K).
# 출력: outputs_multi/prof635_side_distbrake_v1/best.pt, 로그 /tmp/prof635_distbrake.log
set -e
cd /home/ailab-12/git/scc_controller_imitator
PY=/home/ailab-12/miniforge3/envs/ml_con/bin/python
"$PY" -u scripts/train_prof635_side.py \
    --name prof635_side_distbrake_v1 \
    --pedal-suffix distbrake --delta-mode brake_only \
    --epochs 25 --batch-size 512 --lr 3e-4 --wandb \
    2>&1 | tee /tmp/prof635_distbrake.log

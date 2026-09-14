#!/usr/bin/env bash
# prof635 v6 = v5 구조(residual + no-traj, gated) + DriverB 포함 5 datasets 재학습.
# 주의: v6 는 LEGACY (signed/ttc). 현행 배포는 distbrake_v1(brake_only) = scripts/launch_prof635_distbrake.sh.
#   이 launcher 는 --pedal-suffix/--delta-mode 미지정 → train default(ttc/signed)로 학습되어 배포 ckpt 와 다른 모델 생성.
#   배포 모델 재현이 목적이면 이 스크립트가 아니라 launch_prof635_distbrake.sh 를 실행할 것.
# 다음 기회에 바로 실행: bash scripts/launch_prof635_v6_driverb.sh
# DATASETS(train_prof635_side.py) default 에 DriverB_new 포함 → 인자 없이 5개 전부 학습.
#   sangam_lead_0520/0521 + pertO_v2 + pertX_v2 + DriverB_new (~975K samples)
# 출력: outputs_multi/prof635_side_v6_driverb/best.pt, 로그 /tmp/prof635_v6.log
set -e
cd /home/ailab-12/git/scc_controller_imitator
PY=/home/ailab-12/miniforge3/envs/ml_con/bin/python
"$PY" -u scripts/train_prof635_side.py \
    --name prof635_side_v6_driverb \
    --epochs 25 --batch-size 512 --lr 3e-4 --wandb \
    2>&1 | tee /tmp/prof635_v6.log

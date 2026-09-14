#!/usr/bin/env bash
# prof635 + learned NN gate (gate_head MLP) 재학습.
#   배포 ckpt(prof635_side_distbrake_v1.pt) warm-start → f_2 side + gate_head MLP joint 학습.
#   main(#654) frozen. rule gate(threshold/IIR/EMA/distill/beat) 전면 제거 — gate = NN forward output.
#   gradient: pedal L1 loss → final_pedal(=base + g·delta) → g → gate_head MLP 까지 흐름.
# 라벨/recipe 는 배포 distbrake 와 동일 (distbrake suffix, brake_only).
# 출력: outputs_multi/prof635_side_nngate_v1/best.pt
set -e
cd /home/ailab-12/git/scc_controller_imitator
PY=/home/ailab-12/miniforge3/envs/ml_con/bin/python
"$PY" -u scripts/train_prof635_side.py \
    --name prof635_side_nngate_v1 \
    --resume data/ckpts/prof635_side_distbrake_v1.pt \
    --pedal-suffix distbrake --delta-mode brake_only \
    --epochs 20 --batch-size 128 --lr 1e-4 \
    --gate-warm-thresh 0.3 --gate-warm-scale 5.0 \
    2>&1 | tee /tmp/prof635_nngate.log

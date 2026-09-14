#!/usr/bin/env bash
# prof635 + learned NN gate(gate_head MLP) — gate 입력 obj(10) + vrel/closing(10) = 20dim 재학습.
#   목적: obj-only gate(b안 prof635_side_nngate_bce_v1) 한계 해소 — receding(멀어지는 lead)에도
#         obj>0 라 gate 가 켜짐. closing(=-vL_rel, 양수=접근) 입력 추가 → "lead 있고 AND 다가옴"일 때만 open.
#   gate 입력 = a_2 in-lane(16-25) obj logit(10) + obj_prob-masked closing(10). sign 정합:
#     approach(closing>0) → 입력>0 → GELU → gate logit↑ → g↑(제동). receding(closing<0) → 닫힘.
#   f_2 에도 vrel 주입 유지(a-pred, --gate-bce-keep-vrel + --vrel-from-a2logits) — closing 비례 제동.
#   gate target g* = inlane_gt(GT): dist finite(lead) & closing>gate_tau(0.5). (b안과 동일 target)
#   warm-start: prof635_side_distbrake_v1.pt (gate_head 신규 → strict=False, 새 20dim warm-init).
#   main(#654) frozen. closed_loop 무수정 — model.forward 가 a2_logits 에서 obj+vrel 모두 내부 추출.
# 출력: outputs_multi/prof635_side_nngate_objvrel_v1/best.pt
set -e
cd /home/ailab-12/git/scc_controller_imitator
PY=/home/ailab-12/miniforge3/envs/ml_con/bin/python
"$PY" -u scripts/train_prof635_side.py \
    --name prof635_side_nngate_objvrel_v1 \
    --resume data/ckpts/prof635_side_distbrake_v1.pt \
    --pedal-suffix distbrake --delta-mode brake_only --randomize-base \
    --gate-bce --gate-bce-keep-vrel --vrel-from-a2logits \
    --gate-tau 0.5 --w-gate 1.0 \
    --epochs 25 --batch-size 128 --lr 1e-4 \
    --gate-warm-thresh 0.3 --gate-warm-scale 0.3 --gate-warm-vrel-scale 2.0 \
    2>&1 | tee /tmp/prof635_nngate_objvrel.log

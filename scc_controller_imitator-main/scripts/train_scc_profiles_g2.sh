#!/usr/bin/env bash
# SCC 5-profile f_2 학습 (각 profile 독립, distbrake warm-start).
#   gate = 2*sigmoid(obj+vrel) — gate-bce-keep-vrel + vrel-from-a2logits.
#   pedal_label_<ds>_p{N} (inlane_gt 물리감속 A=-(closing^2)/(2*(dist-s_min)) + min(base)).
#   순서: p3(중간) → p5(보수) → p2 → p4 → p1(공격) — 사용자 우선순위.
set -e
cd /home/ailab-12/git/scc_controller_imitator
PY=/home/ailab-12/miniforge3/envs/ml_con/bin/python
for p in p3 p5 p2 p4 p1; do
  echo "===== profile $p 학습 시작 $(date '+%H:%M:%S') ====="
  "$PY" -u scripts/train_prof635_side.py \
    --gate-bce --gate-bce-keep-vrel --vrel-from-a2logits --gate-tau 0.5 \
    --resume data/ckpts/prof635_side_distbrake_v1.pt \
    --delta-mode brake_only --randomize-base --epochs 25 \
    --pedal-suffix "$p" --name "prof635_side_scc_${p}_g2"
  echo "===== profile $p 완료 $(date '+%H:%M:%S') ====="
done
echo "===== ALL 5 PROFILE (p3,p5,p2,p4,p1) DONE ====="

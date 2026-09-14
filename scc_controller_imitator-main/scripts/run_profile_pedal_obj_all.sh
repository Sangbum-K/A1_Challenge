#!/usr/bin/env bash
# 5-profile pedal_label 생성: inlane_gt(obj dist/closing GT) 물리 감속 공식
# + base pedal(pedal_label_<ds>_ttc) 과 element-wise min(A<0 frame 만) 결합.
# scc.txt / EGO_INFO / synth_batch 일절 불필요 — inlane_gt + ttc cache 만 입력 (CPU numpy 벡터화).
# A_MIN=-40, profile s_min: p1=2.0 p2=3.0 p3=4.0 p4=5.5 p5=7.0.
#
# 대상: inlane_gt + pedal_label_*_ttc 가 모두 있는 6 ds × 5 profile = 30 개.
#   lead-bearing(차별화 발생): sangam_lead_0520/0521, DriverB_new, spd_v2.
#   no-lead(inlane_gt dist 전부 NaN → A=0 → target==base): sangam_pertO_v2, sangam_pertX_v2.
# 스크립트가 인자 없이 DSETS 전체를 순회하므로 아래 한 줄이면 충분.
set -euo pipefail
PY=${PY:-python3}
S=/home/ailab-12/git/scc_controller_imitator/scripts/gen_profile_pedal_labels_obj.py
cd /home/ailab-12/git/scc_controller_imitator

OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} \
MKL_NUM_THREADS=${MKL_NUM_THREADS:-2} "$PY" -u "$S"

echo "===== ALL DONE ====="

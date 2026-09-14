#!/usr/bin/env bash
# 5-profile pedal_label python 재계산 — 전 데이터셋 × p1..p5.
# EGO v0 cache (cache/ego_v0_<ds>.npz) 는 데이터셋별 첫 profile 에서 1회 생성, 이후 재사용.
# CPU only (GPU gate 학습 중). ego-workers=16 (CPU 2 reserve).
set -euo pipefail
PY=/home/ailab-12/miniforge3/envs/ml_con/bin/python
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
S=/home/ailab-12/git/scc_controller_imitator/scripts/gen_profile_pedal_labels_recalc.py
cd /home/ailab-12/git/scc_controller_imitator

# ds : prebake_stem : scc_dir
declare -a JOBS=(
  "sangam_lead_0520:f1_prebake_iy_sangam_lead_0520_v1_brakeaug_prev20:scc_synth_v3_sangam_lead_0520"
  "sangam_lead_0521:f1_prebake_iy_sangam_lead_0521_v1_brakeaug_prev20:scc_synth_v3_sangam_lead_0521"
  "spd_nolead_0520:f1_prebake_iy_spd_nolead_0520_v1:scc_synth_v3_spd_nolead_0520"
  "spd_nolead_0521:f1_prebake_iy_spd_nolead_0521_v1:scc_synth_v3_spd_nolead_0521"
  "DriverB_new:f1_prebake_iy_DriverB_new_logged_clipped_prev20:scc_synth_p1_DriverB_new"
)

for job in "${JOBS[@]}"; do
  IFS=':' read -r DS PB SCC <<< "$job"
  for P in p1 p2 p3 p4 p5; do
    echo "===== $DS / $P ====="
    $PY -u "$S" --ds "$DS" \
        --prebake "$D/prebake/$PB.npz" \
        --scc-dir "$D/$SCC" \
        --meta "$D/meta_$DS.json" \
        --profile "$P" --ego-workers 16
  done
done
echo "===== ALL DONE ====="

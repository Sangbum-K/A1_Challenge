#!/bin/bash
# 80bin GT 재생성 — bin80 ablation 용. prepare_a2_aux_targets --n-bins 80 (4-corner projection 재실행).
# raw_base = AILabSSD (EGO_INFO/OBJECT_INFO 소스). out = AILabDatasetLocal aux_targets_*_bin80.
# SLURM 가능 시 이 스크립트를 sbatch wrap. 현재는 로컬 CPU(ProcessPool).
set -e
source /home/ailab-12/miniforge3/etc/profile.d/conda.sh; conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator
RAW=/home/ailab-12/AILabSSD/02_Data/57_A1_E2E/00_raw
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data

for ds in sangam_lead_0520 sangam_lead_0521 spd_nolead_0520; do
  echo "### GT bin80: $ds $(date)"
  python -u scripts/prepare_a2_aux_targets.py \
    --meta "$D/meta_${ds}.json" --raw-base "$RAW" \
    --out-base "$D/aux_targets" --out-tag "${ds}_bin80" \
    --n-bins 80 --fx 640 --cx 640
done
echo "### GT bin80 ALL DONE $(date)"

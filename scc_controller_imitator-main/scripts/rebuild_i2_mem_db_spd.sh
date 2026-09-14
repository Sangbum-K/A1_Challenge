#!/usr/bin/env bash
# Step1: sangam 5종 i2_mem_mf20 을 db_spd a_2 로 재생성 (기존 db i2_mem 덮음).
# a_2 = a2_mf20_v2s_n20_db_spd.pt (speedway 인지 포함, val 0.0317).
# prev-offsets = default(5,10,...,100) → 생략. n_frames=21.
set -euo pipefail
PY=/home/ailab-12/miniforge3/envs/ml_con/bin/python
R=/home/ailab-12/git/scc_controller_imitator
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
A2=$R/data/ckpts/a2_mf20_v2s_n20_db_spd.pt
cd $R

for ds in sangam_lead_0520 sangam_lead_0521 sangam_pertO_v2 sangam_pertX_v2 DriverB_new; do
  echo "===== i2_mem $ds ====="
  $PY -u scripts/build_i2_cache_mf20.py \
    --i1-cache $D/cache/i1_tok_cache_${ds}_prev20.npz \
    --out $D/cache/i2_mem_mf20_${ds}.npz \
    --a2-ckpt $A2 \
    --d-token 256 --d-out 128
done
echo "===== STEP1 ALL DONE ====="

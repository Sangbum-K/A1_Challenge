#!/bin/bash
# a_2 mf20 재학습 v3 (db + speedway lead spd_0522_40m 추가) — perception only.
#   speedway 인지(i1_cache+aux_targets) 포함해 a_2 일반화 보강 (held-out closing 0.49 약점).
#   f_2(제어)는 sangam 만 학습 (사용자 결정) — speedway scc pedal 미사용.
#   warm = db best (mf20f_v2s_5to100_db, 4-set). + spd_0522_40m 5번째 set. recipe 동일.
set -e
source /home/ailab-12/miniforge3/etc/profile.d/conda.sh
conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
CKPT_INIT=outputs_multi/a2_b0_abl/mf20f_v2s_5to100_db/best_loss.pt   # db best(4-set) warm
OUT=outputs_multi/a2_b0_abl
OFFS="5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100"

python -u scripts/train_a2_multiframe.py \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0520_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0520" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0521_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0521" \
  --i1-cache "$D/cache/i1_tok_cache_spd_v2_prev20.npz"           --aux-dir "$D/aux_targets_spd_nolead_0520" \
  --i1-cache "$D/cache/i1_tok_cache_DriverB_new_prev20.npz"      --aux-dir "$D/aux_targets_DriverB_new" \
  --i1-cache "$D/cache/i1_tok_cache_spd_0522_40m_prev20.npz"     --aux-dir "$D/aux_targets_spd_0522_40m" \
  --run-filter "" --run-filter "" --run-filter "" --run-filter "" --run-filter "" \
  --prev-offsets "$OFFS" \
  --yaw-shift off \
  --resume-from "$CKPT_INIT" \
  --d-token 256 --d-out 128 \
  --epochs 12 --batch-size 32 --lr 1e-4 --weight-decay 1e-4 \
  --workers 4 --val-ratio 0.05 \
  --w-obj 1.0 --w-reg 1.0 --use-focal --focal-alpha 0.75 --focal-gamma 2.0 \
  --scale-v 20.0 --scale-dist 40.0 --lds --fx 640 --seed 0 \
  --name mf20f_v2s_5to100_db_spd --out-dir "$OUT"
echo "=== TRAIN DONE mf20f_v2s_5to100_db_spd $(date) ==="

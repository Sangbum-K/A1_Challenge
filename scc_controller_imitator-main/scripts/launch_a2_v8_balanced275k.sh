#!/bin/bash
# a_2 v8_balanced275k — 데이터 불균형 root fix 재학습
#   배경: a2_v7b 는 B1/B2 버그 join 으로 pertO·spd 0% 제외된 211K(DriverB 64% 편중) 학습.
#   fix:  B1(run-filter 파일명 전체 매칭) → pertO 30,593 합류
#         B2(spd_v2 cache + spd_nolead_0520 aux) → spd 34,498 합류
#   결과: balanced join 275,999 (pertO+spd 포함) 로 재학습. warm start = a2_v7b.
set -e
LOG=/tmp/a2_v8_balanced275k.log
CKPT_INIT=/home/ailab-12/git/scc_controller_imitator/data/ckpts/a2_v7b_d128_alpha075_best_loss.pt
source /home/ailab-12/miniforge3/etc/profile.d/conda.sh
conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data

nohup python -u scripts/train_a2_v5_from_cache.py \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0520_prev20.npz" \
  --aux-dir "$D/aux_targets_sangam_lead_0520" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0521_prev20.npz" \
  --aux-dir "$D/aux_targets_sangam_lead_0521" \
  --i1-cache "$D/cache/i1_tok_cache_DriverB_new_prev20.npz" \
  --aux-dir "$D/aux_targets_DriverB_new" \
  --i1-cache "$D/cache/i1_tok_cache_spd_v2_prev20.npz" \
  --aux-dir "$D/aux_targets_spd_nolead_0520" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_pertO_v2_prev20.npz" \
  --aux-dir "$D/aux_targets_sangam_fx640" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_pertX_v2_prev20.npz" \
  --aux-dir "$D/aux_targets_sangam_fx640" \
  --run-filter "" --run-filter "" --run-filter "" --run-filter "" --run-filter "pertO" --run-filter "pertX" \
  --resume-from "$CKPT_INIT" \
  --d-out 128 \
  --epochs 13 \
  --batch-size 256 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --workers 4 \
  --val-ratio 0.05 \
  --w-obj 1.0 --w-reg 1.0 \
  --use-focal --focal-alpha 0.75 --focal-gamma 2.0 \
  --scale-v 20.0 --scale-dist 40.0 \
  --fx 640 \
  --seed 0 \
  --name a2_v8_balanced275k \
  --out-dir outputs_multi/a2 \
  > "$LOG" 2>&1 &
PID=$!
echo "a_2 v8_balanced275k pid=$PID log=$LOG"
echo "$PID" > /tmp/a2_v8_balanced275k.pid

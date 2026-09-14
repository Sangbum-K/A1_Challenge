#!/bin/bash
# a_2 v8 — Gaussian 분포 추론 (interleaved mean+σ 13ch) + inverse-density weight (A1)
#   목적: vL_rel/dist 의 mean-collapse (tail→중앙 압축) 해소
#   - 출력: [obj, vL_abs,σ, vY_abs,σ, vL_rel,σ, vY_rel,σ, prox,σ, dist,σ] (13ch interleaved)
#   - loss: obj focal(α0.75) + reg Gaussian NLL + LDS(vL_rel+dist batch inv-density)
#   - resume: v7b (fc_proj+encoder+fusion 재사용, head 7→13 reinit)
set -e
LOG=/tmp/a2_v8_cache.log
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
  --aux-dir "$D/aux_targets_spd_fx640" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_pertO_v2_prev20.npz" \
  --aux-dir "$D/aux_targets_sangam_fx640" --run-filter "" --run-filter "" --run-filter "" --run-filter "" --run-filter "pertO" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_pertX_v2_prev20.npz" \
  --aux-dir "$D/aux_targets_sangam_fx640" \
  --resume-from "$CKPT_INIT" \
  --d-out 128 \
  --epochs 25 \
  --batch-size 256 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --workers 4 \
  --val-ratio 0.05 \
  --w-obj 1.0 --w-reg 1.0 \
  --use-focal --focal-alpha 0.75 --focal-gamma 2.0 \
  --gaussian-nll --lds --lds-nbins 20 \
  --scale-v 20.0 --scale-dist 40.0 \
  --fx 640 \
  --name v8_d128_gaussNLL_lds \
  --out-dir outputs_multi/a2 \
  > "$LOG" 2>&1 &
PID=$!
echo "a_2 v8 pid=$PID log=$LOG"
echo "$PID" > /tmp/a2_v8.pid

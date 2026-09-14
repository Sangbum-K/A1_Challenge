#!/bin/bash
# a_2 20-frame eval (val-split held-out 동일). 인자: <cache_kind b0|v2s> <ckpt> <out.json>
#   b0  -> b0 cache (C_b=192), v7b auto-skip.
#   v2s -> 원본 cache (C_b=256), v7b 포함.
set -e
source /home/ailab-12/miniforge3/etc/profile.d/conda.sh
conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
KIND=${1:?b0|v2s}; CKPT=${2:?ckpt}; OUT=${3:?out.json}
OFFS="5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100"
if [ "$KIND" = "b0" ]; then PREFIX=i1_tok_cache_b0_; SUBDIR=b0/; else PREFIX=i1_tok_cache_; SUBDIR=""; fi
python -u scripts/eval_a2_multiframe.py \
  --i1-cache "$D/cache/${SUBDIR}${PREFIX}sangam_lead_0520_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0520" \
  --i1-cache "$D/cache/${SUBDIR}${PREFIX}sangam_lead_0521_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0521" \
  --i1-cache "$D/cache/${SUBDIR}${PREFIX}spd_v2_prev20.npz"           --aux-dir "$D/aux_targets_spd_nolead_0520" \
  --run-filter "" --run-filter "" --run-filter "" \
  --v7b data/ckpts/a2_v7b_d128_alpha075_best_loss.pt \
  --mf n20 "$CKPT" 21 off "$OFFS" \
  --val-split --val-ratio 0.05 --seed 0 \
  --out "$OUT"
echo "=== EVAL DONE $KIND $OUT $(date) ==="

#!/bin/bash
# 분포 분리 실험: Exp1 hardneg(OHEM) / Exp2 margin / Exp12 both — v2s 5f recipe + loss 항.
set -e
source /home/ailab-12/miniforge3/etc/profile.d/conda.sh; conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
OUT=outputs_multi/a2_objsep; mkdir -p "$OUT"
INIT=data/ckpts/a2_v7b_d128_alpha075_best_loss.pt
C=( --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0520_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0520"
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0521_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0521"
  --i1-cache "$D/cache/i1_tok_cache_spd_v2_prev20.npz" --aux-dir "$D/aux_targets_spd_nolead_0520"
  --run-filter "" --run-filter "" --run-filter "" --prev-offsets "10,20,30,40,50" --yaw-shift off
  --resume-from "$INIT" --d-token 256 --d-out 128 --epochs 12 --batch-size 192 --lr 1e-4 --weight-decay 1e-4
  --workers 4 --val-ratio 0.05 --w-obj 1.0 --w-reg 1.0 --use-focal --focal-alpha 0.75 --focal-gamma 2.0
  --scale-v 20.0 --scale-dist 40.0 --lds --fx 640 --seed 0 --out-dir "$OUT" )
echo "### [1/4] hardneg $(date)"; python -u scripts/train_a2_multiframe.py "${C[@]}" --name hardneg --hard-neg-w 1.0 --hard-neg-frac 0.1
echo "### [2/4] margin $(date)";  python -u scripts/train_a2_multiframe.py "${C[@]}" --name margin  --margin-w 0.5 --margin-m 2.0
echo "### [3/4] both $(date)";    python -u scripts/train_a2_multiframe.py "${C[@]}" --name both    --hard-neg-w 1.0 --margin-w 0.5 --margin-m 2.0
echo "### [4/4] eval $(date)"
python -u scripts/eval_a2_multiframe.py \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0520_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0520" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0521_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0521" \
  --i1-cache "$D/cache/i1_tok_cache_spd_v2_prev20.npz" --aux-dir "$D/aux_targets_spd_nolead_0520" \
  --run-filter "" --run-filter "" --run-filter "" \
  --v7b data/ckpts/a2_v7b_d128_alpha075_best_loss.pt --val-split --val-ratio 0.05 --seed 0 \
  --mf baseline outputs_multi/a2_b0_abl/mf5f_v2s3set_10_20_30_40_50/best_loss.pt 6 off "10,20,30,40,50" \
  --mf hardneg "$OUT/hardneg/best_loss.pt" 6 off "10,20,30,40,50" \
  --mf margin  "$OUT/margin/best_loss.pt"  6 off "10,20,30,40,50" \
  --mf both    "$OUT/both/best_loss.pt"    6 off "10,20,30,40,50" \
  --out "$OUT/eval_objsep.json"
echo "### ALL DONE $(date)"

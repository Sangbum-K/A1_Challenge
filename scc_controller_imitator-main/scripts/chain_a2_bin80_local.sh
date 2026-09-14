#!/bin/bash
# bin80 ablation 로컬 학습 (SLURM sif 미완 우회). token cache 라 로컬 ml_con 으로 충분.
# 80bin stride16 1D-merge × {10f, 20f}. cache_hr(tokens_s16) + GT bin80 + v7b resume.
set -e
source /home/ailab-12/miniforge3/etc/profile.d/conda.sh; conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # seg/agent 공존 OOM 방어
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
OUT=outputs_multi/a2_bin80; mkdir -p "$OUT"
INIT=data/ckpts/a2_v7b_d128_alpha075_best_loss.pt
CC=( --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0520_hr.npz" --aux-dir "$D/aux_targets_sangam_lead_0520_bin80"
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0521_hr.npz" --aux-dir "$D/aux_targets_sangam_lead_0521_bin80"
  --i1-cache "$D/cache/i1_tok_cache_spd_v2_hr.npz" --aux-dir "$D/aux_targets_spd_nolead_0520_bin80"
  --yaw-shift off --resume-from "$INIT" --hr --d-token 256 --d-out 128 --epochs 12 --lr 1e-4
  --weight-decay 1e-4 --workers 4 --val-ratio 0.05 --w-obj 1.0 --w-reg 1.0
  --use-focal --focal-alpha 0.75 --focal-gamma 2.0 --scale-v 20.0 --scale-dist 40.0
  --lds --fx 640 --seed 0 --out-dir "$OUT" )

echo "### bin80_s10n10 (1.0s,10f) bs48 $(date)"
python -u scripts/train_a2_multiframe.py "${CC[@]}" --batch-size 48 \
  --prev-offsets "10,20,30,40,50,60,70,80,90,100" --name bin80_s10n10
echo "### bin80_s10n20 (1.0s,20f) bs24 $(date)"
python -u scripts/train_a2_multiframe.py "${CC[@]}" --batch-size 24 \
  --prev-offsets "5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100" --name bin80_s10n20
echo "### bin80 ALL DONE $(date)"

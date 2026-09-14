#!/bin/bash
# a_2 20-frame ablation (b0 vs v2_s, 동일 recipe). 5f(train_a2_b0_5f.sh) 와 유일 차이:
#   --prev-offsets "5,10,...,100" (20 prev frame, n_frames=21).  batch 축소(메모리↑).
#   동일 3-set / focal a0.75 g2.0 / LDS 20bin / d-out128 / ep12 / lr1e-4 / wd1e-4 / fx640 / seed0
#   init = data/ckpts/a2_v7b_d128_alpha075_best_loss.pt
#   prev20 cache (tick stride=5) 가 offset 100 까지 커버 — 추가 cache 재생성 불필요.
set -e
source /home/ailab-12/miniforge3/etc/profile.d/conda.sh
conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
CKPT_INIT=data/ckpts/a2_v7b_d128_alpha075_best_loss.pt
OUT=outputs_multi/a2_b0_abl
mkdir -p "$OUT"

DTOK=${1:?d-token (b0=192, v2s=256)}
NAME=${2:?name}
PREFIX=${3-i1_tok_cache_b0_}   # b0: i1_tok_cache_b0_ ; v2s: i1_tok_cache_
SUBDIR=${4-b0/}                # b0: b0/ ; v2s: ""
BS=${5:-32}
OFFS="5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100"

python -u scripts/train_a2_multiframe.py \
  --i1-cache "$D/cache/${SUBDIR}${PREFIX}sangam_lead_0520_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0520" \
  --i1-cache "$D/cache/${SUBDIR}${PREFIX}sangam_lead_0521_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0521" \
  --i1-cache "$D/cache/${SUBDIR}${PREFIX}spd_v2_prev20.npz"           --aux-dir "$D/aux_targets_spd_nolead_0520" \
  --run-filter "" --run-filter "" --run-filter "" \
  --prev-offsets "$OFFS" \
  --yaw-shift off \
  --resume-from "$CKPT_INIT" \
  --d-token "$DTOK" --d-out 128 \
  --epochs 12 --batch-size "$BS" --lr 1e-4 --weight-decay 1e-4 \
  --workers 4 --val-ratio 0.05 \
  --w-obj 1.0 --w-reg 1.0 --use-focal --focal-alpha 0.75 --focal-gamma 2.0 \
  --scale-v 20.0 --scale-dist 40.0 --lds --fx 640 --seed 0 \
  --name "$NAME" --out-dir "$OUT"
echo "=== TRAIN DONE $NAME $(date) ==="

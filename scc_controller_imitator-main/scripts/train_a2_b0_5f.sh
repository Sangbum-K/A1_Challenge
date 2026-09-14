#!/bin/bash
# a_2-on-b0 ablation: b0 i1 cache 위에서 a_2 학습 (sweep 최고 조합 5f_10_20_30_40_50).
#   recipe = run_mf_sweep.sh 와 동일 (d-out128, focal0.75, lr1e-4, ep12, bs192, lds, fx640).
#   차이: i1-cache = b0(C_b=192), --d-token 192 (공정 비교: encoder.proj 192->192).
#   데이터 = 4-set subset (spd_v2 + sangam_lead_0520/0521 + DriverB_new).
#   v2_s 비교쌍 = outputs_multi/a2/mf_sweep_5f_10_20_30_40_50 (단, 그것은 6-set 학습).
#   → 공정 비교 위해 v2_s 도 동일 4-set 으로 재학습(train_a2_v2s_5f_4set.sh) 후 같은 held-out eval.
set -e
source /home/ailab-12/miniforge3/etc/profile.d/conda.sh
conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
CKPT_INIT=data/ckpts/a2_v7b_d128_alpha075_best_loss.pt
OUT=outputs_multi/a2_b0_abl
mkdir -p "$OUT"

CACHE_DIR=${1:-b0}        # b0 = b0 cache, v2s = 원본 256ch cache (비교쌍 재학습용)
DTOK=${2:-192}            # b0=192, v2s=256
NAME=${3:-mf5f_b0_10_20_30_40_50}
PREFIX=${4-i1_tok_cache_b0_}   # b0 cache prefix; v2s 면 "i1_tok_cache_" 로 원본 (빈문자 아님)
SUBDIR=${5-b0/}                # b0 cache subdir; v2s 면 "" (콜론없는 - : 빈문자 허용)

# 3-set subset (DriverB 제외 — AILabDataset NFS 가 느려 b0 cache 빌드 80분+ 소요).
#   spd_v2 + sangam_lead_0520 + sangam_lead_0521 = 77,594 (lead-rich, obj/vrel/dist 검증에 충분).
#   sweep 와 동일 cache 순서 유지 (eval val-split 재현). spd 는 spd_v2 cache + aux_spd_nolead_0520.
python -u scripts/train_a2_multiframe.py \
  --i1-cache "$D/cache/${SUBDIR}${PREFIX}sangam_lead_0520_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0520" \
  --i1-cache "$D/cache/${SUBDIR}${PREFIX}sangam_lead_0521_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0521" \
  --i1-cache "$D/cache/${SUBDIR}${PREFIX}spd_v2_prev20.npz"           --aux-dir "$D/aux_targets_spd_nolead_0520" \
  --run-filter "" --run-filter "" --run-filter "" \
  --prev-offsets "10,20,30,40,50" \
  --yaw-shift off \
  --resume-from "$CKPT_INIT" \
  --d-token "$DTOK" --d-out 128 \
  --epochs 12 --batch-size 192 --lr 1e-4 --weight-decay 1e-4 \
  --workers 4 --val-ratio 0.05 \
  --w-obj 1.0 --w-reg 1.0 --use-focal --focal-alpha 0.75 --focal-gamma 2.0 \
  --scale-v 20.0 --scale-dist 40.0 --lds --fx 640 --seed 0 \
  --name "$NAME" --out-dir "$OUT"
echo "=== TRAIN DONE $NAME $(date) ==="

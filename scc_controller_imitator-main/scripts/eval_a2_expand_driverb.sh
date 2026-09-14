#!/bin/bash
# a_2 b0 vs v2_s 확대 held-out 재검증 (방식 a).
#   held-out = DriverB_new (135K, 두 a_2 ckpt 가 학습에 안 쓴 완전 held-out lead-rich set).
#   공정 비교: 동일 aux GT, 동일 eval 스크립트, b0=b0-cache(C_b192)+b0 ckpt / v2s=v2s-cache(C_b256)+v2s ckpt.
#   b0/v2s cache 가 동일 key 집합·순서 → sample 1:1 동일. 전수 eval(--val-split 없음).
set -e
source /home/ailab-12/miniforge3/etc/profile.d/conda.sh
conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
OUT=outputs_multi/a2_b0_abl
mkdir -p "$OUT/expand_driverb"

# 40K stride subset (b0/v2s 동일 key) — NFS 느려 전수 대신 균등 subset.
B0_CACHE="$D/cache/b0/i1_tok_cache_b0_DriverB_new_40k_prev20.npz"
V2S_CACHE="$D/cache/i1_tok_cache_DriverB_new_40k_subset_prev20.npz"

echo "### [v2s] DriverB_new 40k held-out eval $(date)"
python -u scripts/eval_a2_multiframe.py \
  --i1-cache "$V2S_CACHE" --aux-dir "$D/aux_targets_DriverB_new" \
  --run-filter "" \
  --v7b data/ckpts/a2_v7b_d128_alpha075_best_loss.pt \
  --mf v2s_5f "$OUT/mf5f_v2s3set_10_20_30_40_50/best_loss.pt" 6 off "10,20,30,40,50" \
  --out "$OUT/expand_driverb/eval_v2s_driverb.json" 2>&1 | tee "$OUT/expand_driverb/v2s_stdout.log"

echo "### [b0] DriverB_new 40k held-out eval $(date)"
python -u scripts/eval_a2_multiframe.py \
  --i1-cache "$B0_CACHE" --aux-dir "$D/aux_targets_DriverB_new" \
  --run-filter "" \
  --v7b data/ckpts/a2_v7b_d128_alpha075_best_loss.pt \
  --mf b0_5f "$OUT/mf5f_b0_10_20_30_40_50/best_loss.pt" 6 off "10,20,30,40,50" \
  --out "$OUT/expand_driverb/eval_b0_driverb.json" 2>&1 | tee "$OUT/expand_driverb/b0_stdout.log"

echo "### EXPAND EVAL DONE $(date)"

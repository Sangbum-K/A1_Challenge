#!/bin/bash
# a_2-on-b0 ablation 전체 체인 (b0 cache 3-set 준비 완료 후 실행).
#   subset = spd_v2 + sangam_lead_0520 + sangam_lead_0521 (77,594; DriverB 는 NFS 느려 제외).
#   1) b0  5f 학습 (d_token=192, b0 cache 3-set)
#   2) v2s 5f 학습 (d_token=256, v2_s cache 동일 3-set) — 공정 비교쌍 (같은 데이터/recipe/epoch)
#   3) eval: 동일 held-out (val-split seed=0 5%, 3-cache 동일 순서·key) 에서
#       - run V2S: v2s cache + v2s 5f ckpt (+ v7b 참고)
#       - run B0 : b0 cache  + b0 5f ckpt
#      b0/v2s cache 가 같은 key/순서 → val-split sample 1:1 동일 → 직접 비교 가능.
set -e
source /home/ailab-12/miniforge3/etc/profile.d/conda.sh
conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
OUT=outputs_multi/a2_b0_abl
mkdir -p "$OUT"

echo "### [1/3] train b0 5f $(date)"
bash scripts/train_a2_b0_5f.sh b0 192 mf5f_b0_10_20_30_40_50 i1_tok_cache_b0_ b0/

echo "### [2/3] train v2s 5f (3set, fair pair) $(date)"
bash scripts/train_a2_b0_5f.sh v2s 256 mf5f_v2s3set_10_20_30_40_50 i1_tok_cache_ ""

echo "### [3/3] eval $(date)"
# run V2S (256ch cache, v7b 참고 포함)
python -u scripts/eval_a2_multiframe.py \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0520_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0520" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0521_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0521" \
  --i1-cache "$D/cache/i1_tok_cache_spd_v2_prev20.npz"           --aux-dir "$D/aux_targets_spd_nolead_0520" \
  --run-filter "" --run-filter "" --run-filter "" \
  --v7b data/ckpts/a2_v7b_d128_alpha075_best_loss.pt \
  --val-split --val-ratio 0.05 --seed 0 \
  --mf v2s_5f "$OUT/mf5f_v2s3set_10_20_30_40_50/best_loss.pt" 6 off "10,20,30,40,50" \
  --out "$OUT/eval_v2s_3set.json"

# run B0 (192ch cache; v7b 자동 skip)
python -u scripts/eval_a2_multiframe.py \
  --i1-cache "$D/cache/b0/i1_tok_cache_b0_sangam_lead_0520_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0520" \
  --i1-cache "$D/cache/b0/i1_tok_cache_b0_sangam_lead_0521_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0521" \
  --i1-cache "$D/cache/b0/i1_tok_cache_b0_spd_v2_prev20.npz"           --aux-dir "$D/aux_targets_spd_nolead_0520" \
  --run-filter "" --run-filter "" --run-filter "" \
  --v7b data/ckpts/a2_v7b_d128_alpha075_best_loss.pt \
  --val-split --val-ratio 0.05 --seed 0 \
  --mf b0_5f "$OUT/mf5f_b0_10_20_30_40_50/best_loss.pt" 6 off "10,20,30,40,50" \
  --out "$OUT/eval_b0.json"

echo "### ALL DONE $(date)"

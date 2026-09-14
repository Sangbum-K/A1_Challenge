#!/bin/bash
# temporal-span / frame-density ablation: a_2 vL_rel 정확도(특히 far std) 에 대한
#   ① span(0.5s vs 1.0s) ② 밀도(10 vs 20 frame) 효과 분리.
# cache 는 전체 frame token 보유 → 재생성 불필요, prev_offsets 만 다르게 학습.
# init/lr/epoch/seed 동일(공정). loss 항(OHEM/margin) off = 순수 recipe(baseline 과 동일).
set -e
source /home/ailab-12/miniforge3/etc/profile.d/conda.sh; conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # seg-mask 생성과 GPU 공존(12GB) OOM 방어
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
OUT=outputs_multi/a2_tspan; mkdir -p "$OUT"
INIT=data/ckpts/a2_v7b_d128_alpha075_best_loss.pt
BASE=( --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0520_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0520"
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0521_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0521"
  --i1-cache "$D/cache/i1_tok_cache_spd_v2_prev20.npz" --aux-dir "$D/aux_targets_spd_nolead_0520"
  --run-filter "" --run-filter "" --run-filter "" --yaw-shift off
  --resume-from "$INIT" --d-token 256 --d-out 128 --epochs 12 --lr 1e-4 --weight-decay 1e-4
  --workers 4 --val-ratio 0.05 --w-obj 1.0 --w-reg 1.0 --use-focal --focal-alpha 0.75 --focal-gamma 2.0
  --scale-v 20.0 --scale-dist 40.0 --lds --fx 640 --seed 0 --out-dir "$OUT" )

OFF_S10N10="10,20,30,40,50,60,70,80,90,100"                                  # 1.0s span, 10 frame, gap 10
OFF_S05N10="5,10,15,20,25,30,35,40,45,50"                                    # 0.5s span, 10 frame, gap 5
OFF_S10N20="5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100"     # 1.0s span, 20 frame, gap 5

echo "### [1/4] s10n10 (1.0s,10f) bs128 $(date)"
python -u scripts/train_a2_multiframe.py "${BASE[@]}" --batch-size 128 --prev-offsets "$OFF_S10N10" --name s10n10
echo "### [2/4] s05n10 (0.5s,10f) bs128 $(date)"
python -u scripts/train_a2_multiframe.py "${BASE[@]}" --batch-size 128 --prev-offsets "$OFF_S05N10" --name s05n10
echo "### [3/4] s10n20 (1.0s,20f) bs64 $(date)"
python -u scripts/train_a2_multiframe.py "${BASE[@]}" --batch-size 64  --prev-offsets "$OFF_S10N20" --name s10n20

echo "### [4/4] eval (vL_rel err by dist + objsep) $(date)"
python -u scripts/eval_a2_multiframe.py \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0520_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0520" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0521_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0521" \
  --i1-cache "$D/cache/i1_tok_cache_spd_v2_prev20.npz" --aux-dir "$D/aux_targets_spd_nolead_0520" \
  --run-filter "" --run-filter "" --run-filter "" \
  --v7b "$INIT" --val-split --val-ratio 0.05 --seed 0 \
  --mf base05n5 outputs_multi/a2_b0_abl/mf5f_v2s3set_10_20_30_40_50/best_loss.pt 6  off "10,20,30,40,50" \
  --mf s05n10  "$OUT/s05n10/best_loss.pt" 11 off "$OFF_S05N10" \
  --mf s10n10  "$OUT/s10n10/best_loss.pt" 11 off "$OFF_S10N10" \
  --mf s10n20  "$OUT/s10n20/best_loss.pt" 21 off "$OFF_S10N20" \
  --out "$OUT/eval_tspan.json"
echo "### ALL DONE $(date)"

#!/bin/bash
# a_2 mf20 재학습 v2 (DriverB 추가) — 2026-05-27.
#   배경: mf20f_v2s_5to100 은 3-set(sangam_lead_0520/0521 + spd_v2 nolead)만 학습 → DriverB_new 는 OOD.
#         오인지 측정에서 DriverB closing Pearson 0.18 / mid-far presence 약함 = OOD 일반화 실패.
#   조치: DriverB_new 를 4번째 데이터셋으로 추가(도심 = 넓은 vL_rel + mid/far lead 풍부).
#         recipe 는 mf20 와 동일(데이터 추가 효과 격리). warm = mf20 best (always warm-start).
#   주의: a_2 가 바뀌면 i2_mem cache 재생성 + f_2/gate/scc 재학습 필요(downstream 재정합) — 결과 확인 후.
set -e
source /home/ailab-12/miniforge3/etc/profile.d/conda.sh
conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
CKPT_INIT=outputs_multi/a2_b0_abl/mf20f_v2s_5to100/best_loss.pt   # 현 배포 mf20 (= a2_mf20_v2s_n20.pt)
OUT=outputs_multi/a2_b0_abl
OFFS="5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100"

python -u scripts/train_a2_multiframe.py \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0520_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0520" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0521_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0521" \
  --i1-cache "$D/cache/i1_tok_cache_spd_v2_prev20.npz"           --aux-dir "$D/aux_targets_spd_nolead_0520" \
  --i1-cache "$D/cache/i1_tok_cache_DriverB_new_prev20.npz"      --aux-dir "$D/aux_targets_DriverB_new" \
  --run-filter "" --run-filter "" --run-filter "" --run-filter "" \
  --prev-offsets "$OFFS" \
  --yaw-shift off \
  --resume-from "$CKPT_INIT" \
  --d-token 256 --d-out 128 \
  --epochs 12 --batch-size 32 --lr 1e-4 --weight-decay 1e-4 \
  --workers 4 --val-ratio 0.05 \
  --w-obj 1.0 --w-reg 1.0 --use-focal --focal-alpha 0.75 --focal-gamma 2.0 \
  --scale-v 20.0 --scale-dist 40.0 --lds --fx 640 --seed 0 \
  --name mf20f_v2s_5to100_db --out-dir "$OUT"
echo "=== TRAIN DONE mf20f_v2s_5to100_db $(date) ==="

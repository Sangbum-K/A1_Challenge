#!/usr/bin/env bash
# seg backbone 경량화 mIoU sweep.
# baseline (tf_efficientnetv2_s, fx640_v3) 와 동일 데이터/split/epoch budget.
#   epochs=15 batch=8 amp lr=3e-4 (from-scratch: 경량 backbone 은 채널 달라 v2_s ckpt warm-start 불가,
#   ImageNet pretrained backbone 에서 시작. lr 은 코드 default 3e-4).
#   데이터: speedway_agent1 + sangam_fx640 (--no-default), baseline 과 동일 2개 extra.
# 후보 순차 실행 (동시 학습 금지 — 한 프로세스에서 차례로).
set -e
PY=/home/ailab-12/miniforge3/envs/ml_con/bin/python
cd /home/ailab-12/git/scc_controller_imitator

EXTRA="--no-default \
  --extra-meta /home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/speedway_agent1_meta.json \
  --extra-mask-dir /home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/seg_masks_spd \
  --extra-raw-base /home/ailab-12/AILabSSD/02_Data/57_A1_E2E/00_raw \
  --extra-meta /home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/sangam_fx640_meta.json \
  --extra-mask-dir /home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/seg_masks_sangam_fx640 \
  --extra-raw-base /home/ailab-12/AILabSSD/02_Data/57_A1_E2E/00_raw"

for BB in tf_efficientnetv2_b0 tf_efficientnetv2_b1 tf_efficientnetv2_b2; do
  NAME="${BB#tf_efficientnetv2_}"   # b0 / b1 / b2
  echo "==================== TRAIN $BB (name=$NAME) ===================="
  $PY scripts/train_seg_pretrain.py \
    --backbone "$BB" --name "lite_${NAME}" \
    --epochs 15 --batch-size 8 --lr 3e-4 --amp --workers 6 \
    --seed 0 --val-ratio 0.1 \
    $EXTRA
done
echo "==================== SWEEP DONE ===================="

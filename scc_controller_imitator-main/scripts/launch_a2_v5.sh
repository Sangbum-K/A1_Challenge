#!/bin/bash
# a_2 v5 — encoder + fusion frozen (v4 ckpt reuse), fc_proj (256→128) + new AuxHead Linear(128→7) 만 학습
# 목적: iy_model 의 d_model=128 호환 image_mem 생성 가능한 a_2 모델
# 기존 데이터 + 기존 loss (사용자 확정 spec)

set -e
LOG=/tmp/a2_v5.log
CKPT_INIT=/home/ailab-12/git/scc_controller_imitator/data/ckpts/a2_fx640_v4_best_loss.pt
[ -f "$CKPT_INIT" ] || { echo "ERR: $CKPT_INIT 없음"; exit 1; }

source /home/ailab-12/miniforge3/etc/profile.d/conda.sh
conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator

D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data

nohup python -u scripts/train_a2_head.py \
  --pretrained outputs_multi/seg_pretrain_fx640_v3/best_miou_backbone.pt \
  --resume-from "$CKPT_INIT" \
  --d-out 128 \
  --freeze-enc-fusion \
  --epochs 5 \
  --batch-size 6 \
  --lr 3e-4 \
  --lr-encoder 0 \
  --weight-decay 1e-4 \
  --workers 8 \
  --val-ratio 0.1 \
  --w-obj 1.0 --w-reg 1.0 \
  --scale-v 20.0 --scale-dist 40.0 \
  --no-default \
  --no-require-obj \
  --close-weight 3.0 \
  --extra-meta "$D/speedway_agent1_meta.json" \
  --extra-a2-dir "$D/aux_targets_spd_fx640" \
  --extra-raw-base "$D/../raw_cache" \
  --extra-meta "$D/meta_spd_nolead_0520.json" \
  --extra-a2-dir "$D/aux_targets_spd_nolead_0520" \
  --extra-raw-base /home/ailab-12/AILabSSD/02_Data/57_A1_E2E/00_raw \
  --extra-meta "$D/meta_spd_nolead_0521.json" \
  --extra-a2-dir "$D/aux_targets_spd_nolead_0521" \
  --extra-raw-base /home/ailab-12/AILabSSD/02_Data/57_A1_E2E/00_raw \
  --extra-meta "$D/meta_sangam_lead_0520.json" \
  --extra-a2-dir "$D/aux_targets_sangam_lead_0520" \
  --extra-raw-base /home/ailab-12/AILabSSD/02_Data/57_A1_E2E/00_raw \
  --extra-meta "$D/meta_sangam_lead_0521.json" \
  --extra-a2-dir "$D/aux_targets_sangam_lead_0521" \
  --extra-raw-base /home/ailab-12/AILabSSD/02_Data/57_A1_E2E/00_raw \
  --prev-dtick 20 \
  --close-lead-meta "$D/cache/close_lead_meta_v4.npz" \
  --close-thr 30.0 \
  --amp \
  --name fx640_v5_d128 \
  --out-dir outputs_multi/a2 \
  > "$LOG" 2>&1 &

PID=$!
echo "a_2 v5 launched pid=$PID log=$LOG"
echo "$PID" > /tmp/a2_v5.pid

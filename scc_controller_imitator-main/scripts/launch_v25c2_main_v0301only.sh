#!/bin/bash
# v25c2 — main(Professor #654 계보) 추가 finetune, ci_aug = v0301 ONLY.
#   resume = v25c best (= 현재 B4 의 frozen main 과 byte-identical).
#   main_no_image=True, side branch 없음 (순수 main waypoint/steer 정제).
#   주의: 이 새 main 을 B 에 쓰려면 side 재학습(또는 phase2 joint) 필요 — side 는 v25c main 위 residual.
#   v0301 은 waypoint-only(brake 없음) → steer/waypoint 만 정제됨.
#   "더" finetune 이므로 lr 낮춤(1e-4) + 1000ep + val 감시(단독 데이터 overfit 주의 — best_loss 기준).
set -e
LOG=/tmp/v25c2_main_v0301.log
CKPT_INIT=/home/ailab-12/git/scc_controller_imitator/data/ckpts/iy_v25c_best_loss.pt
[ -f "$CKPT_INIT" ] || { echo "ERR: $CKPT_INIT 없음"; exit 1; }
source /home/ailab-12/miniforge3/etc/profile.d/conda.sh
conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
CI=/home/ailab-12/AILabDatasetLocal/02_Custom_Dataset/35_e2e_morai/controller_imitator

nohup python -u scripts/train_iy.py \
  --resume-from "$CKPT_INIT" \
  --prebake "$D/prebake/f1_prebake_iy_sangam_lead_0520_v1_brakeaug.npz" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0520.npz" \
  --limit 1000 \
  --ci-aug-dirs "$CI/controller_imitator_data_multi_valid_v0301" \
  --main-no-image \
  --d-model 128 --nhead 8 --num-layers 6 \
  --traj-zero-after 20 \
  --scale-steer 10.0 --scale-ax 10.0 \
  --close-thr 30.0 --close-weight 3.0 \
  --lr 1.0e-4 --lr-prof 1.0e-4 --weight-decay 1e-4 \
  --fx 640 \
  --epochs 1000 --batch-size 512 --workers 6 --val-ratio 0.05 \
  --wandb --wandb-run v25c2_main_v0301only \
  --name v25c2_main_v0301only \
  --out-dir outputs_multi/iy \
  > "$LOG" 2>&1 &
echo $! > /tmp/v25c2_main_v0301.pid
echo "v25c2 main v0301-only pid=$(cat /tmp/v25c2_main_v0301.pid) log=$LOG"
echo "best → outputs_multi/iy_v25c2_main_v0301only/best_loss.pt"

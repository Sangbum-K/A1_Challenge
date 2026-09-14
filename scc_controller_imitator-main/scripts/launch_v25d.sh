#!/bin/bash
# v25d: f_1 (v25c main alone baseline) + f_2 (Side SCC module) 가동
#   f_1 = main_no_image transformer (Professor RAW finetuned, waypoint tracking)
#   f_2 = Side SCC = cross-attn (Q=traj, KV=image_mem) + a_2 v4 feat early inject +
#         TTC augment label floor (in-lane bin 16-25 gating)
# 2-phase: phase1 (ep 1-15) f_1 frozen, f_2 학습 / phase2 (ep 16-50) joint low f_1 lr
# 데이터 = v25c 와 동일 (sangam_lead_0520 limit 1000 + ci_aug v0237/v0301/v0302)
set -e
LOG=/tmp/v25d.log
CKPT_INIT=/home/ailab-12/git/scc_controller_imitator/data/ckpts/iy_v25c_best_loss.pt
[ -f "$CKPT_INIT" ] || { echo "ERR: $CKPT_INIT 없음 — v25c 먼저 완료 필요"; exit 1; }

source /home/ailab-12/miniforge3/etc/profile.d/conda.sh
conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator

nohup python -u scripts/train_iy.py \
  --resume-from "$CKPT_INIT" \
  --prebake     /home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/prebake/f1_prebake_iy_sangam_lead_0520_v1_brakeaug.npz \
  --i1-cache    /home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/cache/i1_tok_cache_sangam_lead_0520.npz \
  --a2-cache    /home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/cache/a2_out_cache_sangam_lead_0520.npz \
  --limit 1000 \
  --ci-aug-dirs \
    /home/ailab-12/AILabDatasetLocal/02_Custom_Dataset/35_e2e_morai/controller_imitator/controller_imitator_data_multi_valid_v0237 \
    /home/ailab-12/AILabDatasetLocal/02_Custom_Dataset/35_e2e_morai/controller_imitator/controller_imitator_data_multi_valid_v0301 \
    /home/ailab-12/AILabDatasetLocal/02_Custom_Dataset/35_e2e_morai/controller_imitator/controller_imitator_data_multi_valid_v0302 \
  --main-no-image \
  --side-cross-attn \
  --use-a2-feat \
  --ttc-augment \
  --ttc-augment-steps 10 \
  --ttc-augment-max-brake 20.0 \
  --ttc-augment-dist-max 50.0 \
  --ttc-augment-ttc-max 4.0 \
  --ttc-augment-safety-margin 2.0 \
  --under-brake-weight 1.0 \
  --under-brake-thr -2.0 \
  --traj-zero-after 15 \
  --phase1-epochs 15 \
  --main-lr-phase2 1.5e-5 \
  --side-lr 3e-4 \
  --lr 3e-4 --lr-prof 1.5e-5 \
  --fx 640 \
  --epochs 50 --batch-size 512 --workers 6 --val-ratio 0.05 \
  --wandb --wandb-run v25d_side_a2_ttc_resume_v25c \
  --name v25d_side_a2_ttc_resume_v25c \
  --out-dir /home/ailab-12/git/scc_controller_imitator/outputs_multi/iy \
  > "$LOG" 2>&1 &

PID=$!
echo "v25d launched pid=$PID log=$LOG"
echo "$PID" > /tmp/v25d.pid

#!/bin/bash
# v26 B-minACC — lead-gated ACC override (additive residual 대체).
#   ax_step0 = main_ax + g·clamp(b_side - main_ax, max=0)
#     b_side = side 의 절대 ACC brake ceiling (cross-attn Q=traj, KV=i2_mem)
#     g      = sigmoid(a_2 in-lane(16-25) obj max + gate_bias) — 전방차 인지 시에만 동작
#   장점: side 가 main_ax 를 몰라도 됨(절대값+downward), main finetune 과 decouple, FP 안전(obj gate).
#   resume = B4 phase1 best (side cross-attn 4L decoder warm). main frozen (phase1 전체).
#   b_side 직접 supervision (→ target ax step0, ttc-augmented) 로 dead-start 회피.
set -e
LOG=/tmp/v26_Bminacc.log
# v0301-finetuned main 에서 resume (side 는 fresh — min-arb 절대 brake ceiling 새 학습).
CKPT_INIT=/home/ailab-12/git/scc_controller_imitator/outputs_multi/iy_v25c2_main_v0301only/best_loss.pt
[ -f "$CKPT_INIT" ] || { echo "ERR: $CKPT_INIT 없음 — v0301 main finetune 먼저"; exit 1; }
source /home/ailab-12/miniforge3/etc/profile.d/conda.sh
conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
CI=/home/ailab-12/AILabDatasetLocal/02_Custom_Dataset/35_e2e_morai/controller_imitator

nohup python -u scripts/train_iy.py \
  --resume-from "$CKPT_INIT" \
  --prebake "$D/prebake/f1_prebake_iy_sangam_lead_0520_v1_brakeaug_prev20.npz" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0520_prev20.npz" \
  --i2-cache "$D/cache/i2_mem_cache_sangam_lead_0520_v7b_prev20.npz" \
  --inlane-gt-cache "$D/cache/inlane_gt_sangam_lead_0520.npz" \
  --prebake-extra \
    "$D/prebake/f1_prebake_iy_sangam_lead_0521_v1_brakeaug_prev20.npz" \
    "$D/prebake/f1_prebake_iy_sangam_pertO_v2_prev20.npz" \
    "$D/prebake/f1_prebake_iy_sangam_pertX_v2_prev20.npz" \
    "$D/prebake/f1_prebake_iy_DriverB_new_logged_fixed_prev20.npz" \
    "$D/prebake/f1_prebake_iy_spd_streaming_brakeaug_prev20.npz" \
  --i1-cache-extra \
    "$D/cache/i1_tok_cache_sangam_lead_0521_prev20.npz" \
    "$D/cache/i1_tok_cache_sangam_pertO_v2_prev20.npz" \
    "$D/cache/i1_tok_cache_sangam_pertX_v2_prev20.npz" \
    "$D/cache/i1_tok_cache_DriverB_new_prev20.npz" \
    "$D/cache/i1_tok_cache_spd_v2_prev20.npz" \
  --i2-cache-extra \
    "$D/cache/i2_mem_cache_sangam_lead_0521_v7b_prev20.npz" \
    "$D/cache/i2_mem_cache_sangam_pertO_v2_v7b_prev20.npz" \
    "$D/cache/i2_mem_cache_sangam_pertX_v2_v7b_prev20.npz" \
    "$D/cache/i2_mem_cache_DriverB_new_v7b_prev20.npz" \
    "$D/cache/i2_mem_cache_spd_v2_v7b_prev20.npz" \
  --inlane-gt-cache-extra \
    "$D/cache/inlane_gt_sangam_lead_0521.npz" \
    "$D/cache/inlane_gt_sangam_pertO_v2.npz" \
    "$D/cache/inlane_gt_sangam_pertX_v2.npz" \
    "$D/cache/inlane_gt_DriverB_new.npz" \
    "$D/cache/inlane_gt_spd_v2.npz" \
  --prebake-extra-limits 0 500000 500000 0 0 \
  --ci-aug-dirs "$CI/controller_imitator_data_multi_valid_v0237" \
  --ci-aug-limit 100000 \
  --main-no-image --side-cross-attn --side-num-layers 4 --side-dim-feedforward 512 \
  --side-min-arb --side-min-arb-w 0.5 \
  --target-first-only --traj-zero-after 20 \
  --phase1-epochs 30 \
  --main-lr-phase2 1.5e-5 --side-lr 3e-4 \
  --lr 3e-4 --lr-prof 1.5e-5 \
  --close-thr 30.0 --close-weight 5.0 \
  --ttc-augment --ttc-acc-style \
  --ttc-augment-steps 10 --ttc-augment-max-brake 20.0 \
  --ttc-acc-d0 5.0 --ttc-acc-headway 1.0 --ttc-acc-kgap 0.2 --ttc-acc-kclo 0.6 --ttc-acc-ttc-emerg 2.0 --ttc-acc-kclo-ttc-decay 5.0 \
  --fx 640 \
  --epochs 30 --batch-size 256 --workers 6 --val-ratio 0.05 \
  --wandb --wandb-run v26_Bminacc \
  --name v26_Bminacc \
  --out-dir outputs_multi/iy \
  > "$LOG" 2>&1 &
echo $! > /tmp/v26_Bminacc.pid
echo "v26 B-minACC pid=$(cat /tmp/v26_Bminacc.pid) log=$LOG"
echo "best → outputs_multi/iy_v26_Bminacc/best_loss.pt"

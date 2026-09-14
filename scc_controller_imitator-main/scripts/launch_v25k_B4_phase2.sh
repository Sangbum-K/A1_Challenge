#!/bin/bash
# v25k-B4 PHASE2 finetune — phase1 best 에서 resume, main unfreeze (joint, low lr).
#   phase1 (main frozen, side overlay only) best 는 outputs_multi/iy_v25k_B4_gttc_accgate_r2 에 보존.
#   이 run 은 별도 out_dir(iy_v25k_B4_phase2_ft) → 두 best 공존, 서로 안 덮어씀.
#   --phase1-epochs 0 → ep0 부터 phase2 (main lr = main-lr-phase2, Professor steer 보존용 저lr).
#   close-approach val metric 포함 (train_iy.py 최신 코드).
set -e
LOG=/tmp/v25k_B4_phase2.log
CKPT_INIT=/home/ailab-12/git/scc_controller_imitator/outputs_multi/iy_v25k_B4_gttc_accgate_r2/best_loss.pt
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
  --target-first-only --traj-zero-after 20 \
  --phase1-epochs 0 \
  --main-lr-phase2 1.0e-5 --side-lr 1.5e-4 \
  --lr 1.5e-4 --lr-prof 1.0e-5 \
  --close-thr 30.0 --close-weight 5.0 \
  --ttc-augment --ttc-acc-style \
  --ttc-augment-steps 10 --ttc-augment-max-brake 20.0 \
  --ttc-acc-d0 5.0 --ttc-acc-headway 1.0 --ttc-acc-kgap 0.2 --ttc-acc-kclo 0.6 --ttc-acc-ttc-emerg 2.0 --ttc-acc-kclo-ttc-decay 5.0 \
  --fx 640 \
  --epochs 12 --batch-size 256 --workers 6 --val-ratio 0.05 \
  --wandb --wandb-run v25k_B4_phase2_ft \
  --name v25k_B4_phase2_ft \
  --out-dir outputs_multi/iy \
  > "$LOG" 2>&1 &
echo $! > /tmp/v25k_B4_phase2.pid
echo "v25k-B4 phase2 pid=$(cat /tmp/v25k_B4_phase2.pid) log=$LOG"

#!/bin/bash
# =============================================================================
# SLURM sbatch wrapper — a_2 multi-frame head 학습 (objectness + vL_rel/dist reg)
#
#   원본 launcher: scripts/launch_a2_multiframe.sh (AILAB-12 로컬 NVMe 기준)
#   이 wrapper 는 동일 학습 인자 + recipe 를 cluster 의 apptainer/.sif 환경에서 실행.
#   경로만 NFS(SCC_DATA_DIR)로 치환, 학습 스크립트(train_a2_multiframe.py)는 무수정.
#
#   제출: (cluster login 노드에서) sbatch slurm_train_a2_multiframe.sh [NAME] [NF] [YAW]
#     예) sbatch slurm_train_a2_multiframe.sh a2_2f_yaw   2 on
#         sbatch slurm_train_a2_multiframe.sh a2_2f_noyaw 2 off
#         sbatch slurm_train_a2_multiframe.sh a2_3f_yaw   3 on
#   ※ AILAB-12 에는 slurm client 없음 → 이 머신에서는 제출 불가. cluster login 노드에서.
# =============================================================================
#SBATCH --job-name=scc_a2_multiframe
#SBATCH --comment=submitter:jiwon
#SBATCH --partition=partition-4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/home/ailab/AILabSSD/99_Management/slurm-logs/%x_%j.out
#SBATCH --error=/home/ailab/AILabSSD/99_Management/slurm-logs/%x_%j.err

set -euo pipefail

# ---- sbatch 인자(positional) ----
NAME=${1:-a2_2f_yaw}
NF=${2:-2}
YAW=${3:-on}

# ---- cluster 경로 (NFS) ----
SIF_IMAGE=/home/ailab/AILabSSD/01_SlurmWorkspace/a1_e2e_fixed_v2.sif
WORKSPACE=/home/ailab/AILabSSD/03_Shared_Repository/jiwon/scc_controller_imitator
SCC_DATA_DIR=/home/ailab/AILabSSD/02_Data/57_A1_E2E/01_processed/scc_controller_imitator/data

[ -f "${SIF_IMAGE}" ] || { echo "SIF not found: ${SIF_IMAGE}"; exit 1; }
[ -d "${WORKSPACE}" ] || { echo "WORKSPACE(NFS repo) not found: ${WORKSPACE} — git clone/rsync 먼저"; exit 1; }
[ -d "${SCC_DATA_DIR}" ] || { echo "SCC_DATA_DIR not found: ${SCC_DATA_DIR}"; exit 1; }

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

apptainer exec --nv \
    --bind "${WORKSPACE}:${WORKSPACE}" \
    --bind "${SCC_DATA_DIR}:${SCC_DATA_DIR}" \
    --env SCC_DATA_DIR="${SCC_DATA_DIR}" \
    --env NAME="${NAME}" --env NF="${NF}" --env YAW="${YAW}" \
    --env WORKSPACE="${WORKSPACE}" \
    "${SIF_IMAGE}" \
    bash <<'CONTAINER_EOF'
set -euo pipefail
echo "=== HOST ==="; hostname
echo "=== ENV ==="
source /opt/conda/etc/profile.d/conda.sh
conda activate a1_learning
export PYTHONNOUSERSITE=True
export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "${WORKSPACE}"
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

D="${SCC_DATA_DIR}"
P="$D/prebake"
CKPT_INIT="$D/ckpts/a2_v7b_d128_alpha075_best_loss.pt"   # repo data/ckpts → NFS data/ckpts 사본
[ -f "${CKPT_INIT}" ] || { echo "init ckpt not found: ${CKPT_INIT}"; exit 1; }

echo "=== RUN a_2 multiframe NAME=${NAME} NF=${NF} YAW=${YAW} ==="
python -u scripts/train_a2_multiframe.py \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0520_prev20.npz" \
  --aux-dir  "$D/aux_targets_sangam_lead_0520" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0521_prev20.npz" \
  --aux-dir  "$D/aux_targets_sangam_lead_0521" \
  --i1-cache "$D/cache/i1_tok_cache_DriverB_new_prev20.npz" \
  --aux-dir  "$D/aux_targets_DriverB_new" \
  --i1-cache "$D/cache/i1_tok_cache_spd_v2_prev20.npz" \
  --aux-dir  "$D/aux_targets_spd_nolead_0520" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_pertO_v2_prev20.npz" \
  --aux-dir  "$D/aux_targets_sangam_fx640" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_pertX_v2_prev20.npz" \
  --aux-dir  "$D/aux_targets_sangam_fx640" \
  --run-filter "" --run-filter "" --run-filter "" --run-filter "" --run-filter "pertO" --run-filter "pertX" \
  --prebake "$P/f1_prebake_iy_sangam_lead_0520_a2inlane_prev20.npz" \
  --prebake "$P/f1_prebake_iy_sangam_lead_0521_a2inlane_prev20.npz" \
  --prebake "$P/f1_prebake_iy_DriverB_new_a2inlane_prev20.npz" \
  --prebake "$P/f1_prebake_iy_sangam_pertO_v2_a2inlane_prev20.npz" \
  --prebake "$P/f1_prebake_iy_sangam_pertX_v2_a2inlane_prev20.npz" \
  --n-frames "$NF" --yaw-shift "$YAW" \
  --resume-from "$CKPT_INIT" \
  --d-out 128 \
  --epochs 12 \
  --batch-size 192 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --workers 4 \
  --val-ratio 0.05 \
  --w-obj 1.0 --w-reg 1.0 \
  --use-focal --focal-alpha 0.75 --focal-gamma 2.0 \
  --scale-v 20.0 --scale-dist 40.0 \
  --lds \
  --fx 640 \
  --name "$NAME" \
  --out-dir "${WORKSPACE}/outputs_multi"
echo "=== DONE (best: ${WORKSPACE}/outputs_multi/${NAME}/best_loss.pt) ==="
CONTAINER_EOF

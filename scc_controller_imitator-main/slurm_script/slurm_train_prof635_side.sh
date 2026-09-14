#!/bin/bash
# =============================================================================
# SLURM sbatch wrapper — prof635 frozen main(steer) + f_2 side(pedal) 학습
#
#   원본 launcher: scripts/launch_prof635_v6_driverb.sh / launch_prof635_distbrake.sh
#   train_prof635_side.py 의 DATASETS default 5종(sangam_lead_0520/0521 + pertO_v2 +
#   pertX_v2 + DriverB_new) 을 인자 없이 전부 학습. 경로는 SCC_DATA_DIR 기준.
#   입력: prebake/, cache/i2_mem_*_v7b_prev20.npz, cache/pedal_label_*_{suffix}.npz
#   main(steer) frozen ckpt: data/ckpts/main_prof654.pth (NFS 사본).
#
#   제출: (cluster login 노드에서)
#     sbatch slurm_train_prof635_side.sh [NAME] [PEDAL_SUFFIX] [DELTA_MODE]
#       NAME         기본 prof635_side_slurm
#       PEDAL_SUFFIX ttc(default, closing-ACC) | distbrake(거리-onset, 현행 배포)
#       DELTA_MODE   brake_only(default) | signed
#     예) sbatch slurm_train_prof635_side.sh prof635_distbrake distbrake brake_only
#   ※ AILAB-12 에는 slurm client 없음 → 제출은 cluster login 노드에서.
#   주의: wandb 사용 시 컨테이너 내 WANDB_API_KEY env / netrc 필요. 기본 OFF.
# =============================================================================
#SBATCH --job-name=scc_prof635_side
#SBATCH --comment=submitter:jiwon
#SBATCH --partition=partition-4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/home/ailab/AILabSSD/99_Management/slurm-logs/%x_%j.out
#SBATCH --error=/home/ailab/AILabSSD/99_Management/slurm-logs/%x_%j.err

set -euo pipefail

NAME=${1:-prof635_side_slurm}
PEDAL_SUFFIX=${2:-ttc}
DELTA_MODE=${3:-brake_only}

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
    --env NAME="${NAME}" --env PEDAL_SUFFIX="${PEDAL_SUFFIX}" --env DELTA_MODE="${DELTA_MODE}" \
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

# main(steer) frozen ckpt — train_prof635_side.py default 가 repo 상대 data/ckpts/main_prof654.pth.
# NFS data dir 사본을 명시 지정(절대경로) → repo data/ckpts 동기화 여부와 무관하게 동작.
MAIN_CKPT="${SCC_DATA_DIR}/ckpts/main_prof654.pth"
[ -f "${MAIN_CKPT}" ] || { echo "main ckpt not found: ${MAIN_CKPT}"; exit 1; }

echo "=== RUN prof635 side NAME=${NAME} suffix=${PEDAL_SUFFIX} delta=${DELTA_MODE} ==="
python -u scripts/train_prof635_side.py \
    --name "${NAME}" \
    --main-ckpt "${MAIN_CKPT}" \
    --pedal-suffix "${PEDAL_SUFFIX}" \
    --delta-mode "${DELTA_MODE}" \
    --epochs 25 --batch-size 512 --lr 3e-4 --workers 6
echo "=== DONE (best: ${WORKSPACE}/outputs_multi/${NAME}/best.pt) ==="
CONTAINER_EOF

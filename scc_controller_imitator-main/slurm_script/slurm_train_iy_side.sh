#!/bin/bash
# =============================================================================
# SLURM sbatch wrapper — f_2 (iy side cross-attn) 학습  [Professor #654 main + Side SCC]
#
#   원본 launcher: scripts/launch_v25h_B.sh (AILAB-12 로컬 NVMe 기준)
#   main = Professor #654 (text-only, --main-no-image), side = cross-attn(Q=traj_out, KV=i_2 cache).
#   train_iy.py 무수정. 경로만 NFS(SCC_DATA_DIR / AILabDataset ci-aug)로 치환.
#
#   입력: prebake/, cache/i1_tok_*, cache/i2_mem_*_v6_prev20, + ci-aug(controller_imitator).
#   init: ${SCC_DATA_DIR}/ckpts/iy_v25c_best_loss.pt (repo data/ckpts → NFS 사본; §4 복사 필요).
#
#   제출: (cluster login 노드) sbatch slurm_train_iy_side.sh [NAME] [SIDE_LAYERS]
#     예) sbatch slurm_train_iy_side.sh v25h_B2_cluster 4
#   ※ AILAB-12 에는 slurm client 없음 — 제출은 클러스터 login 노드에서.
#   주의: --wandb 미지정(기본 OFF). wandb 쓰려면 컨테이너 WANDB_API_KEY env 추가.
# =============================================================================
#SBATCH --job-name=scc_iy_side
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

NAME=${1:-v25h_B_cluster}
SIDE_LAYERS=${2:-4}

SIF_IMAGE=/home/ailab/AILabSSD/01_SlurmWorkspace/a1_e2e_fixed_v2.sif
WORKSPACE=/home/ailab/AILabSSD/03_Shared_Repository/jiwon/scc_controller_imitator
SCC_DATA_DIR=/home/ailab/AILabSSD/02_Data/57_A1_E2E/01_processed/scc_controller_imitator/data
# ci-aug(controller_imitator) — AILabDataset NFS (복사 안 함, 직접 bind)
CI_ROOT=/home/ailab/AILabDataset/02_Custom_Dataset/35_e2e_morai/controller_imitator

[ -f "${SIF_IMAGE}" ] || { echo "SIF not found: ${SIF_IMAGE}"; exit 1; }
[ -d "${WORKSPACE}" ] || { echo "WORKSPACE(NFS repo) not found: ${WORKSPACE} — git clone/rsync 먼저(§3)"; exit 1; }
[ -d "${SCC_DATA_DIR}" ] || { echo "SCC_DATA_DIR not found: ${SCC_DATA_DIR}"; exit 1; }
[ -d "${CI_ROOT}" ] || { echo "ci-aug root not found: ${CI_ROOT}"; exit 1; }

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

apptainer exec --nv \
    --bind "${WORKSPACE}:${WORKSPACE}" \
    --bind "${SCC_DATA_DIR}:${SCC_DATA_DIR}" \
    --bind "${CI_ROOT}:${CI_ROOT}" \
    --env SCC_DATA_DIR="${SCC_DATA_DIR}" \
    --env NAME="${NAME}" --env SIDE_LAYERS="${SIDE_LAYERS}" \
    --env WORKSPACE="${WORKSPACE}" --env CI_ROOT="${CI_ROOT}" \
    "${SIF_IMAGE}" \
    bash <<'CONTAINER_EOF'
set -euo pipefail
echo "=== HOST ==="; hostname
source /opt/conda/etc/profile.d/conda.sh
conda activate a1_learning
export PYTHONNOUSERSITE=True
export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "${WORKSPACE}"
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

D="${SCC_DATA_DIR}"
CI="${CI_ROOT}"
CKPT_INIT="$D/ckpts/iy_v25c_best_loss.pt"   # repo data/ckpts → NFS data/ckpts 사본
[ -f "${CKPT_INIT}" ] || { echo "init ckpt not found: ${CKPT_INIT} (§4 ckpts 복사 필요)"; exit 1; }

echo "=== RUN iy side NAME=${NAME} side_layers=${SIDE_LAYERS} ==="
python -u scripts/train_iy.py \
  --resume-from "$CKPT_INIT" \
  --prebake "$D/prebake/f1_prebake_iy_sangam_lead_0520_v1_brakeaug_prev20.npz" \
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0520_prev20.npz" \
  --i2-cache "$D/cache/i2_mem_cache_sangam_lead_0520_v6_prev20.npz" \
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
    "$D/cache/i2_mem_cache_sangam_lead_0521_v6_prev20.npz" \
    "$D/cache/i2_mem_cache_sangam_pertO_v2_v6_prev20.npz" \
    "$D/cache/i2_mem_cache_sangam_pertX_v2_v6_prev20.npz" \
    "$D/cache/i2_mem_cache_DriverB_new_v6_prev20.npz" \
    "$D/cache/i2_mem_cache_spd_v2_v6_prev20.npz" \
  --prebake-extra-limits 0 500000 500000 0 0 \
  --ci-aug-dirs "$CI/controller_imitator_data_multi_valid_v0237" \
  --ci-aug-limit 100000 \
  --main-no-image \
  --side-cross-attn \
  --side-num-layers "$SIDE_LAYERS" \
  --side-dim-feedforward 512 \
  --target-first-only \
  --traj-zero-after 15 \
  --phase1-epochs 20 \
  --main-lr-phase2 1.5e-5 \
  --side-lr 3e-4 \
  --lr 3e-4 --lr-prof 1.5e-5 \
  --close-thr 30.0 --close-weight 3.0 \
  --fx 640 \
  --epochs 30 --batch-size 256 --workers 6 --val-ratio 0.05 \
  --name "$NAME" \
  --out-dir "${WORKSPACE}/outputs_multi/iy"
echo "=== DONE (best: ${WORKSPACE}/outputs_multi/iy/${NAME}/best_loss.pt) ==="
CONTAINER_EOF

#!/bin/bash
# =============================================================================
# SLURM sbatch wrapper (선택) — i_2 v7b memory cache 재생성
#
#   원본: scripts/rebuild_i2_v7b_full.sh (AILAB-12 로컬). build_i2_cache.py 호출.
#   입력: cache/i1_tok_cache_*_prev20.npz + (옵션) prebake/*_a2inlane_*.npz
#         + a_2 ckpt(data/ckpts/a2_v7b_d128_alpha075_best_loss.pt → NFS 사본).
#   출력: cache/i2_mem_cache_*_v7b_prev20.npz (기존 덮어씀).
#
#   ※ 통상 학습만 한다면 불필요(i2 cache 는 이미 복사됨). i1 cache 재생성/
#     a_2 ckpt 교체 시에만 사용. GPU 추론(a_2 forward) 필요 → --gres=gpu:1.
#   제출: (cluster login 노드) sbatch slurm_build_caches.sh
# =============================================================================
#SBATCH --job-name=scc_build_i2cache
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

SIF_IMAGE=/home/ailab/AILabSSD/01_SlurmWorkspace/a1_e2e_fixed_v2.sif
WORKSPACE=/home/ailab/AILabSSD/03_Shared_Repository/jiwon/scc_controller_imitator
SCC_DATA_DIR=/home/ailab/AILabSSD/02_Data/57_A1_E2E/01_processed/scc_controller_imitator/data

[ -f "${SIF_IMAGE}" ] || { echo "SIF not found: ${SIF_IMAGE}"; exit 1; }
[ -d "${WORKSPACE}" ] || { echo "WORKSPACE(NFS repo) not found: ${WORKSPACE}"; exit 1; }
[ -d "${SCC_DATA_DIR}" ] || { echo "SCC_DATA_DIR not found: ${SCC_DATA_DIR}"; exit 1; }

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

apptainer exec --nv \
    --bind "${WORKSPACE}:${WORKSPACE}" \
    --bind "${SCC_DATA_DIR}:${SCC_DATA_DIR}" \
    --env SCC_DATA_DIR="${SCC_DATA_DIR}" \
    --env WORKSPACE="${WORKSPACE}" \
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
CKPT="$D/ckpts/a2_v7b_d128_alpha075_best_loss.pt"
[ -f "${CKPT}" ] || { echo "a_2 ckpt not found: ${CKPT}"; exit 1; }

# name|i1_cache|prebake(없으면 NONE → fallback 전체)
DATASETS=(
  "sangam_lead_0520|i1_tok_cache_sangam_lead_0520_prev20.npz|f1_prebake_iy_sangam_lead_0520_a2inlane_prev20.npz"
  "sangam_lead_0521|i1_tok_cache_sangam_lead_0521_prev20.npz|f1_prebake_iy_sangam_lead_0521_a2inlane_prev20.npz"
  "sangam_pertO_v2|i1_tok_cache_sangam_pertO_v2_prev20.npz|f1_prebake_iy_sangam_pertO_v2_a2inlane_prev20.npz"
  "sangam_pertX_v2|i1_tok_cache_sangam_pertX_v2_prev20.npz|f1_prebake_iy_sangam_pertX_v2_a2inlane_prev20.npz"
  "DriverB_new|i1_tok_cache_DriverB_new_prev20.npz|f1_prebake_iy_DriverB_new_a2inlane_prev20.npz"
  "spd_v2|i1_tok_cache_spd_v2_prev20.npz|NONE"
)
for ds in "${DATASETS[@]}"; do
  IFS='|' read -r name i1 pb <<< "$ds"
  OUT="$D/cache/i2_mem_cache_${name}_v7b_prev20.npz"
  echo "=== [build] $name -> $(basename "$OUT") ==="
  PBARG=""
  [ "$pb" != "NONE" ] && PBARG="--prebake $D/prebake/$pb"
  python -u scripts/build_i2_cache.py \
    --i1-cache "$D/cache/$i1" \
    $PBARG \
    --out "$OUT" \
    --a2-ckpt "$CKPT" --fx 640
done
echo "=== ALL DONE ==="
CONTAINER_EOF

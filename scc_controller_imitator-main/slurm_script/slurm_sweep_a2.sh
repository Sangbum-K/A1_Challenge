#!/bin/bash
# =============================================================================
# SLURM sbatch ARRAY — a_2 multi-frame head 하이퍼파라미터 sweep (본선 대량)
#
#   slurm_train_a2_multiframe.sh 의 데이터/recipe 를 그대로 쓰되, SLURM array index 로
#   (lr × focal-alpha × n-frames) 조합을 한 번에 제출. train_a2_multiframe.py 무수정.
#
#   조합 정의는 아래 CONFIGS 배열. array 크기를 CONFIGS 길이에 맞춰 --array 로 제출.
#   제출: (cluster login 노드)
#     N=$(grep -c '^  "' slurm_sweep_a2.sh) ; # 또는 아래 표 길이 수동 지정
#     sbatch --array=0-7%2 slurm_sweep_a2.sh        # 0..7 중 동시 2개씩
#   ※ %2 = 동시 실행 cap (GPU 여유에 맞게 조절). AILAB-12 엔 slurm client 없음.
#
#   각 task 의 NAME = a2sw_<idx>_<lr>_<alpha>_<nf>f  (outputs_multi/<NAME>/best_loss.pt).
# =============================================================================
#SBATCH --job-name=scc_a2_sweep
#SBATCH --comment=submitter:jiwon
#SBATCH --partition=partition-4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/home/ailab/AILabSSD/99_Management/slurm-logs/%x_%A_%a.out
#SBATCH --error=/home/ailab/AILabSSD/99_Management/slurm-logs/%x_%A_%a.err

set -euo pipefail

# ── sweep grid (idx 0..N-1). "LR|ALPHA|NF|YAW" ──────────────────────────────
CONFIGS=(
  "3e-4|0.75|2|on"
  "3e-4|0.75|3|on"
  "1e-4|0.75|2|on"
  "5e-4|0.75|2|on"
  "3e-4|0.90|2|on"
  "3e-4|0.50|2|on"
  "3e-4|0.75|2|off"
  "1e-4|0.90|3|on"
)

IDX=${SLURM_ARRAY_TASK_ID:-0}
[ "${IDX}" -lt "${#CONFIGS[@]}" ] || { echo "array idx ${IDX} >= ${#CONFIGS[@]} configs"; exit 1; }
IFS='|' read -r LR ALPHA NF YAW <<< "${CONFIGS[$IDX]}"
NAME="a2sw_${IDX}_lr${LR}_a${ALPHA}_${NF}f_${YAW}"

SIF_IMAGE=/home/ailab/AILabSSD/01_SlurmWorkspace/a1_e2e_fixed_v2.sif
WORKSPACE=/home/ailab/AILabSSD/03_Shared_Repository/jiwon/scc_controller_imitator
SCC_DATA_DIR=/home/ailab/AILabSSD/02_Data/57_A1_E2E/01_processed/scc_controller_imitator/data

[ -f "${SIF_IMAGE}" ] || { echo "SIF not found: ${SIF_IMAGE}"; exit 1; }
[ -d "${WORKSPACE}" ] || { echo "WORKSPACE(NFS repo) not found: ${WORKSPACE} — §3 먼저"; exit 1; }
[ -d "${SCC_DATA_DIR}" ] || { echo "SCC_DATA_DIR not found: ${SCC_DATA_DIR}"; exit 1; }

echo "=== sweep idx=${IDX} -> NAME=${NAME} (lr=${LR} alpha=${ALPHA} nf=${NF} yaw=${YAW}) ==="
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

apptainer exec --nv \
    --bind "${WORKSPACE}:${WORKSPACE}" \
    --bind "${SCC_DATA_DIR}:${SCC_DATA_DIR}" \
    --env SCC_DATA_DIR="${SCC_DATA_DIR}" \
    --env NAME="${NAME}" --env LR="${LR}" --env ALPHA="${ALPHA}" --env NF="${NF}" --env YAW="${YAW}" \
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
P="$D/prebake"
CKPT_INIT="$D/ckpts/a2_v7b_d128_alpha075_best_loss.pt"
[ -f "${CKPT_INIT}" ] || { echo "init ckpt not found: ${CKPT_INIT} (§4 ckpts 복사 필요)"; exit 1; }

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
  --lr "$LR" \
  --weight-decay 1e-4 \
  --workers 4 \
  --val-ratio 0.05 \
  --w-obj 1.0 --w-reg 1.0 \
  --use-focal --focal-alpha "$ALPHA" --focal-gamma 2.0 \
  --scale-v 20.0 --scale-dist 40.0 \
  --lds \
  --fx 640 \
  --name "$NAME" \
  --out-dir "${WORKSPACE}/outputs_multi"
echo "=== DONE idx=${IDX} (best: ${WORKSPACE}/outputs_multi/${NAME}/best_loss.pt) ==="
CONTAINER_EOF

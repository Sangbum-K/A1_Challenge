#!/bin/bash
# =============================================================================
# SLURM sbatch — DriverB_new SAM image-plane segmentation mask 생성
#
#   목적: DriverB_new (logged 실주행) 의 SAM mask 가 없어 seg pretrain 다양성에
#         포함 못함. GroundingDINO SwinB + SAM2.1 base+ ensemble 로 생성.
#   스크립트: scripts/generate_image_seg_masks.py (CAMERA_1, ROI y=[235,485),
#            (5class,250,1280) uint8 → seg_masks_DriverB_new/{frame_id}.npz)
#
#   입력(raw, NFS): <DATA2>/00_raw/sangam_track/DriverB_new/<run>/CAMERA_1/<tick>_Intensity.jpeg
#                   meta: <SCC_DATA_DIR>/meta_DriverB_new.json (135,113 frames)
#   출력: <SCC_DATA_DIR>/seg_masks_DriverB_new/{frame_id}.npz
#
#   ── 제출 전제 (반드시 확인) ───────────────────────────────────────────────
#   1) .sif 에 SAM deps 존재?  ── ★ HYPOTHESIS, 미확인 ★
#      a1_e2e_fixed_v2.sif 의 a1_learning env 에 groundingdino + sam2 +
#      jiwon_e2e/src.data.gt_pipeline import 가 되는지 확인 필요.
#      (로컬 ml_con env 에는 설치됨이 confirmed. .sif 는 학습용이라 미보장.)
#      → import 실패 시: SAM deps 포함 .sif 재빌드 또는 conda env 교체 필요.
#   2) raw NFS 접근: DriverB raw 는 AILabDataset(volume3) 에만 있음.
#      AILabSSD(00_raw) 에는 DriverB 없음 (synth/rule 데이터만). DATA2 bind 필수.
#   3) GD/SAM2 weight: <SCC_E2E_ROOT>/checkpoints/grounded_sam2/ (jiwon_e2e).
#   4) meta_DriverB_new.json 사전 복사 ── ★ 현재 클러스터 SCC_DATA_DIR 에 부재 ★
#      AILAB-12 로컬에만 있음(AILabDatasetLocal, 클러스터 미마운트). 제출 전:
#        scp/cp meta_DriverB_new.json → <SCC_DATA_DIR>/meta_DriverB_new.json (24MB)
#      (raw 이미지/weight 는 NFS 라 복사 불필요, meta json 만)
#
#   제출: (cluster login 노드) sbatch slurm_gen_seg_masks_driverb.sh
#   ※ AILAB-12 에는 slurm client 없음 — 사용자가 클러스터에서 제출.
#
#   부분처리: 전체 135k frame. 단일 GPU 로 길면 --start/--end 로 shard 분할하여
#            여러 job 으로 병렬 제출 가능 (아래 START/END env).
# =============================================================================
#SBATCH --job-name=scc_gen_seg_driverb
#SBATCH --comment=submitter:jiwon
#SBATCH --partition=partition-4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/home/ailab/AILabSSD/99_Management/slurm-logs/%x_%j.out
#SBATCH --error=/home/ailab/AILabSSD/99_Management/slurm-logs/%x_%j.err

set -euo pipefail

# ── 경로 (클러스터 prefix = /home/ailab) ─────────────────────────────────────
SIF_IMAGE=/home/ailab/AILabSSD/01_SlurmWorkspace/a1_e2e_fixed_v2.sif
# WORKSPACE = AILabSSD/03 으로 통일 (다른 wrapper·SETUP_RUNBOOK §2 와 일치).
WORKSPACE=/home/ailab/AILabSSD/03_Shared_Repository/jiwon/scc_controller_imitator
# DriverB raw 가 있는 NFS (Custom_Dataset). 00_raw 하위에 sangam_track/DriverB_new.
DATA2=/home/ailab/AILabDataset/02_Custom_Dataset/57_A1_E2E
# jiwon_e2e (gt_pipeline + GD/SAM2 weight). 클러스터 prefix.
E2E_ROOT=/home/ailab/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing
# 출력/메타가 위치할 SCC data dir (학습 캐시와 동일 위치).
SCC_DATA_DIR=/home/ailab/AILabSSD/02_Data/57_A1_E2E/01_processed/scc_controller_imitator/data

# shard 분할용 (전체 처리시 0 / 999999999). sbatch --export 로 override 가능.
START=${START:-0}
END=${END:-999999999}

[ -f "${SIF_IMAGE}" ] || { echo "SIF not found: ${SIF_IMAGE}"; exit 1; }
[ -d "${WORKSPACE}" ] || { echo "WORKSPACE(NFS repo) not found: ${WORKSPACE}"; exit 1; }
[ -d "${DATA2}/00_raw/sangam_track/DriverB_new" ] || { echo "DriverB raw not found under ${DATA2}/00_raw"; exit 1; }
[ -d "${E2E_ROOT}" ] || { echo "E2E_ROOT(jiwon_e2e) not found: ${E2E_ROOT}"; exit 1; }
[ -d "${SCC_DATA_DIR}" ] || { echo "SCC_DATA_DIR not found: ${SCC_DATA_DIR}"; exit 1; }

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

apptainer exec --nv \
    --bind "${WORKSPACE}:${WORKSPACE}" \
    --bind "${DATA2}:${DATA2}" \
    --bind "${E2E_ROOT}:${E2E_ROOT}" \
    --bind "${SCC_DATA_DIR}:${SCC_DATA_DIR}" \
    --env WORKSPACE="${WORKSPACE}" \
    --env SCC_DATA_DIR="${SCC_DATA_DIR}" \
    --env SCC_E2E_ROOT="${E2E_ROOT}" \
    --env DATA2="${DATA2}" \
    --env START="${START}" \
    --env END="${END}" \
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

# GD_CONFIG: a1_learning env 의 groundingdino site-packages cfg 경로 자동탐색.
GD_CFG=$(python -c "import os,groundingdino; print(os.path.join(os.path.dirname(groundingdino.__file__),'config','GroundingDINO_SwinB_cfg.py'))" 2>/dev/null || true)
if [ -n "${GD_CFG}" ] && [ -f "${GD_CFG}" ]; then
  export SCC_GD_CONFIG="${GD_CFG}"
  echo "GD_CONFIG = ${GD_CFG}"
else
  echo "WARN: GroundingDINO cfg auto-detect 실패 — 스크립트 기본값 사용 (.sif 에 deps 없으면 import 단계에서 실패)"
fi

# .sif SAM deps preflight (실패시 즉시 종료하여 원인 명확화)
python - <<'PY'
import sys
try:
    import torch, groundingdino, sam2  # noqa
    from src.data.gt_pipeline import N_CLASSES  # jiwon_e2e (SCC_E2E_ROOT on sys.path via script)
    print("preflight: torch/groundingdino/sam2 import OK")
except Exception as e:
    print("PREFLIGHT FAIL — .sif 에 SAM deps 부재 가능:", repr(e)); sys.exit(3)
PY

D="${SCC_DATA_DIR}"
META="$D/meta_DriverB_new.json"
RAW="${DATA2}/00_raw"
OUT="$D/seg_masks_DriverB_new"
[ -f "${META}" ] || { echo "meta not found: ${META}"; exit 1; }
mkdir -p "${OUT}"

echo "=== [gen seg] DriverB_new  start=${START} end=${END}  -> ${OUT} ==="
python -u scripts/generate_image_seg_masks.py \
  --meta "${META}" \
  --raw-img-base "${RAW}" \
  --out-dir "${OUT}" \
  --start "${START}" --end "${END}"
echo "=== DONE ==="
CONTAINER_EOF

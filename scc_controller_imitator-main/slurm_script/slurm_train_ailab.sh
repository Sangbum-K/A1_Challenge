#!/bin/bash
#SBATCH --job-name=misc
#SBATCH --comment=submitter:jiwon
#SBATCH --partition=partition-4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --time=100:00:00
#SBATCH --output=/home/ailab/AILabSSD/99_Management/slurm-logs/%x_%j.out
#SBATCH --error=/home/ailab/AILabSSD/99_Management/slurm-logs/%x_%j.err

set -euo pipefail

SIF_IMAGE=/home/ailab/AILabSSD/01_SlurmWorkspace/a1_e2e_fixed_v2.sif
WORKSPACE=/home/ailab/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator
DATA1=/home/ailab/AILabSSD/02_Data/57_A1_E2E
DATA2=/home/ailab/AILabDataset/02_Custom_Dataset/57_A1_E2E

[ -f "${SIF_IMAGE}" ] || { echo "SIF not found: ${SIF_IMAGE}"; exit 1; }

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

apptainer exec --nv \
    --bind "${WORKSPACE}:${WORKSPACE}" \
    --bind "${DATA1}:${DATA1}" \
    --bind "${DATA2}:${DATA2}" \
    "${SIF_IMAGE}" \
    bash <<'CONTAINER_EOF'
set -euo pipefail

echo "=== HOST ==="
hostname
echo "=== USER ==="
id

echo "=== GET INTO REPO ==="
cd /home/ailab/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator

echo "=== ACTIVATE ENV ==="
source /opt/conda/etc/profile.d/conda.sh
conda activate a1_learning
export PYTHONNOUSERSITE=True

echo "=== CUDA ENV ==="
export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export FORCE_CUDA=1

echo "=== RUN ==="
source /opt/conda/etc/profile.d/conda.sh && \
conda activate a1_learning && \
cd mmdetection3d && \
export PYTHONPATH=$(pwd):$PYTHONPATH && \
cd .. && \
python tools/analysis_tools/visualization_pickle_infos.py \
    data/pkl/sangam_agent_data_overall_moraiv2_infos_a1_train.pkl \
    --data_root . \
    --all_frames True \
    --num_frames 200 \
    --output_dir analysis_output/pickle_visualization/sangam_agent_data_overall
echo "=== DONE ==="

CONTAINER_EOF
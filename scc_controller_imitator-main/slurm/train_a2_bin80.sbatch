#!/bin/bash
# a_2 bin80 ablation 학습 (80bin × {10f,20f}) — AILAB SLURM (3090/4090/a6000) Singularity job.
# 데이터/코드/sif 는 AILabSSD(공유 NFS). 실행노드 기준 경로 = /home/ailab/AILabSSD/...
#SBATCH -J jiwon_a2bin80
#SBATCH -p partition-4090
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4090:1
#SBATCH -o /home/ailab/AILabSSD/01_SlurmWorkspace/jiwon_a2bin80/slurm_log/%j.out.log
#SBATCH -e /home/ailab/AILabSSD/01_SlurmWorkspace/jiwon_a2bin80/slurm_log/%j.err.log
set -e

WS=/home/ailab/AILabSSD/01_SlurmWorkspace/jiwon_a2bin80
SIF=/home/ailab/AILabSSD/01_SlurmWorkspace/scc_train.sif
mkdir -p ${WS}/slurm_log

echo "Nodelist: $SLURM_JOB_NODELIST  GPU: $SLURM_GPUS_PER_NODE  $(date)"

singularity run \
  --bind ${WS}/scc_controller_imitator:/workspace/scc \
  --bind ${WS}/data:/workspace/data \
  --nv ${SIF} \
  bash -c '
    set -e
    cd /workspace/scc
    export PYTHONPATH=/workspace/scc
    PY=/opt/conda/envs/scc_train/bin/python
    $PY -c "import torch;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available())"
    D=/workspace/data
    INIT=${D}/ckpts/a2_v7b_d128_alpha075_best_loss.pt
    OFF_10="10,20,30,40,50,60,70,80,90,100"
    OFF_20="5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100"
    CC="--i1-cache ${D}/cache/i1_tok_cache_sangam_lead_0520_hr.npz --aux-dir ${D}/aux_targets_sangam_lead_0520_bin80 \
        --i1-cache ${D}/cache/i1_tok_cache_sangam_lead_0521_hr.npz --aux-dir ${D}/aux_targets_sangam_lead_0521_bin80 \
        --i1-cache ${D}/cache/i1_tok_cache_spd_v2_hr.npz --aux-dir ${D}/aux_targets_spd_nolead_0520_bin80 \
        --yaw-shift off --resume-from ${INIT} --hr --d-token 256 --d-out 128 --epochs 12 --lr 1e-4 \
        --weight-decay 1e-4 --workers 8 --val-ratio 0.05 --w-obj 1.0 --w-reg 1.0 \
        --use-focal --focal-alpha 0.75 --focal-gamma 2.0 --scale-v 20.0 --scale-dist 40.0 \
        --lds --fx 640 --seed 0 --out-dir ${D}/a2_bin80"
    echo "### [1/2] bin80_s10n10 (1.0s,10f) $(date)"
    $PY scripts/train_a2_multiframe.py ${CC} --batch-size 128 --prev-offsets "${OFF_10}" --name bin80_s10n10
    echo "### [2/2] bin80_s10n20 (1.0s,20f) $(date)"
    $PY scripts/train_a2_multiframe.py ${CC} --batch-size 64  --prev-offsets "${OFF_20}" --name bin80_s10n20
    echo "### ALL DONE $(date)"
  '

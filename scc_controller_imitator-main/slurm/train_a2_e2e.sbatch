#!/bin/bash
# a_1(seg backbone) + a_2 end-to-end fine-tune (raw image → obj/aux) — AILAB SLURM Singularity job.
# 기존 train_a2_bin80.sbatch (cache 학습) 와 달리 raw 이미지 bind 가 추가됨 (backbone 도 학습).
#SBATCH -J jiwon_a2e2e
#SBATCH -p partition-4090
#SBATCH -t 08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4090:1
#SBATCH -o /home/ailab/AILabSSD/01_SlurmWorkspace/jiwon_a2e2e/slurm_log/%j.out.log
#SBATCH -e /home/ailab/AILabSSD/01_SlurmWorkspace/jiwon_a2e2e/slurm_log/%j.err.log
set -e

WS=/home/ailab/AILabSSD/01_SlurmWorkspace/jiwon_a2e2e
SIF=/home/ailab/AILabSSD/01_SlurmWorkspace/scc_train.sif
RAW=/home/ailab/AILabSSD/02_Data/57_A1_E2E/00_raw
mkdir -p ${WS}/slurm_log

echo "Nodelist: $SLURM_JOB_NODELIST  GPU: $SLURM_GPUS_PER_NODE  $(date)"

singularity run \
  --bind ${WS}/scc_controller_imitator:/workspace/scc \
  --bind ${WS}/data:/workspace/data \
  --bind ${RAW}:/workspace/raw \
  --nv ${SIF} \
  bash -c '
    set -e
    cd /workspace/scc
    export PYTHONPATH=/workspace/scc
    PY=/opt/conda/envs/scc_train/bin/python
    $PY -c "import torch;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available())"
    D=/workspace/data
    SEG=${D}/ckpts/seg_pretrain_fx640_v3_backbone.pt
    AUX="--aux-dir ${D}/aux_targets_sangam_lead_0520 \
         --aux-dir ${D}/aux_targets_sangam_lead_0521 \
         --aux-dir ${D}/aux_targets_spd_nolead_0520"
    COMMON="${AUX} --raw-base /workspace/raw --seg-ckpt ${SEG} \
        --d-token 256 --d-out 128 --epochs 8 --lr 1e-4 --backbone-lr-mult 0.1 \
        --weight-decay 1e-4 --workers 8 --val-ratio 0.05 --w-obj 1.0 --w-reg 1.0 \
        --use-focal --focal-alpha 0.75 --focal-gamma 2.0 --scale-v 20.0 --scale-dist 40.0 \
        --fx 640 --seed 0 --amp --out-dir ${D}/a2_e2e"
    echo "### [1/1] a2_e2e_n2 (curr + t-20)  $(date)"
    $PY scripts/train_a2_e2e.py ${COMMON} --n-frames 2 --prev-offsets 20 --batch-size 16 --name a2_e2e_n2
    echo "### ALL DONE $(date)"
  '

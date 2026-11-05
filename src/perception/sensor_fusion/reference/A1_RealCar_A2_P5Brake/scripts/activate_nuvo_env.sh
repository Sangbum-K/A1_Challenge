#!/usr/bin/env bash
# Source this file; do not execute it in a child shell.
_A1_EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_A1_SANGBUM_DIR="$(dirname "$(dirname "${_A1_EXPERIMENT_DIR}")")"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "ROS2 Humble was not found at /opt/ros/humble" >&2
  return 1
fi

source /opt/ros/humble/setup.bash
# This project can carry its own .venv; until then it borrows the verified one
# from the a2 throughput project next to it (same Nuvo, same torch/cu118 stack).
if [[ -f "${_A1_EXPERIMENT_DIR}/.venv/bin/activate" ]]; then
  source "${_A1_EXPERIMENT_DIR}/.venv/bin/activate"
elif [[ -f "${_A1_EXPERIMENT_DIR}/../A1_RealCamera_A2_21FPS/.venv/bin/activate" ]]; then
  source "${_A1_EXPERIMENT_DIR}/../A1_RealCamera_A2_21FPS/.venv/bin/activate"
else
  echo "no .venv: create one here or keep ../A1_RealCamera_A2_21FPS/.venv" >&2
  return 1
fi
export FASTRTPS_DEFAULT_PROFILES_FILE="${_A1_EXPERIMENT_DIR}/configs/fastdds_large_shm.xml"

# The compiler toolkit is local to sangbum. PyTorch's runtime/cuDNN remain isolated
# in .venv and are selected by the cu118 wheel's own runtime paths.
export CUDA_HOME="${_A1_SANGBUM_DIR}/cuda-11.8"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONNOUSERSITE=1
unset _A1_EXPERIMENT_DIR _A1_SANGBUM_DIR

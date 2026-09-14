#!/usr/bin/env bash
# SCC Controller Imitator 환경 셋업.
#
# 옵션 A. 기존 ml_con 환경 재사용 (권장)
#     conda activate ml_con
#     pip install -r requirements.txt
#
# 옵션 B. 새 격리 환경 scc_ml_con 생성
#     conda env create -f environment.yml
#     conda activate scc_ml_con
#
# 본 스크립트는 옵션 A 를 자동화한다.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_ENV="${SCC_ENV:-ml_con}"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${TARGET_ENV}"; then
    echo "[setup_env] '${TARGET_ENV}' env not found. 새 env 가 필요하면 environment.yml 사용."
    exit 1
fi

conda activate "${TARGET_ENV}"
echo "[setup_env] active env: $(python -c 'import sys; print(sys.executable)')"
python -m pip install --upgrade pip
python -m pip install -r "${REPO_ROOT}/requirements.txt"

echo "[setup_env] sanity check..."
python "${REPO_ROOT}/scripts/sanity_check_env.py"
echo "[setup_env] done."

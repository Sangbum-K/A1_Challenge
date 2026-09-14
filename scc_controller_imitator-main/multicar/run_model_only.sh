#!/usr/bin/env bash
# 예선 방식 단일 모델 구동 — multicar(namespace/ini/bridge 세트) 말고 closed_loop 하나만.
#
# 예선 때처럼 namespace 없이 /app/... topic 직결(closed_loop_node.py 하위호환: CAR_NS 미설정).
# moraibridge 는 이 스크립트가 안 띄운다 — 별도로 단일 bridge(namespace 없음)가 떠 있어야
# closed_loop 가 camera/ego topic 을 받는다. 모델만 prefix 로 골라 구동한다.
#   (multicar namespace 가 필요하면 CAR_NS=CarN env 로 호출 → /CarN/app 으로 prefix)
#
# 모델 선택 = prefix (run_multicar.sh 의 emit_model_env 매핑 재사용 → 단일 출처).
#
# 사용:
#   bash multicar/run_model_only.sh scc_p3_f654              # sangam/mild
#   bash multicar/run_model_only.sh baseline speedway high
#   bash multicar/run_model_only.sh qualifier
#   bash multicar/run_model_only.sh scc_p3_ng_gate_f654     # delta(no_gate) + 시계열 beat gate
#   prefix: qualifier | acc_gate | baseline | cmain | scc_p{1-5}_f654 | scc_p{1-5}_f656
#           | scc_p{1-5}_ng_f65{4,6} (gate 없음) | scc_p{1-5}_ng_gate_f65{4,6} (delta+시계열 beat gate)
#   env:  PYTHON=<python> 강제. CAR_NS=CarN (선택; 미설정이면 예선 단일차 namespace 없음).
set -e
SCC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${1:?usage: $0 <prefix> [track=sangam] [map=mild]}"
export MULTICAR_TRACK="${2:-sangam}"
export MULTICAR_MAP="${3:-mild}"

# python 선택 (run_closed_loop_prof635.sh 와 동일: PYTHON > 활성 conda > ml_con > python3)
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
    if [[ -n "$CONDA_PREFIX" && -x "$CONDA_PREFIX/bin/python" ]] && "$CONDA_PREFIX/bin/python" -c "import torch" 2>/dev/null; then
        PY="$CONDA_PREFIX/bin/python"
    else
        for p in "$HOME"/miniforge3/envs/ml_con/bin/python "$HOME"/miniconda3/envs/ml_con/bin/python "$HOME"/anaconda3/envs/ml_con/bin/python; do
            [[ -x "$p" ]] && "$p" -c "import torch" 2>/dev/null && { PY="$p"; break; }
        done
        [[ -z "$PY" ]] && PY=python3
    fi
fi
"$PY" -c "import torch" 2>/dev/null || { echo "[err] '$PY' 에 torch 없음. 'conda activate <torch env>' 후 재실행하거나 PYTHON=<python> 지정." >&2; exit 1; }

# emit_model_env 재사용 (run_multicar.sh 의 prefix→모델 env 매핑과 단일 출처).
source <(sed -n '/^emit_model_env()/,/^}/p' "$SCC_ROOT/multicar/run_multicar.sh")
MODEL_ENV="$(emit_model_env "$PREFIX")" || { echo "[err] 알 수 없는 prefix: $PREFIX (qualifier|acc_gate|baseline|cmain|scc_p{1-5}_f654|scc_p{1-5}_f656|scc_p{1-5}_ng_f65{4,6}|scc_p{1-5}_ng_gate_f65{4,6})" >&2; exit 1; }

source /opt/ros/noetic/setup.bash 2>/dev/null || true
export PYTHONPATH="$SCC_ROOT:$SCC_ROOT/external:$PYTHONPATH"
echo "[model-only] python = $PY"
echo "[model-only] prefix=$PREFIX  track=$MULTICAR_TRACK map=$MULTICAR_MAP  CAR_NS=${CAR_NS:-<none, 예선 단일차 /app/...>}"
echo "[model-only] model env = $MODEL_ENV"
echo "[model-only] bridge 미기동 — 단일 moraibridge 가 별도로 떠 있어야 camera/ego topic 수신"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "(no GPU)"

cd "$SCC_ROOT"
# CAR_NS 는 호출 env 상속 (미설정=예선 단일차, 설정 시 multicar namespace). 강제하지 않음.
# shellcheck disable=SC2086
export $MODEL_ENV
exec "$PY" -u scripts/closed_loop_node.py \
    _debug_viz:=false _debug_viz_seg:=false _debug_viz_a2:=false

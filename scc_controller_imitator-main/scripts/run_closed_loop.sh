#!/usr/bin/env bash
# closed-loop ROS test node launcher.
#
# 사용:
#   bash scripts/run_closed_loop.sh             # debug viz off
#   bash scripts/run_closed_loop.sh --viz       # debug viz on (matplotlib)
#   bash scripts/run_closed_loop.sh --viz --rosmaster http://localhost:11311

set -e

# ---------------- paths ----------------
SCC_ROOT="/home/ailab-12/git/scc_controller_imitator"
A1_ROOT="/home/ailab-12/git/A1_Rule"
PY="/home/ailab-12/miniforge3/envs/ml_con/bin/python"

# ---------------- args ----------------
DEBUG_VIZ="false"; DEBUG_VIZ_SEG="false"; DEBUG_VIZ_A2="false"
ROS_MASTER=""
for arg in "$@"; do
    case "$arg" in
        --viz)       DEBUG_VIZ="true" ;;
        --viz-seg)   DEBUG_VIZ="true"; DEBUG_VIZ_SEG="true" ;;
        --viz-a2)    DEBUG_VIZ="true"; DEBUG_VIZ_A2="true" ;;
        --viz-all)   DEBUG_VIZ="true"; DEBUG_VIZ_SEG="true"; DEBUG_VIZ_A2="true" ;;
        --rosmaster=*) ROS_MASTER="${arg#--rosmaster=}" ;;
        -h|--help)
            cat <<USAGE
usage: $0 [--viz | --viz-seg | --viz-a2 | --viz-all] [--rosmaster=URL]
  --viz       debug viz on (base panels)
  --viz-seg   + seg head overlay (image_curr 위 4-class color)
  --viz-a2    + a_2 head overlay (image_curr 위 per-bin dist/Δv)
  --viz-all   = --viz --viz-seg --viz-a2
USAGE
            exit 0
            ;;
        *) echo "unknown arg: $arg" >&2; exit 1 ;;
    esac
done

# ---------------- env ----------------
echo "[closed_loop] sourcing ROS env (autohyu_msgs 내제화 완료, A1_Rule 의존성 제거)"
# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash 2>/dev/null || true
# A1_Rule devel source 도 가능 (but 불필요)
# source "$A1_ROOT/devel/setup.bash"

# scc_controller_imitator src + external/autohyu_msgs import
export PYTHONPATH="$SCC_ROOT:$SCC_ROOT/external:$PYTHONPATH"

# ROS Master override
if [[ -n "$ROS_MASTER" ]]; then
    export ROS_MASTER_URI="$ROS_MASTER"
    echo "[closed_loop] ROS_MASTER_URI = $ROS_MASTER"
fi

# CUDA — 학습 동시 진행시 GPU 메모리 점유 확인용
echo "[closed_loop] GPU status:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "(no GPU)"

echo "[closed_loop] launching node (debug_viz=$DEBUG_VIZ seg=$DEBUG_VIZ_SEG a2=$DEBUG_VIZ_A2)"
cd "$SCC_ROOT"
exec "$PY" -u scripts/closed_loop_node.py \
    _debug_viz:="$DEBUG_VIZ" \
    _debug_viz_seg:="$DEBUG_VIZ_SEG" \
    _debug_viz_a2:="$DEBUG_VIZ_A2"

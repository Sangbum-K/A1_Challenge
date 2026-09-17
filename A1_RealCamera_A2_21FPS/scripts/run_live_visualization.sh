#!/usr/bin/env bash
set -eo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SECONDS="${1:-600}"
OUTPUT_DIR="${2:-${EXPERIMENT_DIR}/outputs/real_camera_visual_$(date +%Y%m%d_%H%M%S)}"
EXPOSURE_US="${3:-20000}"
LOG_FULL_A2="${4:-1}"
SAVE_OVERLAY_VIDEO="${5:-0}"
LOG_ARGS=()
if [[ "${LOG_FULL_A2}" != "0" ]]; then
  LOG_ARGS+=(--log-full-a2)
fi
VIDEO_ARGS=()
if [[ "${SAVE_OVERLAY_VIDEO}" != "0" ]]; then
  VIDEO_ARGS+=(--save-overlay-video)
fi

source "${EXPERIMENT_DIR}/scripts/activate_nuvo_env.sh"
mkdir -p "${OUTPUT_DIR}"

if pgrep -f '[c]amera_driver_gv' >/dev/null; then
  echo "camera_driver_gv is already running; stop the old process first" >&2
  exit 1
fi

ros2 launch "${EXPERIMENT_DIR}/launch/hikrobot_a2_camera_1280x720.launch.py" \
  guid:=192.168.1.51 \
  exposure_us:="${EXPOSURE_US}" >"${OUTPUT_DIR}/camera_driver.log" 2>&1 &
DRIVER_PID=$!

cleanup() {
  kill "${DRIVER_PID}" 2>/dev/null || true
  wait "${DRIVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

TOPIC=/camera_driver_gv/vis/image_raw
for _ in {1..10}; do
  if ros2 topic list 2>/dev/null | rg -qx "${TOPIC}"; then
    break
  fi
  sleep 1
done
if ! ros2 topic list 2>/dev/null | rg -qx "${TOPIC}"; then
  echo "camera topic did not appear: ${TOPIC}" >&2
  tail -80 "${OUTPUT_DIR}/camera_driver.log" >&2
  exit 1
fi

echo "[visual] output=${OUTPUT_DIR}"
echo "[visual] camera exposure=${EXPOSURE_US} us"
echo "[visual] press q or Esc in the OpenCV window to stop"
python "${EXPERIMENT_DIR}/scripts/live_camera_a2_21fps.py" \
  --input ros2 \
  --ros-topic "${TOPIC}" \
  --camera-fps 21 \
  --target-fps 21 \
  --width 1280 \
  --height 720 \
  --geometry-note "1280x720 centered hardware crop; FOV/calibration not yet validated" \
  --seg-ckpt "${EXPERIMENT_DIR}/senior_code/data/ckpts/seg_pretrain_fx640_v3_backbone.pt" \
  --a2-ckpt "${EXPERIMENT_DIR}/senior_code/data/ckpts/a2_mf20_v2s_n20.pt" \
  --warmup-iters 10 \
  --duration "${RUN_SECONDS}" \
  --display \
  "${LOG_ARGS[@]}" \
  "${VIDEO_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}"

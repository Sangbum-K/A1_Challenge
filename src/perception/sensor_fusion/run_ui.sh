#!/usr/bin/env bash
# C++ live display; this launcher never transmits vehicle commands.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve this workspace independently of a previously sourced workspace.
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH
unset ROS_NAMESPACE ROS_DISCOVERY_SERVER FASTDDS_DEFAULT_PROFILES_FILE CYCLONEDDS_URI
# A1_Total layout: <ws>/src/perception/sensor_fusion, merged install at <ws>/install.
WS="$(cd "${ROOT}/../../.." && pwd)"
source "$WS/config/ros.env"
SOURCE_LOCALHOST="$ROS_LOCALHOST_ONLY"
case "$A1_FUSION_TRANSPORT" in
  direct)
    if [[ "$A1_LIDAR_SOURCE_DOMAIN" != "$ROS_DOMAIN_ID" ]]; then
      echo "Direct mode requires A1_LIDAR_SOURCE_DOMAIN=ROS_DOMAIN_ID." >&2; exit 1
    fi
    LIDAR_INPUT="$A1_LIDAR_SOURCE_TOPIC"
    DDS_PROFILE="$ROOT/cpp/config/fastdds_large_images.xml"
    ;;
  bridge)
    export ROS_DOMAIN_ID=107 ROS_LOCALHOST_ONLY=1
    if [[ "$A1_LIDAR_SOURCE_DOMAIN" == 107 ]]; then echo "Bridge source must differ from 107." >&2; exit 1; fi
    LIDAR_INPUT=/lead_fusion/input/lidar_points
    DDS_PROFILE="$ROOT/cpp/config/fastdds_isolated_ui.xml"
    ;;
  *) echo "A1_FUSION_TRANSPORT must be direct or bridge" >&2; exit 1;;
esac
INSTALL="${WS}/install"
source /opt/ros/humble/setup.bash
source "${INSTALL}/setup.bash"
export PYTHONPATH="${INSTALL}/local/lib/python3.10/dist-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${INSTALL}/lib:${WS}/third_party/libtorch/lib:${CUDA_HOME:-${WS}/third_party/cuda-11.8}/lib64:${LD_LIBRARY_PATH:-}"
export FASTRTPS_DEFAULT_PROFILES_FILE="$DDS_PROFILE"
OUT="${A1_UI_OUT:-${ROOT}/outputs/ui_$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT" "$ROOT/outputs"
exec 9>"$ROOT/outputs/ui.lock"
if ! flock -n 9; then echo "UI already running. Use ./run_sensor_fusion.sh status or stop." >&2; exit 1; fi
if [[ -z "${DISPLAY:-}" ]]; then
  if [[ -S /tmp/.X11-unix/X1 ]]; then export DISPLAY=:1
  elif [[ -S /tmp/.X11-unix/X0 ]]; then export DISPLAY=:0
  else echo "No desktop display available." >&2; exit 1; fi
fi
export ROS_LOG_DIR="$OUT/ros_logs"
# DDS isolation cannot isolate physical sensor ownership.
CAMERA_ENABLED="${A1_CAMERA_ENABLED:-1}"
if [[ "$CAMERA_ENABLED" != 0 && "$CAMERA_ENABLED" != 1 ]]; then echo "A1_CAMERA_ENABLED must be 0 or 1" >&2; exit 1; fi
CAMERA_MODE="${A1_CAMERA_MODE:-$([[ "$CAMERA_ENABLED" == 0 ]] && echo off || echo start)}"
case "$CAMERA_MODE" in start|shared|off) ;; *) echo "A1_CAMERA_MODE must be start, shared or off" >&2; exit 1;; esac
if [[ "$CAMERA_MODE" == shared && "$A1_FUSION_TRANSPORT" != direct ]]; then
  echo "Shared camera mode requires direct transport in the camera publisher domain." >&2; exit 1
fi
if [[ "$CAMERA_MODE" == start ]] && { pgrep -x camera_driver_g >/dev/null || pgrep -f '^/[^ ]*/camera_aravis2/camera_driver_gv( |$)' >/dev/null; }; then
  echo "Camera is already owned by another process; not starting a second driver." >&2; exit 1
fi
# Only consume the localization-owned stream. Never bind UDP 2368 here.
LIDAR_SOURCE_DOMAIN="$A1_LIDAR_SOURCE_DOMAIN"
LIDAR_SOURCE_TOPIC="$A1_LIDAR_SOURCE_TOPIC"
BRIDGE="$INSTALL/lib/a1_fusion_cpp/lidar_domain_bridge"
if [[ "$A1_FUSION_TRANSPORT" == bridge && ! -x "$BRIDGE" ]]; then echo "Build lidar_domain_bridge first." >&2; exit 1; fi
# Same subscriber contract in both modes; no driver launch and no ROS CLI daemon.
env ROS_DOMAIN_ID="$LIDAR_SOURCE_DOMAIN" ROS_LOCALHOST_ONLY="${A1_ROS_LOCALHOST_ONLY:-$SOURCE_LOCALHOST}" FASTRTPS_DEFAULT_PROFILES_FILE="$ROOT/cpp/config/fastdds_large_images.xml" \
  python3 "$WS/tools/topic_audit.py" --seconds 6 --lidar-topic "$LIDAR_SOURCE_TOPIC" --check-lidar >"$OUT/lidar_preflight.log" 2>&1 || {
  cat "$OUT/lidar_preflight.log" >&2
  echo "Shared LiDAR input unavailable. No sensor drivers started." >&2; exit 1;
}
printf 'ROS_DOMAIN_ID=%s\nROS_LOCALHOST_ONLY=%s\nA1_FUSION_TRANSPORT=%s\nLIDAR_INPUT=%s\n' "$ROS_DOMAIN_ID" "$ROS_LOCALHOST_ONLY" "$A1_FUSION_TRANSPORT" "$LIDAR_INPUT" >"$OUT/isolation.env"
cp "$ROOT/cpp/config/association.yaml" "$OUT/association_used.yaml"
python3 "$ROOT/cpp/tools/isolation_preflight.py"
# Avoid taking over or mixing with an already running sensor/pipeline.
for topic in /camera_driver_gv/vis/image_raw /lead_fusion/input/lidar_points /lead_fusion/a2 /lead_fusion/association /lead_fusion/estimate /lead_fusion/ego_motion; do
  info="$(ros2 topic info "$topic" --no-daemon --spin-time 1 2>/dev/null || true)"
  if [[ "$topic" == /camera_driver_gv/vis/image_raw && "$CAMERA_MODE" == shared ]]; then
    if [[ ! "$info" =~ Publisher\ count:\ 1([^0-9]|$) ]]; then
      echo "Shared camera requires exactly one image publisher in domain $ROS_DOMAIN_ID." >&2; exit 1
    fi
    continue
  fi
  if [[ "$info" =~ Publisher\ count:\ ([1-9][0-9]*) ]]; then
    echo "Already running: $topic. Stop that run or choose its intended ROS_DOMAIN_ID." >&2
    exit 1
  fi
done
pids=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do kill -INT -- "-$pid" 2>/dev/null || true; done
  for attempt in {1..30}; do
    alive=false
    for pid in "${pids[@]}"; do if kill -0 "$pid" 2>/dev/null; then alive=true; fi; done
    if [[ "$alive" == false ]]; then break; fi
    sleep 0.2
  done
  for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  sleep 1
  for pid in "${pids[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  rm -f "$ROOT/outputs/ui.pid"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf '%s\n' "$$" > "$ROOT/outputs/ui.pid"
printf '%s\n' "$OUT" > "$ROOT/outputs/ui_latest_path.txt"
launch() {
  local name="$1"; shift
  setsid "$@" >"$OUT/$name.log" 2>&1 &
  pids+=("$!")
}
if [[ "$CAMERA_MODE" == start ]]; then
launch camera bash "$ROOT/run.sh" camera-driver-gpu exposure_auto:=Off exposure_us:="${A1_CAMERA_EXPOSURE_US:-35000}" gain_auto:=Off gain_db:=0
fi
if [[ "$CAMERA_MODE" != off ]]; then
launch a2 bash "$ROOT/run.sh" camera-gpu
fi
if [[ "$A1_FUSION_TRANSPORT" == bridge ]]; then
launch lidar_bridge env A1_BRIDGE_TARGET_PROFILE="$ROOT/cpp/config/fastdds_isolated_ui.xml" ROS_LOCALHOST_ONLY=0 FASTRTPS_DEFAULT_PROFILES_FILE="$ROOT/cpp/config/fastdds_large_images.xml" "$BRIDGE" "$LIDAR_SOURCE_DOMAIN" "$LIDAR_SOURCE_TOPIC" 107
fi
launch shadow env A1_CPP_INSTALL="$INSTALL" bash "$ROOT/run.sh" fusion -p "lidar_topic:=$LIDAR_INPUT"
launch ui bash "$ROOT/run.sh" association -p "lidar_topic:=$LIDAR_INPUT" -p "csv_path:=$OUT/association.csv" -p show_window:=true -p "preview_path:=$OUT/latest.jpg" -p "recording_dir:=$OUT/recordings" "$@"
echo "Shared LiDAR source: domain=$LIDAR_SOURCE_DOMAIN topic=$LIDAR_SOURCE_TOPIC (external owner)."
echo "UI running: transport=$A1_FUSION_TRANSPORT domain=$ROS_DOMAIN_ID input=$LIDAR_INPUT camera=$CAMERA_MODE"
echo "Ego requires /a1_gnss/odom plus GNSS status in this domain; see shadow.yaml verification gates. Bridge forwards LiDAR only."
echo "Logs, screenshot and continuous MP4: $OUT"
echo "Ctrl+C closes only the processes started here. No CAN output."
wait -n "${pids[@]}"

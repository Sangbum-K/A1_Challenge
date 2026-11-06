#!/usr/bin/env bash
# C++ live display; this launcher never transmits vehicle commands.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Deliberately ignore a sourced teammate workspace/domain.
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH
unset ROS_NAMESPACE ROS_DISCOVERY_SERVER FASTDDS_DEFAULT_PROFILES_FILE CYCLONEDDS_URI
export ROS_DOMAIN_ID=107 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source /opt/ros/humble/setup.bash
source "${ROOT}/install/share/hesai_ros_driver/local_setup.bash"
source "${ROOT}/install_gpu/share/gsof_msgs/local_setup.bash"
source "${ROOT}/install_gpu/share/a1_fusion_cpp/local_setup.bash"
export AMENT_PREFIX_PATH="${ROOT}/install_gpu:${AMENT_PREFIX_PATH:-}"
export PYTHONPATH="${ROOT}/install_gpu/local/lib/python3.10/dist-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${ROOT}/install_gpu/lib:${ROOT}/vendor/libtorch/lib:${ROOT}/../../cuda-11.8/lib64:${LD_LIBRARY_PATH:-}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${ROOT}/cpp/config/fastdds_isolated_ui.xml"
OUT="${A1_UI_OUT:-${ROOT}/outputs/ui_$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
exec 9>"$ROOT/outputs/ui.lock"
if ! flock -n 9; then echo "UI already running. Use ./run_sensor_fusion.sh status or stop." >&2; exit 1; fi
if [[ -z "${DISPLAY:-}" ]]; then
  if [[ -S /tmp/.X11-unix/X1 ]]; then export DISPLAY=:1
  elif [[ -S /tmp/.X11-unix/X0 ]]; then export DISPLAY=:0
  else echo "No desktop display available." >&2; exit 1; fi
fi
export ROS_LOG_DIR="$OUT/ros_logs"
# DDS isolation cannot isolate physical sensor ownership.
if pgrep -x camera_driver_g >/dev/null || pgrep -f '^/[^ ]*/camera_aravis2/camera_driver_gv( |$)' >/dev/null; then
  echo "Camera is already owned by another process; not starting a second driver." >&2; exit 1
fi
if ss -H -lun 'sport = :2368' | read -r _; then
  echo "LiDAR UDP 2368 is already in use; not starting a second driver." >&2; exit 1
fi
printf 'ROS_DOMAIN_ID=107\nROS_LOCALHOST_ONLY=1\nRMW_IMPLEMENTATION=rmw_fastrtps_cpp\n' >"$OUT/isolation.env"
cp "$ROOT/cpp/config/association.yaml" "$OUT/association_used.yaml"
python3 "$ROOT/cpp/tools/isolation_preflight.py"
# Avoid taking over or mixing with an already running sensor/pipeline.
for topic in /camera_driver_gv/vis/image_raw /lidar_points /lead_fusion/a2 /lead_fusion/association /lead_fusion/estimate /lead_fusion/ego_motion; do
  info="$(ros2 topic info "$topic" 2>/dev/null || true)"
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
launch camera bash "$ROOT/run.sh" camera-driver-gpu exposure_auto:=Continuous exposure_min_us:=1000 exposure_max_us:=20000 gain_auto:=Continuous gain_max_db:=12
launch lidar "$ROOT/install/lib/hesai_ros_driver/hesai_ros_driver_node" --ros-args -p "config_path:=$ROOT/cpp/config/lidar_host_time.yaml"
launch a2 bash "$ROOT/run.sh" camera-gpu
launch shadow env A1_CPP_INSTALL="$ROOT/install_gpu" bash "$ROOT/run.sh" fusion
launch ui bash "$ROOT/run.sh" association -p show_window:=true -p "preview_path:=$OUT/latest.jpg" -p "recording_dir:=$OUT/recordings" "$@"
echo "UI running. ROS_DOMAIN_ID=$ROS_DOMAIN_ID; GNSS IMU must use this domain."
echo "Logs, screenshot and continuous MP4: $OUT"
echo "Ctrl+C closes only the processes started here. No CAN output."
wait -n "${pids[@]}"

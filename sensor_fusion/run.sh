#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-help}"
if (($#)); then shift; fi
# A1_Total layout: this project lives at <ws>/src/perception/sensor_fusion and the
# ROS package (cpp/) is built by colcon into the merged workspace install.
# A1_Challenge keeps it at <repo>/sensor_fusion, where the repo root is <ws>.
if [[ "$ROOT" == */src/perception/sensor_fusion ]]; then WS="$(cd "${ROOT}/../../.." && pwd)"; else WS="$(cd "${ROOT}/.." && pwd)"; fi
BUILD="${A1_CPP_BUILD:-${WS}/build/sensor_fusion_manual}"
INSTALL="${A1_CPP_INSTALL:-${WS}/install}"
LIBTORCH="${A1_LIBTORCH:-${WS}/third_party/libtorch}"
export CUDA_HOME="${CUDA_HOME:-${WS}/third_party/cuda-11.8}"
ros_env() {
  source /opt/ros/humble/setup.bash
  if [[ ! -f "${INSTALL}/setup.bash" ]]; then echo "Workspace not built: run ${WS}/build.sh" >&2; exit 1; fi
  source "${INSTALL}/setup.bash"
  export PYTHONPATH="${INSTALL}/local/lib/python3.10/dist-packages:${PYTHONPATH:-}"
  export LD_LIBRARY_PATH="${INSTALL}/lib:${LD_LIBRARY_PATH:-}"
  export AMENT_PREFIX_PATH="${INSTALL}:${AMENT_PREFIX_PATH:-}"
}
case "$MODE" in
  target-test)
    # Synthetic inputs only, isolated from the live domain 107 and sensor domain 0.
    ros_env
    export ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    unset ROS_DISCOVERY_SERVER FASTDDS_DEFAULT_PROFILES_FILE CYCLONEDDS_URI
    export FASTRTPS_DEFAULT_PROFILES_FILE="${ROOT}/cpp/config/fastdds_isolated_ui.xml"
    test_out="${A1_TARGET_TEST_OUT:-${ROOT}/outputs/camera_target_tests}"
    mkdir -p "$test_out"
    export ROS_LOG_DIR="$test_out/ros_logs"
    ctest --test-dir "${WS}/build/a1_fusion_cpp" --output-on-failure
    ROS_DOMAIN_ID=93 python3 "${ROOT}/cpp/tests/association_smoke.py" --binary "${INSTALL}/lib/a1_fusion_cpp/association_node" --map-filter
    ROS_DOMAIN_ID=94 python3 "${ROOT}/cpp/tests/primary_p5_smoke.py" --binary "${INSTALL}/lib/a1_fusion_cpp/fusion_node"
    ROS_DOMAIN_ID=95 python3 "${ROOT}/cpp/tests/far_cloud_smoke.py" --binary "${INSTALL}/lib/a1_fusion_cpp/association_node" --config "${ROOT}/cpp/config/association.yaml" --output "$test_out/far_cloud.json"
    ;;
  lidar-check|topic-audit)
    source "$WS/config/ros.env"
    export ROS_DOMAIN_ID="$A1_LIDAR_SOURCE_DOMAIN"
    unset ROS_DISCOVERY_SERVER FASTDDS_DEFAULT_PROFILES_FILE CYCLONEDDS_URI
    export FASTRTPS_DEFAULT_PROFILES_FILE="${ROOT}/cpp/config/fastdds_large_images.xml"
    source /opt/ros/humble/setup.bash
    if [[ -f "$INSTALL/setup.bash" ]]; then source "$INSTALL/setup.bash"; fi
    audit_args=()
    if [[ "$MODE" == lidar-check ]]; then audit_args+=(--check-lidar); fi
    exec python3 "$WS/tools/topic_audit.py" --lidar-topic "$A1_LIDAR_SOURCE_TOPIC" "${audit_args[@]}" "$@"
    ;;
  lidar-rate)
    # Same domain/topic as lidar-check; measures the rate with a best_effort subscriber (see the tool's header).
    source "$WS/config/ros.env"
    export ROS_DOMAIN_ID="$A1_LIDAR_SOURCE_DOMAIN"
    unset ROS_DISCOVERY_SERVER FASTDDS_DEFAULT_PROFILES_FILE CYCLONEDDS_URI
    export FASTRTPS_DEFAULT_PROFILES_FILE="${ROOT}/cpp/config/fastdds_large_images.xml"
    source /opt/ros/humble/setup.bash
    exec python3 "${ROOT}/cpp/tools/lidar_rate_check.py" --topic "$A1_LIDAR_SOURCE_TOPIC" "$@"
    ;;
  build|build-gpu)
    # Built together with the rest of the workspace (GPU LibTorch, CUDA 11.8).
    exec bash "${WS}/build.sh" --packages-up-to a1_fusion_cpp "$@"
    ;;
  camera-gpu|benchmark-gpu|camera-driver-gpu|association|association-check)
    export LD_LIBRARY_PATH="${LIBTORCH}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-${ROOT}/cpp/config/fastdds_large_images.xml}"
    ros_env
    if [[ "$MODE" == association ]]; then
      exec "${INSTALL}/lib/a1_fusion_cpp/association_node" --ros-args --params-file "${ROOT}/cpp/config/calibration.yaml" --params-file "${ROOT}/cpp/config/association.yaml" "$@"
    fi
    if [[ "$MODE" == association-check ]]; then
      export ROS_LOG_DIR="${ROOT}/outputs/association_ros_logs"
      exec python3 "${ROOT}/cpp/tools/association_audit.py" "$@"
    fi
    if [[ "$MODE" == camera-driver-gpu ]]; then
      exec ros2 launch "${ROOT}/cpp/launch/hikrobot_gpu.launch.py" exposure_us:=3000 fps:=21 "$@"
    fi
    if [[ "$MODE" == camera-gpu ]]; then
      exec "${INSTALL}/lib/a1_fusion_cpp/a2_camera_node" --ros-args -p device:=cuda -p "model_dir:=${ROOT}/cpp/models_gpu" "$@"
    fi
    export ROS_LOG_DIR="${ROOT}/outputs/gpu_ros_logs"
    exec python3 "${ROOT}/cpp/tools/gpu_camera_benchmark.py" "$@"
    ;;
  drivers)
    # Hesai / Trimble drivers are workspace packages (src/external).
    exec bash "${WS}/build.sh" --packages-select gsof_msgs trimble_interfaces trimble_driver_ros hesai_ros_driver "$@"
    ;;
  sensor-check|sensor-check-gpu)
    if [[ "$MODE" == sensor-check-gpu ]]; then
      export LD_LIBRARY_PATH="${LIBTORCH}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
      export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-${ROOT}/cpp/config/fastdds_large_images.xml}"
    fi
    ros_env
    export ROS_LOG_DIR="${ROOT}/outputs/sensor_logs"
    if [[ "$MODE" == sensor-check-gpu ]]; then
      exec python3 "${ROOT}/cpp/tools/live_sensor_audit.py" --with-cpp --host-lidar-time \
        --device cuda --cpp-install "${INSTALL}" --models "${ROOT}/cpp/models_gpu" \
        --out "${ROOT}/outputs/sensor_check_gpu" "$@"
    fi
    exec python3 "${ROOT}/cpp/tools/live_sensor_audit.py" --with-cpp --host-lidar-time --out "${ROOT}/outputs/sensor_check" "$@"
    ;;
  core)
    cmake -S "${ROOT}/cpp" -B "${BUILD}/core" -DBUILD_ROS=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build "${BUILD}/core" -j2
    ctest --test-dir "${BUILD}/core" --output-on-failure
    ;;
  verify)
    bash "$0" core
    A1_P5_CLI="${BUILD}/core/p5_cli" python3 -B -m pytest -p no:cacheprovider -q "${ROOT}/cpp/tests/test_p5_reference.py" "${ROOT}/cpp/tests/test_track_eval.py" "${ROOT}/cpp/tests/test_map_surfaces_check.py" "${ROOT}/cpp/tests/test_lidar_rate_check.py"
    "${BUILD}/core/replay_p5_csv" \
      "${ROOT}/cpp/evidence/lidar_p5_20261008_10hz.csv" \
      "${ROOT}/cpp/evidence/p5_cpp_replay.csv"
    ;;
  export)
    python3 "${ROOT}/cpp/tools/export_a2.py" --out "${ROOT}/cpp/models" "$@"
    ;;
  camera)
    ros_env
    exec "${INSTALL}/lib/a1_fusion_cpp/a2_camera_node" --ros-args -p "model_dir:=${ROOT}/cpp/models" "$@"
    ;;
  fusion)
    ros_env
    exec "${INSTALL}/lib/a1_fusion_cpp/fusion_node" --ros-args --params-file "${ROOT}/cpp/config/calibration.yaml" --params-file "${ROOT}/cpp/config/shadow.yaml" "$@"
    ;;
  ros-test)
    ros_env
    export ROS_LOCALHOST_ONLY=1 ROS_LOG_DIR="${ROOT}/outputs/test_logs"
    ROS_DOMAIN_ID=87 python3 "${ROOT}/cpp/tests/ros_smoke.py" --binary "${INSTALL}/lib/a1_fusion_cpp/fusion_node"
    ROS_DOMAIN_ID=88 python3 "${ROOT}/cpp/tests/camera_smoke.py" --binary "${INSTALL}/lib/a1_fusion_cpp/a2_camera_node" --models "${ROOT}/cpp/models"
    ROS_DOMAIN_ID=89 python3 "${ROOT}/cpp/tests/imu_smoke.py" --binary "${INSTALL}/lib/a1_fusion_cpp/fusion_node"
    ROS_DOMAIN_ID=90 python3 "${ROOT}/cpp/tests/unverified_ego_smoke.py" --binary "${INSTALL}/lib/a1_fusion_cpp/fusion_node"
    # This smoke refuses domains below 100.
    ROS_DOMAIN_ID=101 python3 "${ROOT}/cpp/tests/gnss_ego_smoke.py" --binary "${INSTALL}/lib/a1_fusion_cpp/fusion_node"
    ;;
  replay)
    # Re-runs perception on a recorded bag: no sensors, no vehicle output.
    # The map server is not in the bag: set A1_MAP_SURFACES to the same .surfaces.pcd.
    bag="${1:-}"
    if [[ -z "$bag" || ! -d "$bag" ]]; then echo "Usage: bash run.sh replay <bag_dir> [--rerun-a2]" >&2; exit 1; fi
    shift
    rerun_a2=false
    if [[ "${1:-}" == --rerun-a2 ]]; then rerun_a2=true; shift; fi
    ros_env
    export ROS_DOMAIN_ID="${A1_REPLAY_DOMAIN:-96}" ROS_LOCALHOST_ONLY=1
    export FASTRTPS_DEFAULT_PROFILES_FILE="${ROOT}/cpp/config/fastdds_large_images.xml"
    out="${A1_REPLAY_OUT:-${bag%/}_replay_$(date +%Y%m%d_%H%M%S)}"
    mkdir -p "$out"
    export ROS_LOG_DIR="$out/ros_logs"
    pids=()
    trap 'for pid in "${pids[@]}"; do kill -INT "$pid" 2>/dev/null || true; done; wait 2>/dev/null || true' EXIT
    if [[ -n "${A1_MAP_SURFACES:-}" ]]; then
      "${INSTALL}/lib/a1_fusion_cpp/map_surface_server" --ros-args -p "surface_file:=${A1_MAP_SURFACES}" >"$out/map.log" 2>&1 &
      pids+=("$!")
    fi
    play_topics=(/camera_driver_gv/vis/image_raw /lidar_points /lead_fusion/input/lidar_points /a1_gnss/odom
                 /gsof_client/gsof/ins_solution_49 /tf /tf_static /lidar_odometry/pose_quality)
    if [[ "$rerun_a2" == true ]]; then
      export LD_LIBRARY_PATH="${LIBTORCH}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
      "${INSTALL}/lib/a1_fusion_cpp/a2_camera_node" --ros-args -p device:=cuda -p "model_dir:=${ROOT}/cpp/models_gpu" -p use_sim_time:=true >"$out/a2.log" 2>&1 &
      pids+=("$!")
    else
      play_topics+=(/lead_fusion/a2)
    fi
    "${INSTALL}/lib/a1_fusion_cpp/fusion_node" --ros-args --params-file "${ROOT}/cpp/config/calibration.yaml" --params-file "${ROOT}/cpp/config/shadow.yaml" \
      -p use_sim_time:=true -p "csv_path:=$out/fusion.csv" >"$out/fusion.log" 2>&1 &
    pids+=("$!")
    "${INSTALL}/lib/a1_fusion_cpp/association_node" --ros-args --params-file "${ROOT}/cpp/config/calibration.yaml" --params-file "${ROOT}/cpp/config/association.yaml" \
      -p use_sim_time:=true -p "csv_path:=$out/association.csv" -p "recording_dir:=$out/recordings" "$@" >"$out/association.log" 2>&1 &
    pids+=("$!")
    sleep 3
    # Map freshness checks use wall time, so the rate stays at 1.0.
    ros2 bag play "$bag" --clock --rate 1.0 --topics "${play_topics[@]}"
    echo "Replay results: $out"
    ;;
  *)
    echo 'Usage: bash run.sh lidar-check|lidar-rate|topic-audit|core|build|build-gpu|drivers|verify|export|camera|camera-driver-gpu|camera-gpu|benchmark-gpu|association|association-check|fusion|ros-test|target-test|replay|sensor-check|sensor-check-gpu'
    ;;
esac

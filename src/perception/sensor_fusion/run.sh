#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-help}"
if (($#)); then shift; fi
# A1_Total layout: this project lives at <ws>/src/perception/sensor_fusion and the
# ROS package (cpp/) is built by colcon into the merged workspace install.
WS="$(cd "${ROOT}/../../.." && pwd)"
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
    ROS_DOMAIN_ID=93 python3 "${ROOT}/cpp/tests/association_smoke.py" --binary "${INSTALL}/lib/a1_fusion_cpp/association_node"
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
  build|build-gpu)
    # Built together with the rest of the workspace (GPU LibTorch, CUDA 11.8).
    exec bash "${WS}/build.sh" --packages-up-to a1_fusion_cpp "$@"
    ;;
  camera-gpu|benchmark-gpu|camera-driver-gpu|association|association-check)
    export LD_LIBRARY_PATH="${LIBTORCH}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-${ROOT}/cpp/config/fastdds_large_images.xml}"
    ros_env
    if [[ "$MODE" == association ]]; then
      exec "${INSTALL}/lib/a1_fusion_cpp/association_node" --ros-args --params-file "${ROOT}/cpp/config/association.yaml" "$@"
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
    A1_P5_CLI="${BUILD}/core/p5_cli" python3 -B -m pytest -p no:cacheprovider -q "${ROOT}/cpp/tests/test_p5_reference.py" "${ROOT}/reference/A1_RealCar_A2_P5Brake/tests"
    python3 "${ROOT}/reference/A1_RealCar_A2_P5Brake/scripts/verify_portable_bundle.py"
    "${BUILD}/core/replay_p5_csv" \
      "${ROOT}/reference/A1_LeadVehicle_LidarFusion/evidence/lidar_p5_20261008_0300_1200/p5_10hz.csv" \
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
    exec "${INSTALL}/lib/a1_fusion_cpp/fusion_node" --ros-args --params-file "${ROOT}/cpp/config/shadow.yaml" "$@"
    ;;
  ros-test)
    ros_env
    export ROS_LOCALHOST_ONLY=1 ROS_LOG_DIR="${ROOT}/outputs/test_logs"
    ROS_DOMAIN_ID=87 python3 "${ROOT}/cpp/tests/ros_smoke.py" --binary "${INSTALL}/lib/a1_fusion_cpp/fusion_node"
    ROS_DOMAIN_ID=88 python3 "${ROOT}/cpp/tests/camera_smoke.py" --binary "${INSTALL}/lib/a1_fusion_cpp/a2_camera_node" --models "${ROOT}/cpp/models"
    ROS_DOMAIN_ID=89 python3 "${ROOT}/cpp/tests/imu_smoke.py" --binary "${INSTALL}/lib/a1_fusion_cpp/fusion_node"
    ;;
  *)
    echo 'Usage: bash run.sh lidar-check|topic-audit|core|build|build-gpu|drivers|verify|export|camera|camera-driver-gpu|camera-gpu|benchmark-gpu|association|association-check|fusion|ros-test|target-test|sensor-check|sensor-check-gpu'
    ;;
esac

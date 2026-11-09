#!/usr/bin/env bash
# 카메라-LiDAR extrinsic 보정용 bag 한 개 녹화. 사용: bash ~/calib_data/20261109/rec.sh scene01
# 0~3 s 정지 → "밀기 시작" 표시 → 30 s 녹화 후 자동 종료
set -euo pipefail
name="${1:?장소 이름 (예: scene01)}"
ws="$(cd "$(dirname "$0")" && pwd)"
set +u  # ROS setup 스크립트는 정의 안 된 변수를 읽는다
source /opt/ros/humble/setup.bash
source "${MIRU_WS:-$HOME/A1-Challenge-2026-main}/install/setup.bash"
set -u
export FASTRTPS_DEFAULT_PROFILES_FILE="${MIRU_WS:-$HOME/A1-Challenge-2026-main}/src/app/perception/lead_vehicle_fusion/config/fastdds_large_images.xml"
mkdir -p "$ws/moving_bags"
[[ -e "$ws/moving_bags/$name" ]] && { echo "$name 이 이미 있다. 다른 이름을 쓴다."; exit 1; }
echo "녹화 시작. 차를 건드리지 말 것 (3초)"
ros2 bag record -o "$ws/moving_bags/$name" /sensor/camera/image_raw /sensor/camera/camera_info /sensor/lidar/points \
  > "$ws/moving_bags/${name}_record.log" 2>&1 &
pid=$!
sleep 3
printf '\a'; echo ">>> 밀기 시작: 천천히 앞으로, 보닛을 눌러 위아래로 흔들기 (30초)"
for s in $(seq 30 -5 5); do echo "  남은 시간 ${s}s"; sleep 5; done
kill -INT "$pid"; wait "$pid" || true
printf '\a'; echo "녹화 끝: $ws/moving_bags/$name"

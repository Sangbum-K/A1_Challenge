#!/usr/bin/env bash
# moraibridge 한 번에 기동 — 3노드(vehicle_driver / vehicle_control / camera_driver) 동시 실행.
#
# 빌드(최초 1회): cd moraibridge && catkin_make
# 실행:          bash moraibridge/run_bridge.sh
#                bash moraibridge/run_bridge.sh location:=sangam proj_mode:=utm   # roslaunch arg passthrough
#
# 설정: config/simulation.ini (UDP IP/port, 카메라). 노드가 $PWD/config/simulation.ini 로 읽으므로
#       이 스크립트가 cwd 를 moraibridge/ 로 고정한다. control_mode_button(GUI)은 기본 off(headless).
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # = moraibridge/ (= catkin ws root)

# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash

if [[ ! -f "$HERE/devel/setup.bash" ]]; then
  echo "[run_bridge] $HERE/devel/setup.bash 가 없습니다. 먼저 catkin_make 하세요." >&2
  echo "             cd $HERE && catkin_make" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$HERE/devel/setup.bash"
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"

cd "$HERE"   # simulation.ini 가 $PWD/config/simulation.ini 로 읽히도록 (필수)

# simulation.ini [MoraiSimLocation] 첫 비주석 키를 roslaunch arg 로 패스 (CLI 가 같은 키 주면 CLI 우선)
INI="$HERE/config/simulation.ini"; INI_ARGS=()
declare -A CLI=(); for a in "$@"; do [[ "$a" == *:=* ]] && CLI["${a%%:=*}"]=1; done
while IFS='=' read -r k v; do
  k="${k// /}"; v="${v%%;*}"; v="$(echo -n "$v" | xargs)"
  [[ -n "$v" && -z "${CLI[$k]:-}" ]] && INI_ARGS+=("$k:=$v")
done < <(awk '/^\[MoraiSimLocation\]/{f=1;next} /^\[/{f=0} f && /^[[:space:]]*(location|ref_lat|ref_lon|proj_mode)[[:space:]]*=/' "$INI")

echo "[run_bridge] ws=$HERE  cwd=$HERE  ini_args=${INI_ARGS[*]:-<none>}"
echo "[run_bridge] roslaunch morai_bridge morai_bridge.launch ${INI_ARGS[*]} $*"
exec roslaunch morai_bridge morai_bridge.launch "${INI_ARGS[@]}" "$@"

#!/usr/bin/env bash
# Real camera -> 21-frame a2 -> rule gate + P5 brake (m/s², computed only, NOT sent).
#
# 현장에서는 이 세 개만 쓴다:
#   ./run.sh check              출발 전 점검 (카메라·NIC·GPU·커널·디스크)
#   ./run.sh live <이름>         실행. 이름 예: d20, approach10   → outputs/<날짜_시각>_<이름>/
#                               노출 600 µs 고정, 자차 50 km/h 기준 브레이크, 원본 bag 녹화,
#                               화면 녹화(a2_overlay.mp4), 화면 3프레임마다 갱신. q/Esc로 끝내면 브레이크·트래킹
#                               그래프(brake_replay/)까지 자동으로 만든다.
#   ./run.sh stop               비정상 종료 뒤 남은 프로세스 정리
#
# 그 밖의 모드:
#   ./run.sh verify             선배 파일 해시 + 단위시험 (카메라·GPU 불필요)
#   ./run.sh aim                모델 없이 조준·노출 확인
#   ./run.sh bench              처리량만 측정 (화면·영상·bag 없음, 60 s)
#   ./run.sh replay <bag_dir>   녹화한 원본으로 모델+브레이크 다시
#   ./run.sh brake-log <frames.jsonl>   기록된 A2로 브레이크·트래킹만 다시 계산 (CSV/PNG)
#   ./run.sh features           카메라 GenICam 기능 이름 (카메라 idle일 때)
#
# 바꿀 일이 있을 때만 쓰는 옵션:
#   --plan-kph KPH       P5 자차속도 (기본 50). actual = 실제 자차속도(정지면 P5 항상 0)
#   --exposure V         µs 또는 auto (기본 600 고정, 2026-10-08 야외)
#   --secs N             실행 시간 (live 기본 0 = q까지)
#   --no-bag             live에서 원본 녹화 끄기 (기본 켜짐, 약 1.1 GB/분)
#   --no-video           live/replay 화면 녹화(a2_overlay.mp4) 끄기 (기본 켜짐; 화면과 같은
#                        프레임을 별도 스레드로 저장, 3프레임마다 = 7 FPS 실시간 재생)
#   --display-every N    화면 갱신 간격 (live 기본 3)
#   --no-display         화면 없이 기록만
#   --obj-thr P          object/트래킹 threshold (기본 0.5)
#   --vrel-mode M        track (기본, 트래킹 앞차 속도) | min (min(A2, d_dot)) | a2 | ddot
#   --ego-speed MPS      실제 자차속도 상수 (기본 0)
#   --ego-source S       const (기본) | can (0x201 휠속, 미검증)
#   --can-iface IF       기본 can0
#   --brake-config FILE  기본 configs/brake_p5.yaml
#   --out DIR            출력 폴더 직접 지정
#   --guid --gain --fps --offset-y --roi-y0 --ego-vx --localhost-only   카메라/A2 고급 옵션
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPIC=/camera_driver_gv/vis/image_raw
INFO_TOPIC=/camera_driver_gv/vis/camera_info

MODE="${1:-}"; shift || true

GUID="${A1_CAMERA_IP:-192.168.1.51}"
EXPOSURE=""
EXPOSURE_MAX=2000
EXPOSURE_MIN=50
GAIN=""
GAIN_MAX=12
FPS=21
OFFSET_Y=240
OFFSET_X=320
ROI_Y0=235
ROI_HEIGHT=250
SECS=""
EGO_VX=0
EGO_WZ=0
OBJ_THR=0.5
DISPLAY_EVERY=""
PLAN_KPH=50
VREL_MODE=""
EGO_SPEED=0
EGO_SOURCE=const
CAN_IFACE=can0
BRAKE_CONFIG=""
OUT=""
WANT_BAG=""
WANT_VIDEO=""
WANT_DISPLAY=1
WANT_FULL_LOG=1
LOCALHOST_ONLY=0
BAG_DIR=""
FIRST_FRAME_TIMEOUT=15

die() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }
say() { printf '\033[1m[%s]\033[0m %s\n' "${MODE}" "$*"; }

while (($#)); do
  case "$1" in
    --guid) GUID="$2"; shift 2;;
    --exposure) EXPOSURE="$2"; shift 2;;
    --exposure-max) EXPOSURE_MAX="$2"; shift 2;;
    --exposure-min) EXPOSURE_MIN="$2"; shift 2;;
    --gain) GAIN="$2"; shift 2;;
    --gain-max) GAIN_MAX="$2"; shift 2;;
    --fps) FPS="$2"; shift 2;;
    --offset-y) OFFSET_Y="$2"; shift 2;;
    --offset-x) OFFSET_X="$2"; shift 2;;
    --roi-y0) ROI_Y0="$2"; shift 2;;
    --secs) SECS="$2"; shift 2;;
    --ego-vx) EGO_VX="$2"; shift 2;;
    --ego-wz) EGO_WZ="$2"; shift 2;;
    --obj-thr) OBJ_THR="$2"; shift 2;;
    --display-every) DISPLAY_EVERY="$2"; shift 2;;
    --plan-kph) PLAN_KPH="$2"; shift 2;;
    --vrel-mode) VREL_MODE="$2"; shift 2;;
    --ego-speed) EGO_SPEED="$2"; shift 2;;
    --ego-source) EGO_SOURCE="$2"; shift 2;;
    --can-iface) CAN_IFACE="$2"; shift 2;;
    --brake-config) BRAKE_CONFIG="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --bag) WANT_BAG=1; shift;;
    --no-bag) WANT_BAG=0; shift;;
    --video) WANT_VIDEO=1; shift;;
    --no-video) WANT_VIDEO=0; shift;;
    --no-display) WANT_DISPLAY=0; shift;;
    --no-full-log) WANT_FULL_LOG=0; shift;;
    --localhost-only) LOCALHOST_ONLY=1; shift;;
    -h|--help) usage; exit 0;;
    -*) die "unknown option: $1";;
    *) BAG_DIR="$1"; shift;;
  esac
done

[[ "${OBJ_THR}" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] || die "--obj-thr must be between 0 and 1"
[[ -z "${DISPLAY_EVERY}" || "${DISPLAY_EVERY}" =~ ^[1-9][0-9]*$ ]] || die "--display-every must be a positive integer"

BRAKE_CONFIG="${BRAKE_CONFIG:-${EXPERIMENT_DIR}/configs/brake_p5.yaml}"
[[ -f "${BRAKE_CONFIG}" ]] || die "brake config not found: ${BRAKE_CONFIG}"

ROI_Y1=$((ROI_Y0 + ROI_HEIGHT))
SEG_CKPT="${EXPERIMENT_DIR}/senior_code/data/ckpts/seg_pretrain_fx640_v3_backbone.pt"
A2_CKPT="${EXPERIMENT_DIR}/senior_code/data/ckpts/a2_mf20_v2s_n20.pt"

activate() {
  # ROS and virtualenv setup scripts both read unset variables, so `set -u`
  # has to stand down while they run.
  set +u
  # shellcheck source=/dev/null
  source "${EXPERIMENT_DIR}/scripts/activate_nuvo_env.sh"
  set -u
  if ((LOCALHOST_ONLY)); then
    # Keeps DDS discovery multicast off the GigE NIC, which shares no bandwidth
    # budget with the camera stream.  Not part of the verified 2026-09-14 setup,
    # so it stays opt-in.
    export ROS_LOCALHOST_ONLY=1
  fi
}

PIDS=()
STARTED_CAMERA=0
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    [[ -n "${pid}" ]] || continue
    kill -INT "${pid}" 2>/dev/null || true
  done
  for pid in "${PIDS[@]:-}"; do
    [[ -n "${pid}" ]] || continue
    for _ in {1..20}; do kill -0 "${pid}" 2>/dev/null || break; sleep 0.25; done
    kill -TERM "${pid}" 2>/dev/null || true
  done
  # `ros2 launch` can exit and leave the camera driver holding the camera, which
  # makes the next run fail with "already running" (seen 2026-10-07).
  if ((STARTED_CAMERA)); then
    pkill -INT -f "/camera_aravis2/camera_driver_gv" 2>/dev/null || true
    for _ in {1..20}; do pgrep -f "/camera_aravis2/camera_driver_gv" >/dev/null || break; sleep 0.25; done
    pkill -KILL -f "/camera_aravis2/camera_driver_gv" 2>/dev/null || true
  fi
}

start_camera() {
  local exposure_auto exposure_us gain_auto gain_db extra=()
  if [[ "${EXPOSURE}" == "auto" ]]; then
    exposure_auto=Continuous; exposure_us=0
    extra+=("exposure_min_us:=${EXPOSURE_MIN}" "exposure_max_us:=${EXPOSURE_MAX}")
  else
    exposure_auto=Off; exposure_us="${EXPOSURE}"
  fi
  if [[ "${GAIN}" == "auto" ]]; then
    gain_auto=Continuous; gain_db=0
    extra+=("gain_max_db:=${GAIN_MAX}")
  else
    gain_auto=Off; gain_db="${GAIN}"
  fi
  local white=Off
  if [[ "${EXPOSURE}" == "auto" ]]; then white=Continuous; fi

  # Match the driver's executable path, not the bare name: a terminal running
  # `tail -f .../camera_driver.log` would otherwise look like a live driver.
  if pgrep -f "/camera_aravis2/camera_driver_gv" >/dev/null; then
    die "camera_driver_gv is already running; ./run.sh stop"
  fi

  say "camera guid=${GUID} exposure=${EXPOSURE} gain=${GAIN} fps=${FPS} offset_y=${OFFSET_Y}"
  ros2 launch "${EXPERIMENT_DIR}/launch/hikrobot_a2_camera_1280x720.launch.py" \
    guid:="${GUID}" fps:="${FPS}" offset_x:="${OFFSET_X}" offset_y:="${OFFSET_Y}" \
    exposure_auto:="${exposure_auto}" exposure_us:="${exposure_us}" \
    gain_auto:="${gain_auto}" gain_db:="${gain_db}" \
    balance_white_auto:="${white}" "${extra[@]}" \
    >"${OUT}/camera_driver.log" 2>&1 &
  PIDS+=("$!")
  STARTED_CAMERA=1
  trap cleanup EXIT INT TERM

  # The topic name is NOT a readiness signal: camera_aravis2 advertises it
  # before it opens the camera, and the name lingers in discovery after the
  # node has died.  `ros2 launch` also stays alive after its node exits, so the
  # launch pid says nothing either.  Wait for an actual message, and meanwhile
  # watch the log so a failed camera reports in seconds instead of a minute.
  for _ in {1..12}; do
    if grep -q 'FATAL' "${OUT}/camera_driver.log" 2>/dev/null; then
      grep 'FATAL' "${OUT}/camera_driver.log" | head -3 >&2
      die "camera driver failed to open the camera (see ${OUT}/camera_driver.log)"
    fi
    if timeout 4 ros2 topic echo --once --qos-reliability best_effort \
         --field header.frame_id "${TOPIC}" >/dev/null 2>&1; then
      say "camera is streaming"
      return 0
    fi
  done
  tail -40 "${OUT}/camera_driver.log" >&2
  die "no image arrived on ${TOPIC}"
}

record_config() {
  local bag_json=false
  if ((${WANT_BAG:-0})); then bag_json=true; fi
  cat >"${OUT}/run_config.json" <<JSON
{
  "mode": "${MODE}",
  "started_at": "$(date -Iseconds)",
  "camera": {
    "guid": "${GUID}", "fps": ${FPS},
    "offset_x": ${OFFSET_X}, "offset_y": ${OFFSET_Y},
    "exposure": "${EXPOSURE}", "exposure_min_us": ${EXPOSURE_MIN}, "exposure_max_us": ${EXPOSURE_MAX},
    "gain": "${GAIN}", "gain_max_db": ${GAIN_MAX}
  },
  "model_band": {"roi_y0": ${ROI_Y0}, "roi_y1": ${ROI_Y1}},
  "ego_motion": {"vx_mps": ${EGO_VX}, "wz_radps": ${EGO_WZ}},
  "object_threshold": ${OBJ_THR},
  "brake": {
    "config": "${BRAKE_CONFIG}", "plan_kph": "${PLAN_KPH:-actual}",
    "vrel_mode": "${VREL_MODE:-config}", "ego_source": "${EGO_SOURCE}",
    "ego_speed_mps": ${EGO_SPEED}, "sent_to_vehicle": false
  },
  "raw_bag": ${bag_json},
  "duration_s": ${SECS:-0}
}
JSON
}

start_bag() {
  ((${WANT_BAG:-0})) || return 0
  say "recording raw BayerRG8 to ${OUT}/bag (about 1.2 GB per minute)"
  ros2 bag record \
    --qos-profile-overrides-path "${EXPERIMENT_DIR}/configs/bag_qos_override.yaml" \
    -o "${OUT}/bag" -b 4000000000 \
    "${TOPIC}" "${INFO_TOPIC}" >"${OUT}/bag_record.log" 2>&1 &
  PIDS+=("$!")
}

run_model() {
  local note="$1"; shift
  local flags=()
  if ((WANT_DISPLAY)); then flags+=(--display); fi
  if ((${WANT_VIDEO:-0})); then flags+=(--save-overlay-video); fi
  if ((WANT_FULL_LOG)); then flags+=(--log-full-a2); fi
  if [[ -n "${PLAN_KPH}" && "${PLAN_KPH}" != actual ]]; then flags+=(--plan-kph "${PLAN_KPH}"); fi
  if [[ -n "${VREL_MODE}" ]]; then flags+=(--vrel-mode "${VREL_MODE}"); fi
  if [[ "${EXPOSURE}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then flags+=(--fixed-exposure-us "${EXPOSURE}"); fi
  python "${EXPERIMENT_DIR}/scripts/live_a2_brake.py" \
    --input ros2 --ros-topic "${TOPIC}" \
    --width 1280 --height 720 --camera-fps "${FPS}" --target-fps 21 \
    --roi-y0 "${ROI_Y0}" --roi-y1 "${ROI_Y1}" \
    --ego-vx "${EGO_VX}" --ego-wz "${EGO_WZ}" \
    --obj-thr "${OBJ_THR}" \
    --brake-config "${BRAKE_CONFIG}" \
    --ego-speed-source "${EGO_SOURCE}" --ego-speed "${EGO_SPEED}" --can-iface "${CAN_IFACE}" \
    --display-every "${DISPLAY_EVERY:-1}" \
    --seg-ckpt "${SEG_CKPT}" --a2-ckpt "${A2_CKPT}" \
    --warmup-iters 10 --duration "${SECS}" \
    --first-frame-timeout "${FIRST_FRAME_TIMEOUT}" \
    --geometry-note "${note}" \
    --output-dir "${OUT}" "${flags[@]}" "$@"
}

case "${MODE}" in
  check)
    exec bash "${EXPERIMENT_DIR}/scripts/preflight.sh"
    ;;

  verify)
    python3 "${EXPERIMENT_DIR}/scripts/verify_portable_bundle.py"
    python3 -m pytest -q "${EXPERIMENT_DIR}/tests"
    ;;

  brake-log)
    [[ -n "${BAG_DIR}" && -f "${BAG_DIR}" ]] || die "usage: ./run.sh brake-log <frames.jsonl> [--plan-kph K] [--vrel-mode M] [--out DIR]"
    flags=(--config "${BRAKE_CONFIG}" --ego-speed "${EGO_SPEED}" --obj-thr "${OBJ_THR}")
    if [[ -n "${PLAN_KPH}" && "${PLAN_KPH}" != actual ]]; then flags+=(--plan-kph "${PLAN_KPH}"); fi
    if [[ -n "${VREL_MODE}" ]]; then flags+=(--vrel-mode "${VREL_MODE}"); fi
    if [[ -n "${OUT}" ]]; then flags+=(--out "${OUT}"); fi
    # System python3 has matplotlib for the PNG; the brake chain needs only numpy+yaml.
    exec /usr/bin/python3 "${EXPERIMENT_DIR}/scripts/replay_brake_log.py" "${BAG_DIR}" "${flags[@]}"
    ;;

  features)
    activate
    if pgrep -f "/camera_aravis2/camera_driver_gv" >/dev/null; then
      die "stop camera_driver_gv first; the camera allows one control channel"
    fi
    exec arv-tool-0.8 --name="${GUID}" features
    ;;

  stop)
    pkill -INT -f "/camera_aravis2/camera_driver_gv" || true
    pkill -INT -f "python[^ ]* [^ ]*(live_camera_a2_21fps|live_a2_brake)[.]py" || true
    pkill -INT -f 'ros2 bag record' || true
    pkill -INT -f 'hikrobot_a2_camera_1280x720[.]launch[.]py' || true
    sleep 2
    pkill -f "/camera_aravis2/camera_driver_gv" || true
    echo "stopped"
    ;;

  aim)
    activate
    : "${EXPOSURE:=600}" "${GAIN:=0}" "${SECS:=0}"
    OUT="${OUT:-${EXPERIMENT_DIR}/outputs/aim}"
    mkdir -p "${OUT}"
    start_camera
    say "green band = the 250x1280 the model actually sees. w/s to move, q to quit."
    python "${EXPERIMENT_DIR}/scripts/aim_camera.py" \
      --ros-topic "${TOPIC}" --roi-y0 "${ROI_Y0}" --roi-y1 "${ROI_Y1}" \
      --sensor-offset-y "${OFFSET_Y}" --output-dir "${OUT}"
    ;;

  live)
    NAME="${BAG_DIR:-run}"
    [[ "${NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || die "run name: letters, digits, . _ - only (got '${NAME}')"
    activate
    : "${EXPOSURE:=600}" "${GAIN:=0}" "${SECS:=0}"
    : "${WANT_BAG:=1}" "${WANT_VIDEO:=1}" "${DISPLAY_EVERY:=3}"
    OUT="${OUT:-${EXPERIMENT_DIR}/outputs/$(date +%Y%m%d_%H%M%S)_${NAME}}"
    mkdir -p "${OUT}"
    record_config
    start_camera
    start_bag
    say "output=${OUT}"
    say "exposure ${EXPOSURE} us fixed | brake at ego ${PLAN_KPH} km/h | q or Esc to stop"
    status=0
    run_model "real-car a2 + P5 brake run; 1280x720 centered hardware crop; FOV/calibration not validated" \
      || status=$?
    cleanup
    trap - EXIT INT TERM
    if [[ -f "${OUT}/frames.jsonl" ]]; then
      say "making brake/tracking graphs in ${OUT}/brake_replay"
      flags=(--config "${BRAKE_CONFIG}" --ego-speed "${EGO_SPEED}" --obj-thr "${OBJ_THR}")
      if [[ "${PLAN_KPH}" != actual ]]; then flags+=(--plan-kph "${PLAN_KPH}"); fi
      if [[ -n "${VREL_MODE}" ]]; then flags+=(--vrel-mode "${VREL_MODE}"); fi
      /usr/bin/python3 "${EXPERIMENT_DIR}/scripts/replay_brake_log.py" "${OUT}/frames.jsonl" \
        "${flags[@]}" >"${OUT}/brake_replay.log" 2>&1 \
        || say "brake graphs failed; see ${OUT}/brake_replay.log"
    fi
    if ((status == 2)); then
      say "done, but 21 FPS was NOT sustained (see ${OUT}/summary.json pass/likely_bottleneck)"
    elif ((status != 0)); then
      die "runner failed (exit ${status}); see output above and ${OUT}/camera_driver.log"
    fi
    say "results: ${OUT}/summary.json  ${OUT}/a2_overlay.mp4  ${OUT}/brake_replay/brake_replay.png  ${OUT}/brake_replay/tracks_bev.png"
    ;;

  bench)
    activate
    : "${EXPOSURE:=600}" "${GAIN:=0}" "${SECS:=60}"
    WANT_VIDEO=0; WANT_DISPLAY=0; WANT_FULL_LOG=0
    OUT="${OUT:-${EXPERIMENT_DIR}/outputs/bench_$(date +%Y%m%d_%H%M%S)}"
    mkdir -p "${OUT}"
    record_config
    start_camera
    say "throughput only: no display, no video, no bag, no full a2 log"
    run_model "throughput baseline; 1280x720 centered hardware crop; FOV/calibration not validated"
    ;;

  replay)
    activate
    [[ -n "${BAG_DIR}" ]] || die "usage: ./run.sh replay <bag_dir>"
    [[ -d "${BAG_DIR}" ]] || die "not a bag directory: ${BAG_DIR}"
    : "${SECS:=0}"
    WANT_BAG=0
    : "${WANT_VIDEO:=1}" "${DISPLAY_EVERY:=3}"
    OUT="${OUT:-${EXPERIMENT_DIR}/outputs/replay_$(date +%Y%m%d_%H%M%S)}"
    mkdir -p "${OUT}"
    record_config
    trap cleanup EXIT INT TERM
    # The model takes some seconds to load and the bag starts on a fixed delay,
    # so the two race.  Give the bag a head start that the runner outwaits.
    FIRST_FRAME_TIMEOUT=180
    say "playing ${BAG_DIR}"
    ros2 bag play "${BAG_DIR}" --delay 20 \
      --qos-profile-overrides-path "${EXPERIMENT_DIR}/configs/bag_qos_override.yaml" \
      >"${OUT}/bag_play.log" 2>&1 &
    PIDS+=("$!")
    run_model "offline replay of a recorded bag; capture_to_a2_ms_header is not a live latency"
    ;;

  ""|-h|--help|help)
    usage
    ;;

  *)
    die "unknown mode: ${MODE} (try ./run.sh help)"
    ;;
esac

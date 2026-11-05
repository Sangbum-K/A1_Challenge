#!/usr/bin/env bash
# Single entry point for the real-camera a2 experiment.
#
#   ./run.sh check              preflight: bundle, GPU, disk, NIC, camera, stale procs
#   ./run.sh features           dump the camera's GenICam feature names (camera must be idle)
#   ./run.sh aim                live band/exposure/focus check, no model, starts in seconds
#   ./run.sh live               the recognition run: camera + model + live overlay
#   ./run.sh bench              throughput-only run, nothing else competing for the CPU
#   ./run.sh replay <bag_dir>   re-run the model over a recorded bag, as many times as you like
#   ./run.sh stop               kill a driver/runner left behind by a crash
#
# Common options (all modes that start a camera):
#   --guid ID            camera serial or IPv4            (default 192.168.1.51)
#   --exposure V         microseconds, or auto            (live/aim: 2000, bench: 15000)
#                        in live/aim, [ and ] retune it on screen while running
#   --gain V             dB, or auto                      (default 0)
#   --fps HZ             acquisition rate                 (default 21)
#   --offset-y PX        sensor ROI vertical offset       (default 240)
#   --roi-y0 PX          model band top inside the frame  (default 235, height fixed at 250)
#   --secs N             run length; 0 = until q/Ctrl-C   (live: 0, bench: 60)
#   --ego-vx MPS         ego speed fed to a2; leave 0 for a stationary/slow cart
#   --obj-thr P          object display/lead threshold (default 0.5)
#   --display-every N     refresh live display every N model outputs (default 1)
#   --out DIR            output directory
#   --bag                also record the raw BayerRG8 stream (about 1.1 GB/min)
#   --no-video --no-display --localhost-only
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
DISPLAY_EVERY=1
OUT=""
WANT_BAG=0
WANT_VIDEO=1
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
    --out) OUT="$2"; shift 2;;
    --bag) WANT_BAG=1; shift;;
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
[[ "${DISPLAY_EVERY}" =~ ^[1-9][0-9]*$ ]] || die "--display-every must be a positive integer"

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
  if ((WANT_BAG)); then bag_json=true; fi
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
  "raw_bag": ${bag_json},
  "duration_s": ${SECS:-0}
}
JSON
}

start_bag() {
  ((WANT_BAG)) || return 0
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
  if ((WANT_VIDEO)); then flags+=(--save-overlay-video); fi
  if ((WANT_FULL_LOG)); then flags+=(--log-full-a2); fi
  python "${EXPERIMENT_DIR}/scripts/live_camera_a2_21fps.py" \
    --input ros2 --ros-topic "${TOPIC}" \
    --width 1280 --height 720 --camera-fps "${FPS}" --target-fps 21 \
    --roi-y0 "${ROI_Y0}" --roi-y1 "${ROI_Y1}" \
    --ego-vx "${EGO_VX}" --ego-wz "${EGO_WZ}" \
    --obj-thr "${OBJ_THR}" \
    --display-every "${DISPLAY_EVERY}" \
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

  features)
    activate
    if pgrep -f "/camera_aravis2/camera_driver_gv" >/dev/null; then
      die "stop camera_driver_gv first; the camera allows one control channel"
    fi
    exec arv-tool-0.8 --name="${GUID}" features
    ;;

  stop)
    pkill -INT -f "/camera_aravis2/camera_driver_gv" || true
    pkill -INT -f "python[^ ]* [^ ]*live_camera_a2_21fps[.]py" || true
    pkill -INT -f 'ros2 bag record' || true
    pkill -INT -f 'hikrobot_a2_camera_1280x720[.]launch[.]py' || true
    sleep 2
    pkill -f "/camera_aravis2/camera_driver_gv" || true
    echo "stopped"
    ;;

  aim)
    activate
    : "${EXPOSURE:=2000}" "${GAIN:=0}" "${SECS:=0}"
    OUT="${OUT:-${EXPERIMENT_DIR}/outputs/aim}"
    mkdir -p "${OUT}"
    start_camera
    say "green band = the 250x1280 the model actually sees. w/s to move, q to quit."
    python "${EXPERIMENT_DIR}/scripts/aim_camera.py" \
      --ros-topic "${TOPIC}" --roi-y0 "${ROI_Y0}" --roi-y1 "${ROI_Y1}" \
      --sensor-offset-y "${OFFSET_Y}" --output-dir "${OUT}"
    ;;

  live)
    activate
    : "${EXPOSURE:=2000}" "${GAIN:=0}" "${SECS:=0}"
    OUT="${OUT:-${EXPERIMENT_DIR}/outputs/live_$(date +%Y%m%d_%H%M%S)}"
    mkdir -p "${OUT}"
    record_config
    start_camera
    start_bag
    say "output=${OUT}"
    say "[ and ] retune exposure on screen; q or Esc stops and writes summary.json"
    run_model "outdoor recognition run; 1280x720 centered hardware crop; FOV/calibration not validated" \
      --tune-exposure
    ;;

  bench)
    activate
    : "${EXPOSURE:=15000}" "${GAIN:=0}" "${SECS:=60}"
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

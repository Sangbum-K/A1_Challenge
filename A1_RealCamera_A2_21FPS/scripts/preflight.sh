#!/usr/bin/env bash
# Everything that has silently broken a run before, checked in one place.
# Run this before driving off, not after.
set -uo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAMERA_IP="${A1_CAMERA_IP:-192.168.1.51}"
CAMERA_NIC="${A1_CAMERA_NIC:-enp5s0}"
MIN_FREE_GB="${A1_MIN_FREE_GB:-50}"
# The 2026-09-14 baseline (compute p95 15.2 ms) was measured on this kernel.
EXPECTED_KERNEL="${A1_EXPECTED_KERNEL:-6.8.0-138-generic}"

fail=0
warn=0
ok()   { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; fail=$((fail + 1)); }
note() { printf '  \033[33mWARN\033[0m  %s\n' "$*"; warn=$((warn + 1)); }
head_() { printf '\n\033[1m%s\033[0m\n' "$*"; }

head_ "1. bundle"
if python3 "${EXPERIMENT_DIR}/scripts/verify_portable_bundle.py" >/dev/null 2>&1; then
  ok "senior source and checkpoints are byte-identical"
else
  bad "verify_portable_bundle.py failed; senior files changed or missing"
fi

head_ "2. runtime"
if [[ -f /opt/ros/humble/setup.bash ]]; then ok "ROS2 Humble present"; else bad "ROS2 Humble missing"; fi
if [[ -x "${EXPERIMENT_DIR}/.venv/bin/python" ]]; then ok ".venv present"; else bad ".venv missing"; fi
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu="$(nvidia-smi --query-gpu=name,driver_version,temperature.gpu --format=csv,noheader 2>/dev/null | head -1)"
  if [[ -n "${gpu}" ]]; then
    ok "GPU ${gpu}"
    temp="$(printf '%s' "${gpu}" | awk -F', ' '{print $3}')"
    if [[ -n "${temp}" && "${temp}" -ge 80 ]]; then
      note "GPU already at ${temp} C before the run; outdoor ambient will add to this"
    fi
  else
    bad "nvidia-smi returned no GPU"
  fi
else
  bad "nvidia-smi not found"
fi

head_ "3. performance state"
kernel="$(uname -r)"
if [[ "${kernel}" == "${EXPECTED_KERNEL}" ]]; then
  ok "kernel ${kernel} matches the baseline"
elif [[ "${kernel}" == "6.8.1-1059-realtime" ]]; then
  note "kernel ${kernel} with the A1-BSW RT boot measured 15-16 FPS previously but
          passed a 20.97 FPS bench on 2026-09-27; run ./run.sh bench for today's throughput"
else
  note "kernel is ${kernel}, the 21 FPS baseline was measured on ${EXPECTED_KERNEL};
          re-run ./run.sh bench before trusting a timing result"
fi
gov="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
if [[ "${gov}" == "performance" ]]; then
  ok "CPU governor is performance"
elif [[ "${gov}" == "powersave" && "${kernel}" == "${EXPECTED_KERNEL}" ]]; then
  ok "CPU governor is powersave; this setting passed on the measured generic baseline"
elif [[ -n "${gov}" ]]; then
  note "CPU governor is '${gov}'; verify the 47.6 ms frame budget with ./run.sh bench"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  persist="$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null | head -1)"
  if [[ "${persist}" == "Enabled" ]]; then
    ok "GPU persistence mode enabled"
  else
    note "GPU persistence mode is ${persist:-unknown}; the card idles at its minimum clock
          between runs.  sudo nvidia-smi -pm 1"
  fi
fi
heavy="$(ps -eo comm --sort=-pcpu | sed -n '2,6p' | grep -E '^(code|firefox|chrome)' | tr '\n' ' ')"
if [[ -n "${heavy}" ]]; then
  note "desktop apps are among the top CPU consumers (${heavy}); close them before a timed run"
else
  ok "no desktop app dominating the CPU"
fi

head_ "4. storage"
free_gb="$(df -BG --output=avail "${EXPERIMENT_DIR}" 2>/dev/null | tail -1 | tr -dc '0-9')"
if [[ -n "${free_gb}" ]] && (( free_gb >= MIN_FREE_GB )); then
  ok "${free_gb} GB free (raw bag is about 1.2 GB per minute at 1280x720 21 FPS)"
else
  bad "only ${free_gb:-?} GB free; need at least ${MIN_FREE_GB} GB"
fi

head_ "5. camera link"
if ip link show "${CAMERA_NIC}" >/dev/null 2>&1; then
  state="$(ip -br link show "${CAMERA_NIC}" | awk '{print $2}')"
  addr="$(ip -br addr show "${CAMERA_NIC}" | awk '{print $3}')"
  mtu="$(cat "/sys/class/net/${CAMERA_NIC}/mtu" 2>/dev/null)"
  if [[ "${state}" == "UP" ]]; then ok "${CAMERA_NIC} is UP (mtu ${mtu})"; else bad "${CAMERA_NIC} is ${state}; check the cable and connector seating"; fi
  if [[ -n "${addr}" ]]; then ok "${CAMERA_NIC} address ${addr}"; else bad "${CAMERA_NIC} has no IPv4 address"; fi
  if command -v nmcli >/dev/null 2>&1; then
    method="$(nmcli -t -f ipv4.method con show "$(nmcli -t -f GENERAL.CONNECTION dev show "${CAMERA_NIC}" 2>/dev/null | cut -d: -f2)" 2>/dev/null | cut -d: -f2)"
    if [[ "${method}" == "manual" ]]; then
      ok "NetworkManager profile is static"
    elif [[ -n "${method}" ]]; then
      bad "NetworkManager profile is '${method}', not manual; this is how discovery broke before"
    fi
  fi
else
  bad "interface ${CAMERA_NIC} does not exist"
fi
# Two profiles in this machine both claim 192.168.1.100/24 on different NICs.
same_subnet="$(ip -br addr | awk '$3 ~ /^192\.168\.1\./ {print $1}' | tr '\n' ' ')"
count="$(printf '%s' "${same_subnet}" | wc -w)"
if (( count > 1 )); then
  bad "more than one interface is on 192.168.1.0/24 (${same_subnet}); unplug all but ${CAMERA_NIC}"
elif (( count == 1 )); then
  ok "exactly one interface on the camera subnet (${same_subnet})"
else
  bad "no interface is on 192.168.1.0/24; 'camlink' (enp5s0) and 'camlink10g' (enp6s0) both
          hold 192.168.1.100/24, so connect exactly one of them:  nmcli con up camlink"
fi

head_ "6. camera"
if ping -c 2 -W 1 "${CAMERA_IP}" >/dev/null 2>&1; then
  ok "camera ${CAMERA_IP} responds to ping"
else
  bad "camera ${CAMERA_IP} does not respond; run 'arv-tool-0.8' to see its current address"
fi
if command -v arv-tool-0.8 >/dev/null 2>&1; then
  found="$(timeout 15 arv-tool-0.8 2>/dev/null | head -5)"
  if [[ -n "${found}" ]]; then ok "aravis sees: ${found//$'\n'/ | }"; else bad "arv-tool-0.8 found no camera"; fi
else
  note "arv-tool-0.8 not installed; skipping GenICam discovery"
fi

head_ "7. stale processes"
if pgrep -f "/camera_aravis2/camera_driver_gv" >/dev/null; then
  bad "camera_driver_gv is already running; stop it or the camera stream stays locked"
else
  ok "no stale camera_driver_gv"
fi
if pgrep -f "python[^ ]* [^ ]*live_camera_a2_21fps[.]py" >/dev/null; then
  note "an old live_camera_a2_21fps is still running"
else
  ok "no stale runner"
fi

head_ "8. display"
if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then
  ok "DISPLAY=${DISPLAY:-${WAYLAND_DISPLAY}}"
else
  note "no DISPLAY; the runner will fall back to video-only and 'aim' will not work"
fi

printf '\n'
if (( fail > 0 )); then
  printf '\033[31m%d blocking problem(s), %d warning(s)\033[0m\n' "${fail}" "${warn}"
  exit 1
fi
printf '\033[32mready\033[0m (%d warning(s))\n' "${warn}"

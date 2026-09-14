#!/usr/bin/env bash
# ===========================================================================
# 단일차 closed-loop + moraibridge 통합 런처 (ini 연동)
# ---------------------------------------------------------------------------
# moraibridge(simulation.ini 기반 시뮬 I/O) 1세트 + closed_loop 제어기 1개를
# 한 번에 기동. multicar/run_multicar.sh 인프라를 "차 1대" 로 축약한 형태.
#
# 동작:
#   1) moraibridge/config/simulation.ini 의 [MoraiSimLocation] location/ref_lat/ref_lon 을
#      현재 TRACK(sangam|speedway) 에 맞춰 in-place 동기화 (--no-sync 로 끔).
#      → moraibridge 좌표 origin 과 closed_loop raceline(TRACK env) 이 일관되게 맞물림.
#   2) moraibridge: roslaunch morai_bridge morai_bridge.launch (단일차, ns 없음).
#      IniParser 가 cwd/config/simulation.ini 를 읽으므로, repo 의 moraibridge 디렉토리에서
#      띄워 moraibridge/config/simulation.ini 를 그대로 사용. (--no-bridge 로 끔 = closed_loop 만)
#   3) closed_loop: scripts/run_closed_loop_prof635.sh <track> <map> [...] 위임
#      (모델 env / F1 토글 / viz 옵션은 그 런처가 처리).
#   모든 프로세스 background. Ctrl-C(또는 종료) 시 trap 으로 일괄 kill.
#
# 사용:
#   bash scripts/run_closed_loop_sim.sh sangam mild
#   bash scripts/run_closed_loop_sim.sh speedway high --f1=656
#   bash scripts/run_closed_loop_sim.sh sangam mild --viz --f1=654
#   bash scripts/run_closed_loop_sim.sh sangam mild --no-bridge   # bridge 미기동(이미 떠있을 때)
#   bash scripts/run_closed_loop_sim.sh sangam mild --acc         # ACC 모드(run_closed_loop_acc.sh 경로)
#
# env (closed_loop 쪽으로 그대로 전달):
#   F1=654|655|656  MAIN_CKPT=<path>  SIDE_OFF=1  F2_DELTA_SCALE=...  MODEL_ACC=1 ...
#   SIM_INI=<path>  : 동기화 대상 ini override (기본 moraibridge/config/simulation.ini)
#   ROS_MASTER_URI  : roscore 위치
#
# 사전조건 (사용자 몫): roscore 기동 + moraibridge catkin build & source.
#   이 런처는 빌드하지 않음. --no-bridge 면 moraibridge 빌드 불필요(closed_loop 만).
# ===========================================================================
set -u

SCC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# positional: <track> [map]. 나머지(-…)는 closed_loop 런처로 전달.
TRACK="sangam"; MAP="mild"
if [[ -n "${1:-}" && "$1" != -* ]]; then TRACK="$1"; shift; fi
if [[ -n "${1:-}" && "$1" != -* ]]; then MAP="$1"; shift; fi
case "$TRACK" in sangam|speedway) ;; *) echo "track 는 sangam | speedway (받은 값: $TRACK)" >&2; exit 1 ;; esac
case "$MAP" in mild|high) ;; *) echo "map 은 mild | high (받은 값: $MAP)" >&2; exit 1 ;; esac

# 이 런처 전용 플래그(--no-bridge/--no-sync/--acc)는 소비, 나머지는 closed_loop 로 패스스루.
DO_BRIDGE="true"; DO_SYNC="true"; USE_ACC="false"
PASS_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --no-bridge) DO_BRIDGE="false" ;;
        --no-sync)   DO_SYNC="false" ;;
        --acc)       USE_ACC="true" ;;
        -h|--help)
            echo "usage: $0 <sangam|speedway> [mild|high] [--no-bridge] [--no-sync] [--acc] \\"
            echo "          [--f1=654|655|656] [--viz|--viz-seg|--viz-a2|--viz-all] [--rosmaster=URL]"
            echo "  moraibridge(simulation.ini) + closed_loop 통합. nested 옵션은 run_closed_loop_prof635.sh 참고."
            exit 0 ;;
        *) PASS_ARGS+=("$arg") ;;
    esac
done

# ---------------- ROS env ----------------
# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash 2>/dev/null || true
export PYTHONPATH="$SCC_ROOT:$SCC_ROOT/external:${PYTHONPATH:-}"

# ---------------- 1) simulation.ini location 동기화 ----------------
# (closed_loop 런처의 --sync-ini 와 동일 로직 — 여기선 통합 런처에서 직접 수행.)
SIM_INI="${SIM_INI:-$SCC_ROOT/moraibridge/config/simulation.ini}"
case "$TRACK" in
    sangam)   REF_LAT="37.58336692"; REF_LON="126.88224248" ;;
    speedway) REF_LAT="37.29695047"; REF_LON="127.21141031" ;;
esac
PY_SYNC="${PYTHON:-python3}"
if [[ "$DO_SYNC" == "true" ]]; then
    [[ -f "$SIM_INI" ]] || { echo "[err] simulation.ini 없음: $SIM_INI" >&2; exit 1; }
    "$PY_SYNC" - "$SIM_INI" "$TRACK" "$REF_LAT" "$REF_LON" <<'PYEOF'
import sys
path, loc, lat, lon = sys.argv[1:5]
repl = {"location": loc, "ref_lat": lat, "ref_lon": lon}
out = []; in_simloc = False
for line in open(path, encoding="utf-8"):
    raw = line.rstrip("\n"); s = raw.lstrip()
    if s.startswith("["):
        in_simloc = (s.split("]")[0].strip("[").strip() == "MoraiSimLocation")
        out.append(raw); continue
    if in_simloc and "=" in raw and not s.startswith(";"):
        k = raw.split("=", 1)[0]; key = k.strip()
        if key in repl:
            indent = k[: len(k) - len(k.lstrip())]
            out.append(f"{indent}{key}  = {repl[key]}"); continue
    out.append(raw)
open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")
print(f"[sim:sync-ini] {path} ← location={loc} ref_lat={lat} ref_lon={lon}")
PYEOF
fi

# ---------------- 종료 trap ----------------
PIDS=()
cleanup() {
    echo; echo "[sim] 종료 — 프로세스 kill (${#PIDS[@]}개)"
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]}"; do pkill -P "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
    echo "[sim] done."
}
trap cleanup INT TERM EXIT

echo "===================================================================="
echo "[sim] TRACK=$TRACK MAP=$MAP  bridge=$DO_BRIDGE sync=$DO_SYNC acc=$USE_ACC"
echo "[sim] sim_ini=$SIM_INI"
echo "[sim] closed_loop args: ${PASS_ARGS[*]:-<none>}"
echo "===================================================================="

# ---------------- 2) moraibridge ----------------
if [[ "$DO_BRIDGE" == "true" ]]; then
    # moraibridge 가 (구 morai_bridge 보다) 우선 resolve 되도록.
    if [[ -f "$SCC_ROOT/moraibridge/devel/setup.bash" ]]; then
        # shellcheck disable=SC1091
        source "$SCC_ROOT/moraibridge/devel/setup.bash"
    else
        echo "[sim][warn] moraibridge/devel 없음(미빌드). 노드 실행엔 catkin build 필요(이 런처는 빌드 안 함)."
    fi
    export ROS_PACKAGE_PATH="$SCC_ROOT/moraibridge/src:${ROS_PACKAGE_PATH:-}"
    _RES="$(rospack find morai_bridge 2>/dev/null || true)"
    echo "[sim] morai_bridge resolves to: ${_RES:-<none>}"
    [[ "$_RES" == "$SCC_ROOT/moraibridge/src/morai_bridge" ]] \
        || echo "[sim][warn] morai_bridge 가 moraibridge/src 가 아닌 곳으로 resolve 됨."
    mkdir -p "$SCC_ROOT/multicar/run/logs"
    _BRIDGE_LOG="$SCC_ROOT/multicar/run/logs/single_bridge.log"
    # IniParser 가 cwd/config/simulation.ini 를 읽으므로 moraibridge 디렉토리에서 띄움.
    (
        cd "$SCC_ROOT/moraibridge" || exit 1
        export PWD="$SCC_ROOT/moraibridge"
        exec roslaunch morai_bridge morai_bridge.launch \
            location:="$TRACK" ref_lat:="$REF_LAT" ref_lon:="$REF_LON"
    ) > "$_BRIDGE_LOG" 2>&1 &
    PIDS+=($!)
    echo "[sim] moraibridge PID=$! log=$_BRIDGE_LOG"
else
    echo "[sim] --no-bridge → moraibridge 미기동 (closed_loop 만)."
fi

# ---------------- 3) closed_loop 제어기 ----------------
# ACC 모드면 run_closed_loop_acc.sh(= prof635 + MODEL_ACC=1), 아니면 prof635 런처.
CL_SCRIPT="$SCC_ROOT/scripts/run_closed_loop_prof635.sh"
[[ "$USE_ACC" == "true" ]] && CL_SCRIPT="$SCC_ROOT/scripts/run_closed_loop_acc.sh"
# ini 는 위에서 이미 동기화했으므로 nested 런처엔 --sync-ini 를 넘기지 않음.
bash "$CL_SCRIPT" "$TRACK" "$MAP" "${PASS_ARGS[@]}" &
PIDS+=($!)
echo "[sim] closed_loop PID=$! (script=$(basename "$CL_SCRIPT"))"

echo "[sim] Ctrl-C 로 일괄 종료."
wait

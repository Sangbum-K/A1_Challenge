#!/usr/bin/env bash
# prof635 (v5: #654 frozen main + f_2 residual gated side) closed-loop 런처.
#
# 사용:
#   bash scripts/run_closed_loop_prof635.sh sangam mild        # 상암 안정맵
#   bash scripts/run_closed_loop_prof635.sh speedway high      # 스피드웨이 공격맵
#   bash scripts/run_closed_loop_prof635.sh sangam high --viz  # debug viz
#   bash scripts/run_closed_loop_prof635.sh sangam mild --rosmaster=http://localhost:11311
#   bash scripts/run_closed_loop_prof635.sh sangam mild --f1=656   # f_1 main #656 로 override (vs #654)
#   bash scripts/run_closed_loop_prof635.sh sangam mild --sync-ini # moraibridge simulation.ini location 을 track 에 동기화
#   (map 생략 시 mild. raceline = data/control/{track}_{map}_{P}_{M}.csv : dt0p1s + pit polygon)
#
# MODEL_TYPE=prof635 고정. i2_mem 은 seg=fx640_v3 + ego_mot=0 (학습 cache 일치) 로 live 계산.
#
# ── F1 main(f_1) ckpt 토글 (654 vs 656 비교) ────────────────────────────────
#   --f1=654|655|656  또는  F1=654|655|656 env  →  MAIN_CKPT=data/ckpts/main_prof65N.pth.
#   closed_loop_node.py 가 side ckpt 로드 後 이 main 으로 load_main_frozen override (line 362).
#   654/655/656 은 동일 SelfAttention(Professor) main 아키텍처 → load_main_frozen 호환(strict=False).
#   MAIN_CKPT 를 직접 export 하면 --f1 보다 우선(이 스크립트는 MAIN_CKPT 가 이미 있으면 안 건드림).
#   [확인 2026-05-27] data/ckpts/main_prof654.pth(현행 배포·frozen), main_prof655.pth(steer val 0.0231,
#   채택 보류), main_prof656.pth(동계열, digest 성능수치 미기재) 모두 실존.
#
# ── moraibridge simulation.ini 연동 (--sync-ini) ────────────────────────────
#   moraibridge 는 cwd/config/simulation.ini 의 [MoraiSimLocation] location/ref_lat/ref_lon 으로
#   좌표 origin 을 잡고, 이 런처(closed_loop)는 TRACK env 로 raceline 을 고른다. 둘은 별도 채널이라
#   track 을 바꾸면 양쪽을 수동으로 맞춰야 한다. --sync-ini 는 moraibridge/config/simulation.ini 의
#   location/ref_lat/ref_lon 을 현재 TRACK(sangam|speedway)에 맞춰 in-place 치환해 일관성을 보장한다.
#   (이 런처는 moraibridge 를 직접 띄우지 않음 — 사용자가 별도 roslaunch. ini 만 정합 맞춤.
#    moraibridge + closed_loop 를 한 번에 묶어 띄우려면 multicar/run_multicar.sh 사용.)
#   SIM_INI env 로 ini 경로 override 가능 (기본 moraibridge/config/simulation.ini).
set -e

# 머신 무관: 스크립트 위치 기준 repo root.
SCC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# python 선택: PYTHON env > 현재 활성 conda 환경($CONDA_PREFIX) > ml_con(이 머신 fallback) > python3.
# 다른 머신은 'conda activate <torch env>' 후 실행하면 그 환경으로 동작.
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
    if [[ -n "$CONDA_PREFIX" && -x "$CONDA_PREFIX/bin/python" ]] && "$CONDA_PREFIX/bin/python" -c "import torch" 2>/dev/null; then
        PY="$CONDA_PREFIX/bin/python"                       # 활성 conda env 우선
    else
        for p in "$HOME"/miniforge3/envs/ml_con/bin/python "$HOME"/miniconda3/envs/ml_con/bin/python "$HOME"/anaconda3/envs/ml_con/bin/python; do
            [[ -x "$p" ]] && "$p" -c "import torch" 2>/dev/null && { PY="$p"; break; }
        done
        [[ -z "$PY" ]] && PY=python3
    fi
fi
"$PY" -c "import torch" 2>/dev/null || { echo "[err] '$PY' 에 torch 없음. 'conda activate <torch env>' 후 재실행하거나 PYTHON=<python> 지정." >&2; exit 1; }

TRACK="sangam"; MAP="mild"
# positional: <track> [map].  flags(-…) 는 건드리지 않고 앞 1~2 개만 소비.
if [[ -n "$1" && "$1" != -* ]]; then TRACK="$1"; shift; fi
if [[ -n "$1" && "$1" != -* ]]; then MAP="$1"; shift; fi
case "$TRACK" in sangam|speedway) ;; *) echo "track 는 sangam | speedway (받은 값: $TRACK)" >&2; exit 1 ;; esac
case "$MAP" in mild|high) ;; *) echo "map 은 mild | high (받은 값: $MAP)" >&2; exit 1 ;; esac

DEBUG_VIZ="false"; DEBUG_VIZ_SEG="false"; DEBUG_VIZ_A2="false"; ROS_MASTER=""
F1_SEL="${F1:-}"           # --f1=654|655|656 또는 F1 env. 빈 값이면 closed_loop 기본(654 frozen).
SYNC_INI="false"
for arg in "$@"; do
    case "$arg" in
        --viz)      DEBUG_VIZ="true" ;;
        --viz-seg)  DEBUG_VIZ="true"; DEBUG_VIZ_SEG="true" ;;
        --viz-a2)   DEBUG_VIZ="true"; DEBUG_VIZ_A2="true" ;;
        --viz-all)  DEBUG_VIZ="true"; DEBUG_VIZ_SEG="true"; DEBUG_VIZ_A2="true" ;;
        --rosmaster=*) ROS_MASTER="${arg#--rosmaster=}" ;;
        --f1=*)     F1_SEL="${arg#--f1=}" ;;
        --sync-ini) SYNC_INI="true" ;;
        -h|--help) echo "usage: $0 <sangam|speedway> [mild|high] [--viz|--viz-seg|--viz-a2|--viz-all] [--rosmaster=URL] [--f1=654|655|656] [--sync-ini]
  env: SIDE_OFF=1(f_2 끔). gate = 모델 내부 learned NN (rule/평활 옵션 없음)
       F1=654|655|656 (--f1 동일). MAIN_CKPT 직접 export 시 그게 우선.
       SIM_INI=<path> (--sync-ini 대상 ini, 기본 moraibridge/config/simulation.ini)"; exit 0 ;;
        *) echo "unknown arg: $arg" >&2; exit 1 ;;
    esac
done

# ── F1 654/656 토글 → MAIN_CKPT (이미 MAIN_CKPT 가 export 돼 있으면 그걸 존중) ──
if [[ -n "$F1_SEL" && -z "${MAIN_CKPT:-}" ]]; then
    case "$F1_SEL" in
        654|655|656) ;;
        *) echo "[err] --f1/F1 은 654|655|656 (받은 값: $F1_SEL)" >&2; exit 1 ;;
    esac
    _F1_CKPT="$SCC_ROOT/data/ckpts/main_prof${F1_SEL}.pth"
    [[ -f "$_F1_CKPT" ]] || { echo "[err] f_1 ckpt 없음: $_F1_CKPT" >&2; exit 1; }
    export MAIN_CKPT="$_F1_CKPT"
fi

# ── moraibridge simulation.ini location 동기화 (--sync-ini) ──
if [[ "$SYNC_INI" == "true" ]]; then
    SIM_INI="${SIM_INI:-$SCC_ROOT/moraibridge/config/simulation.ini}"
    [[ -f "$SIM_INI" ]] || { echo "[err] sync 대상 ini 없음: $SIM_INI" >&2; exit 1; }
    case "$TRACK" in
        sangam)   _LAT="37.58336692"; _LON="126.88224248" ;;
        speedway) _LAT="37.29695047"; _LON="127.21141031" ;;
    esac
    "$PY" - "$SIM_INI" "$TRACK" "$_LAT" "$_LON" <<'PYEOF'
import sys, re
path, loc, lat, lon = sys.argv[1:5]
# [MoraiSimLocation] 섹션 안의 location/ref_lat/ref_lon active line 만 치환(주석 line 보존).
repl = {"location": loc, "ref_lat": lat, "ref_lon": lon}
out = []
in_simloc = False
for line in open(path, encoding="utf-8"):
    raw = line.rstrip("\n")
    s = raw.lstrip()
    if s.startswith("["):
        in_simloc = (s.split("]")[0].strip("[").strip() == "MoraiSimLocation")
        out.append(raw); continue
    if in_simloc and "=" in raw and not s.startswith(";"):
        k = raw.split("=", 1)[0]; key = k.strip()
        if key in repl:
            indent = k[: len(k) - len(k.lstrip())]
            out.append(f"{indent}{key}  = {repl[key]}")
            continue
    out.append(raw)
open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")
print(f"[sync-ini] {path} ← location={loc} ref_lat={lat} ref_lon={lon}")
PYEOF
fi

# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash 2>/dev/null || true
export PYTHONPATH="$SCC_ROOT:$SCC_ROOT/external:$PYTHONPATH"
export MODEL_TYPE="prof635"
export TRACK="$TRACK"
export MAP="$MAP"
[[ -n "$ROS_MASTER" ]] && export ROS_MASTER_URI="$ROS_MASTER"

echo "[closed_loop:prof635] python = $PY"
echo "[closed_loop:prof635] TRACK=$TRACK MAP=$MAP MODEL_TYPE=prof635 viz=$DEBUG_VIZ seg=$DEBUG_VIZ_SEG a2=$DEBUG_VIZ_A2"
echo "[closed_loop:prof635] gate=learned-NN(gate_head, rule 제거) SIDE_OFF=${SIDE_OFF:-0}"
if [[ -n "${MAIN_CKPT:-}" ]]; then
    echo "[closed_loop:prof635] f_1 main OVERRIDE = $(basename "$MAIN_CKPT") (side ckpt 의 654 main 을 덮어씀)"
else
    echo "[closed_loop:prof635] f_1 main = 기본(main_prof654.pth frozen, override 없음)"
fi
[[ "$SYNC_INI" == "true" ]] && echo "[closed_loop:prof635] moraibridge ini synced → $SIM_INI (location=$TRACK)"
[[ "${MODEL_ACC:-0}" == "1" ]] && echo "[closed_loop:prof635] ACC=ON (5f a_2 predicted vL_rel 주입, f_2=predvrel; run_closed_loop_acc.sh)"
echo "[closed_loop:prof635] raceline = $(ls data/control/${TRACK}_${MAP}_*.csv 2>/dev/null | head -1)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "(no GPU)"

cd "$SCC_ROOT"
exec "$PY" -u scripts/closed_loop_node.py \
    _debug_viz:="$DEBUG_VIZ" _debug_viz_seg:="$DEBUG_VIZ_SEG" _debug_viz_a2:="$DEBUG_VIZ_A2"

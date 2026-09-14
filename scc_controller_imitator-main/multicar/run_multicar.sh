#!/usr/bin/env bash
# ===========================================================================
# SCC multi-car closed-loop 런처
# ---------------------------------------------------------------------------
# 한 번 실행 → multicar/scc_multicar.ini 의 각 [CarN] 마다
#   (moraibridge 1세트 + closed_loop 제어기 1개) 를 namespace·포트 충돌 없이 N대 동시 기동.
#
# 동작:
#   각 [CarN] 에 대해
#     1) multicar/run/CarN/config/simulation.ini 생성
#        (= moraibridge/config/simulation.ini 템플릿 + 차별 포트/IP/location 치환)
#     2) moraibridge: cwd=multicar/run/CarN 에서
#          roslaunch morai_bridge morai_bridge.launch car_ns:=CarN
#        → IniParser 가 cwd/config/simulation.ini (= 위 차별 ini) 를 읽어 차별 포트로 송수신.
#        → group ns=CarN 으로 토픽이 /CarN/app/..., /CarN/raw_img_app/... 로 묶임.
#     3) closed_loop: CAR_NS=CarN + <prefix→모델 env> python scripts/closed_loop_node.py
#        → 토픽이 /CarN/app/... 로 prefix 되어 (2) 와 정확히 맞물림.
#   모든 프로세스는 background. PID 기록. Ctrl-C(또는 종료) 시 trap 으로 일괄 kill.
#
# 사용:
#   bash multicar/run_multicar.sh                         # 기본 ini
#   bash multicar/run_multicar.sh multicar/scc_multicar.ini
#   MASTER_INI=... bash multicar/run_multicar.sh
#
# 사전조건 (사용자 몫):
#   - moraibridge catkin workspace 빌드 + source (morai_bridge 패키지가 rospack 에서 보여야 함).
#     이 런처는 빌드를 하지 않음. ROS_PACKAGE_PATH 에 moraibridge devel/src 가 있어야 함.
#   - roscore (또는 ROS_MASTER_URI) 가 떠 있어야 함.
#   - MORAI 시뮬이 각 차의 포트/IP 로 송수신하도록 시나리오 구성되어 있어야 함.
# ===========================================================================
set -u

SCC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MASTER_INI="${MASTER_INI:-${1:-$SCC_ROOT/multicar/scc_multicar.ini}}"
TEMPLATE_INI="$SCC_ROOT/moraibridge/config/simulation.ini"
RUN_DIR="$SCC_ROOT/multicar/run"
LOG_DIR="$RUN_DIR/logs"

# scc_multicar.ini 는 git untracked (테스터별 로컬 설정 — pull 시 초기화 방지). 없으면 .example 에서 복사.
if [[ ! -f "$MASTER_INI" ]]; then
    if [[ -f "$MASTER_INI.example" ]]; then
        cp "$MASTER_INI.example" "$MASTER_INI"
        echo "[multicar] $MASTER_INI 없음 → .example 복사. 이 파일은 git untracked 이니 자유롭게 수정하세요."
    else
        echo "[err] master ini 없음: $MASTER_INI (.example 도 없음)" >&2; exit 1
    fi
fi
[[ -f "$TEMPLATE_INI" ]] || { echo "[err] template ini 없음: $TEMPLATE_INI" >&2; exit 1; }
mkdir -p "$LOG_DIR"

# ---------------- python 선택 (closed_loop 용, torch 필요) ----------------
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]] && "$CONDA_PREFIX/bin/python" -c "import torch" 2>/dev/null; then
        PY="$CONDA_PREFIX/bin/python"
    else
        for p in "$HOME"/miniforge3/envs/ml_con/bin/python "$HOME"/miniconda3/envs/ml_con/bin/python "$HOME"/anaconda3/envs/ml_con/bin/python; do
            [[ -x "$p" ]] && "$p" -c "import torch" 2>/dev/null && { PY="$p"; break; }
        done
        [[ -z "$PY" ]] && PY=python3
    fi
fi

# ---------------- ROS env ----------------
# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash 2>/dev/null || true
# ★ 이 repo 에는 'morai_bridge' 라는 동일 이름 패키지가 둘 있음:
#     (A) moraibridge/src/morai_bridge      ← 본 작업 대상 (namespace/cwd-ini 지원, 신규 launch)
#     (B) morai_bridge/src/bsw/.../morai_bridge ← 구버전 (object_driver/multi_ego 포함, 이미 빌드됨)
#   roslaunch 가 (A) 를 잡도록 moraibridge devel 을 우선 source. 빌드 안 됐으면 source space 라도
#   ROS_PACKAGE_PATH 맨 앞에 둬서 launch/패키지 resolve 가 (A) 로 가게 함 (node 실행은 빌드 필요).
if [[ -f "$SCC_ROOT/moraibridge/devel/setup.bash" ]]; then
    source "$SCC_ROOT/moraibridge/devel/setup.bash"
    echo "[multicar] sourced moraibridge/devel/setup.bash"
else
    echo "[multicar][warn] moraibridge/devel 없음(미빌드). source space 를 ROS_PACKAGE_PATH 앞에 추가."
    echo "[multicar][warn] 노드 실행에는 catkin build 가 필요함 (이 런처는 빌드 안 함)."
fi
# moraibridge 를 항상 최우선 (구 morai_bridge 보다 앞) 으로.
export ROS_PACKAGE_PATH="$SCC_ROOT/moraibridge/src:${ROS_PACKAGE_PATH:-}"
export PYTHONPATH="$SCC_ROOT:$SCC_ROOT/external:${PYTHONPATH:-}"
# resolve 확인: roslaunch 가 잡을 morai_bridge 가 (A) 인지.
_RESOLVED="$(rospack find morai_bridge 2>/dev/null || true)"
echo "[multicar] morai_bridge resolves to: ${_RESOLVED:-<none>}"
if [[ "$_RESOLVED" != "$SCC_ROOT/moraibridge/src/morai_bridge" ]]; then
    echo "[multicar][warn] morai_bridge 가 moraibridge/src 가 아닌 곳으로 resolve 됨 → namespace launch 미적용 위험."
fi

# ---------------- 종료 trap (일괄 kill) ----------------
PIDS=()
cleanup() {
    echo
    echo "[multicar] 종료 — 모든 프로세스 kill (${#PIDS[@]}개)"
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    # roslaunch 자식까지 정리 (process group)
    for pid in "${PIDS[@]}"; do
        pkill -P "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "[multicar] done."
}
trap cleanup INT TERM EXIT

# ---------------- 모델 prefix → env 매핑 ----------------
# ckpt 는 data/ckpts/ 의 실제 파일 기준. ★ 사용자 확정 필요 항목은 주석에 표기.
# 공통: TRACK/MAP 은 MULTICAR_TRACK/MULTICAR_MAP env 로 override (기본 sangam/mild).
emit_model_env() {
    # $1 = prefix.  echo 로 "KEY=VAL KEY=VAL ..." 한 줄 반환.
    local prefix="$1"
    local common="TRACK=${MULTICAR_TRACK:-sangam} MAP=${MULTICAR_MAP:-mild}"
    case "$prefix" in
        qualifier)
            # 예선 = iy v25k-B4 (Professor main + Side SCC 4L, distbrake 계열).
            # MODEL_VARIANT=B → MODEL_CKPT_IY = iy_v25k_B4_gttc_accgate_best_loss.pt (closed_loop_node.py 내부 결정).
            echo "$common MODEL_TYPE=iy MODEL_VARIANT=B"
            ;;
        acc_gate)
            # prof635 + ACC(5f vrel 주입) + hardneg 5f a_2.
            # ── closed_loop_node.py 가 실제로 읽는 env 만 emit (검증 2026-05-26):
            #   · GATE_A2_CKPT 는 closed_loop_node.py 에 없는 죽은 env → 제거함.
            #     hardneg 5f a_2 를 쓰려면 ACC vrel 산출용 5f a_2 (ACC_A2_MF5F_CKPT) 로 주입해야 함.
            #   · MODEL_ACC=1 + PROF635_CKPT 미지정 → f_2 ckpt 자동 = ACC_F2_CKPT
            #     = data/ckpts/prof635_side_predvrel_ttc.pt (실존 확인). DELTA_MODE=brake_only 와 정합.
            #   · gate/i2_mem 경로 a_2 = MODEL_A2_V6_CKPT 기본 = a2_v7b_d128_alpha075 (override 안 함, 학습 정합).
            #   · GATE_THRESH: closed_loop 기본 0.3 은 obj_amax(=a_2 raw logit, no-lead≈-0.46/lead≈+1.41) 스케일.
            #     -2.2 는 거의 상시 gate=1 → 미지정 시 기본 0.3 사용.
            echo "$common MODEL_TYPE=prof635 MODEL_ACC=1 \
ACC_A2_MF5F_CKPT=$SCC_ROOT/data/ckpts/a2_gate_hardneg_5f.pt \
GATE_THRESH=${GATE_THRESH:-0.3} DELTA_MODE=brake_only F2_DELTA_SCALE=${F2_DELTA_SCALE:-1.3}"
            ;;
        baseline)
            # prof635 기본 (distbrake, gate=EMA). f_2 = prof635_side_distbrake_v1.pt (closed_loop 기본).
            echo "$common MODEL_TYPE=prof635"
            ;;
        scc_p[1-5]_f654)
            # SCC profile (s_min 거리 프로파일 p1=공격 ~ p5=보수) + main #654.
            #   side ckpt = scc_p{N}_side.pt (publish_test_weights.sh 게시; main654 frozen + f_2 + gate_head(obj+vrel 2sigmoid) 전체).
            #   gate = closed_loop prof635 내부 learned gate_head (SCC 학습값 그대로). main=654 내장 → override 불필요.
            #   F2_DELTA_SCALE=1.0: gate(2sigmoid)가 강도 조절하므로 증폭 안 함.
            local pp="${prefix#scc_}"; pp="${pp%_f654}"   # scc_p3_f654 → p3
            echo "$common MODEL_TYPE=prof635 PROF635_CKPT=$SCC_ROOT/data/ckpts/scc_${pp}_side.pt \
DELTA_MODE=brake_only F2_DELTA_SCALE=${F2_DELTA_SCALE:-1.0}"
            ;;
        scc_p[1-5]_f656)
            # 위와 동일 + main #656 override (side ckpt 의 654 main 을 656 으로 덮어씀).
            local pp="${prefix#scc_}"; pp="${pp%_f656}"
            echo "$common MODEL_TYPE=prof635 PROF635_CKPT=$SCC_ROOT/data/ckpts/scc_${pp}_side.pt \
MAIN_CKPT=$SCC_ROOT/data/ckpts/main_prof656.pth DELTA_MODE=brake_only F2_DELTA_SCALE=${F2_DELTA_SCALE:-1.0}"
            ;;
        scc_p3_db_f654)
            # [db] a_2 재학습(mf20+DriverB) + 그 i2_mem 으로 재학습한 p3 side + main #654.
            #   A2_MF20_CKPT override 로 closed_loop a_2 를 db 로 교체(기존 prefix 무영향). side=scc_p3_side_db.
            echo "$common MODEL_TYPE=prof635 PROF635_CKPT=$SCC_ROOT/data/ckpts/scc_p3_side_db.pt \
A2_MF20_CKPT=$SCC_ROOT/data/ckpts/a2_mf20_v2s_n20_db.pt DELTA_MODE=brake_only F2_DELTA_SCALE=${F2_DELTA_SCALE:-1.0}"
            ;;
        scc_p3_db_f656)
            # 위 + main #656 override.
            echo "$common MODEL_TYPE=prof635 PROF635_CKPT=$SCC_ROOT/data/ckpts/scc_p3_side_db.pt \
MAIN_CKPT=$SCC_ROOT/data/ckpts/main_prof656.pth \
A2_MF20_CKPT=$SCC_ROOT/data/ckpts/a2_mf20_v2s_n20_db.pt DELTA_MODE=brake_only F2_DELTA_SCALE=${F2_DELTA_SCALE:-1.0}"
            ;;
        scc_p3_db_ng_f654)
            # [db] gate 없는 p3 (no_gate raw delta, ep09 best) + db a_2 + main #654. final=base+delta.
            echo "$common MODEL_TYPE=prof635 PROF635_CKPT=$SCC_ROOT/data/ckpts/scc_p3_ng_db.pt \
NO_GATE=1 DELTA_MODE=raw A2_MF20_CKPT=$SCC_ROOT/data/ckpts/a2_mf20_v2s_n20_db.pt F2_DELTA_SCALE=${F2_DELTA_SCALE:-1.0}"
            ;;
        scc_p3_db_ng_f656)
            echo "$common MODEL_TYPE=prof635 PROF635_CKPT=$SCC_ROOT/data/ckpts/scc_p3_ng_db.pt \
MAIN_CKPT=$SCC_ROOT/data/ckpts/main_prof656.pth NO_GATE=1 DELTA_MODE=raw \
A2_MF20_CKPT=$SCC_ROOT/data/ckpts/a2_mf20_v2s_n20_db.pt F2_DELTA_SCALE=${F2_DELTA_SCALE:-1.0}"
            ;;
        scc_p3_db_ng_gate_f654)
            # 시계열 beat gate (TemporalMLP, db). final=base+g·delta. recall↑(94.5) FP중(1.5).
            echo "$common MODEL_TYPE=prof635 PROF635_CKPT=$SCC_ROOT/data/ckpts/scc_p3_ng_db.pt \
NO_GATE=1 DELTA_MODE=raw GATE_CKPT=$SCC_ROOT/data/ckpts/scc_gate_mlp_beat_db.pt \
A2_MF20_CKPT=$SCC_ROOT/data/ckpts/a2_mf20_v2s_n20_db.pt F2_DELTA_SCALE=${F2_DELTA_SCALE:-1.0}"
            ;;
        scc_p3_db_ng_gate_f656)
            echo "$common MODEL_TYPE=prof635 PROF635_CKPT=$SCC_ROOT/data/ckpts/scc_p3_ng_db.pt \
MAIN_CKPT=$SCC_ROOT/data/ckpts/main_prof656.pth NO_GATE=1 DELTA_MODE=raw \
GATE_CKPT=$SCC_ROOT/data/ckpts/scc_gate_mlp_beat_db.pt \
A2_MF20_CKPT=$SCC_ROOT/data/ckpts/a2_mf20_v2s_n20_db.pt F2_DELTA_SCALE=${F2_DELTA_SCALE:-1.0}"
            ;;
        scc_p3_db_ng_distill_f654)
            # 시계열 distill gate (EMA 모방, db). FP 최저(0.84). recall EMA수준(92.5). no-lead 억제 강.
            echo "$common MODEL_TYPE=prof635 PROF635_CKPT=$SCC_ROOT/data/ckpts/scc_p3_ng_db.pt \
NO_GATE=1 DELTA_MODE=raw GATE_CKPT=$SCC_ROOT/data/ckpts/scc_gate_mlp_distill_db.pt \
A2_MF20_CKPT=$SCC_ROOT/data/ckpts/a2_mf20_v2s_n20_db.pt F2_DELTA_SCALE=${F2_DELTA_SCALE:-1.0}"
            ;;
        scc_p3_db_ng_distill_f656)
            echo "$common MODEL_TYPE=prof635 PROF635_CKPT=$SCC_ROOT/data/ckpts/scc_p3_ng_db.pt \
MAIN_CKPT=$SCC_ROOT/data/ckpts/main_prof656.pth NO_GATE=1 DELTA_MODE=raw \
GATE_CKPT=$SCC_ROOT/data/ckpts/scc_gate_mlp_distill_db.pt \
A2_MF20_CKPT=$SCC_ROOT/data/ckpts/a2_mf20_v2s_n20_db.pt F2_DELTA_SCALE=${F2_DELTA_SCALE:-1.0}"
            ;;
        scc_p[1-5]_ng_gate_f654)
            # delta(no_gate raw) + 외부 시계열 beat gate + main #654. final=base+g·delta, g=TemporalMLP.
            #   scc_p{N}_ng_f654 와 동일하되 GATE_CKPT=scc_gate_mlp_beat.pt 추가 → closed_loop 가 매 frame
            #   in-lane obj logit deque(K=16) → TemporalMLP g 산출. A1E2EModel-scc forward(base+g·delta)와 동일 식.
            local pp="${prefix#scc_}"; pp="${pp%_ng_gate_f654}"   # scc_p3_ng_gate_f654 → p3
            echo "$common MODEL_TYPE=prof635 PROF635_CKPT=$SCC_ROOT/data/ckpts/scc_${pp}_ng.pt \
NO_GATE=1 DELTA_MODE=raw GATE_CKPT=$SCC_ROOT/data/ckpts/scc_gate_mlp_beat.pt F2_DELTA_SCALE=${F2_DELTA_SCALE:-1.0}"
            ;;
        scc_p[1-5]_ng_gate_f656)
            # 위 + main #656 override.
            local pp="${prefix#scc_}"; pp="${pp%_ng_gate_f656}"
            echo "$common MODEL_TYPE=prof635 PROF635_CKPT=$SCC_ROOT/data/ckpts/scc_${pp}_ng.pt \
MAIN_CKPT=$SCC_ROOT/data/ckpts/main_prof656.pth NO_GATE=1 DELTA_MODE=raw \
GATE_CKPT=$SCC_ROOT/data/ckpts/scc_gate_mlp_beat.pt F2_DELTA_SCALE=${F2_DELTA_SCALE:-1.0}"
            ;;
        scc_p[1-5]_ng_f654)
            # gate 없는 버전 (NO_GATE=1 + delta raw) + main #654. final=base+delta (gate no-lead 과개방→풀악셀 깎임 회피).
            #   ckpt = scc_p{N}_ng.pt (no_gate 학습, pedal_head=raw delta, main #654 내장).
            local pp="${prefix#scc_}"; pp="${pp%_ng_f654}"   # scc_p3_ng_f654 → p3
            echo "$common MODEL_TYPE=prof635 PROF635_CKPT=$SCC_ROOT/data/ckpts/scc_${pp}_ng.pt \
NO_GATE=1 DELTA_MODE=raw F2_DELTA_SCALE=${F2_DELTA_SCALE:-1.0}"
            ;;
        scc_p[1-5]_ng_f656)
            # 위 + main #656 override.
            local pp="${prefix#scc_}"; pp="${pp%_ng_f656}"
            echo "$common MODEL_TYPE=prof635 PROF635_CKPT=$SCC_ROOT/data/ckpts/scc_${pp}_ng.pt \
MAIN_CKPT=$SCC_ROOT/data/ckpts/main_prof656.pth NO_GATE=1 DELTA_MODE=raw F2_DELTA_SCALE=${F2_DELTA_SCALE:-1.0}"
            ;;
        cmain)
            # f_1-only (main_no_image). steer/waypoint tracking 검증용, image/a_2 미사용.
            echo "$common MODEL_TYPE=iy MODEL_VARIANT=C"
            ;;
        *)
            echo "[err] 알 수 없는 prefix='$prefix' (qualifier|acc_gate|baseline|cmain|scc_p{1-5}_f654|scc_p{1-5}_f656|scc_p{1-5}_ng_f65{4,6}|scc_p{1-5}_ng_gate_f65{4,6} 중 하나)" >&2
            return 1
            ;;
    esac
}

# ---------------- ini 파서 (awk) ----------------
# get_val <SECTION> <KEY> : master ini 에서 값 추출 (inline ';' 주석 제거, 공백 trim).
get_val() {
    awk -v sec="$1" -v key="$2" '
        /^[ \t]*\[/ { cur=$0; gsub(/[][ \t]/,"",cur); insec=(cur==sec); next }
        insec {
            line=$0
            sub(/;.*/,"",line)                       # inline 주석 제거
            n=index(line,"=")
            if(n>0){
                k=substr(line,1,n-1); v=substr(line,n+1)
                gsub(/^[ \t]+|[ \t]+$/,"",k); gsub(/^[ \t]+|[ \t]+$/,"",v)
                if(k==key){print v; exit}
            }
        }
    ' "$MASTER_INI"
}

# list_cars : master ini 의 Car* 섹션 이름들 (등장 순서대로).
list_cars() {
    awk '/^[ \t]*\[Car[0-9]+\]/ { s=$0; gsub(/[][ \t]/,"",s); print s }' "$MASTER_INI"
}

# ---------------- 차별 simulation.ini 생성 ----------------
# template 의 해당 키 line 을 차별 값으로 치환 (구조/주석 보존). python 로 안전 치환.
make_car_ini() {
    local car_dir="$1" vstate="$2" cam="$3" ctl="$4" obj="$5"
    local src_ip="$6" dst_ip="$7" loc="$8" proj="$9" rlat="${10}" rlon="${11}"
    mkdir -p "$car_dir/config"
    "$PY" - "$TEMPLATE_INI" "$car_dir/config/simulation.ini" \
        "$vstate" "$cam" "$ctl" "$obj" "$src_ip" "$dst_ip" "$loc" "$proj" "$rlat" "$rlon" <<'PYEOF'
import sys, re
src, dst, vstate, cam, ctl, obj, src_ip, dst_ip, loc, proj, rlat, rlon = sys.argv[1:13]
# key -> new value (해당 section 의 key line value 만 교체, inline 주석은 보존).
repl = {
    "udp_source_ip":                        src_ip,
    "udp_destination_ip":                   dst_ip,
    "udp_vehicle_info_source_port":         vstate,
    "udp_vehicle_info_destination_port":    vstate,
    "udp_ego_ctl_source_port":              ctl,
    "udp_ego_ctl_destination_port":         ctl,
    "udp_object_source_port":               obj,
    "udp_object_destination_port":          obj,
    "udp_camera_driver_source_ports":       cam,   # SCC = camera_1 단일 → 단일 포트로 덮어씀
    "udp_camera_driver_destination_ports":  cam,
    "camera_topic_name_list":               "app/perc/camera_1",
    "camera_topic_frame_id_list":           "camera_1",
    "num_of_camera":                        "1",
    # [MoraiSimLocation] — group ns 하 vehicle_driver 의 /vehicle_driver/ref_* rosparam lookup 이
    # 빗나가므로 ini fallback 으로 채움.
    "location":                             loc,
    "proj_mode":                            proj,
    "ref_lat":                              rlat,
    "ref_lon":                              rlon,
}
out = []
for line in open(src, encoding="utf-8"):
    raw = line.rstrip("\n")
    stripped = raw.lstrip()
    if "=" in raw and not stripped.startswith(";"):
        k = raw.split("=", 1)[0]
        key = k.strip()
        if key in repl:
            # inline 주석 보존
            after = raw.split("=", 1)[1]
            m = re.search(r";", after)
            comment = (" " + after[m.start():].strip()) if m else ""
            indent = k[: len(k) - len(k.lstrip())]
            out.append(f"{indent}{key} = {repl[key]}{('   ' + comment.strip()) if comment else ''}")
            continue
    out.append(raw)
open(dst, "w", encoding="utf-8").write("\n".join(out) + "\n")
print(f"[make_car_ini] {dst}")
PYEOF
}

# ---------------- 트랙/라인 선택 ----------------
# track=sangam(상암)|speedway(용인), map=mild(안정)|high(공격).
#   closed_loop 의 TRACK/MAP env → raceline_dt0p1s.load_map → {track}_{map}_*.csv.
C_TRACK="$(get_val Common track)";         C_TRACK="${C_TRACK:-sangam}"
C_MAP="$(get_val Common map)";             C_MAP="${C_MAP:-mild}"
# track → MORAI sim 좌표 자동 매핑. ini 에 location/ref 를 직접 쓰면 그게 우선(하위호환),
#   안 쓰면 이 표의 값을 사용 → track 한 줄만으로 sim 좌표까지 결정됨.
case "$C_TRACK" in
    sangam)   T_LOC=sangam;   T_RLAT=37.58336692; T_RLON=126.88224248 ;;  # 상암
    speedway) T_LOC=speedway; T_RLAT=37.29695047; T_RLON=127.21141031 ;;  # 용인
    *)        T_LOC="$C_TRACK"; T_RLAT=""; T_RLON="";
              echo "[warn] track=$C_TRACK 자동 좌표 매핑 없음 — ini [Common] 에 location/ref_lat/ref_lon 직접 지정 필요" >&2 ;;
esac

# ---------------- 공통 값 (location/ref: ini 명시 우선, 없으면 track 자동) ----------------
C_SRC_IP="$(get_val Common udp_src_ip)";   C_SRC_IP="${C_SRC_IP:-10.0.0.10}"
C_DST_IP="$(get_val Common udp_dest_ip)";  C_DST_IP="${C_DST_IP:-10.0.0.20}"
C_LOC="$(get_val Common location)";        C_LOC="${C_LOC:-$T_LOC}"
C_PROJ="$(get_val Common proj_mode)";      C_PROJ="${C_PROJ:-utm}"
C_RLAT="$(get_val Common ref_lat)";        C_RLAT="${C_RLAT:-$T_RLAT}"
C_RLON="$(get_val Common ref_lon)";        C_RLON="${C_RLON:-$T_RLON}"

CARS=$(list_cars)
[[ -n "$CARS" ]] || { echo "[err] master ini 에 [CarN] 섹션이 없음: $MASTER_INI" >&2; exit 1; }

echo "===================================================================="
echo "[multicar] master ini : $MASTER_INI"
echo "[multicar] template    : $TEMPLATE_INI"
echo "[multicar] python      : $PY"
echo "[multicar] cars        : $(echo "$CARS" | tr '\n' ' ')"
echo "[multicar] track/map   : $C_TRACK / $C_MAP  → loc=$C_LOC ref=($C_RLAT, $C_RLON)"
echo "[multicar] common ip   : src=$C_SRC_IP dst=$C_DST_IP  proj=$C_PROJ"
echo "===================================================================="

# ---------------- 각 Car 기동 ----------------
for CAR in $CARS; do
    PREFIX="$(get_val "$CAR" prefix)"
    NS="$(get_val "$CAR" namespace)";        NS="${NS:-$CAR}"
    VSTATE="$(get_val "$CAR" vehicle_state_port)"
    CAM="$(get_val "$CAR" camera_port)"
    CTL="$(get_val "$CAR" control_port)"
    OBJ="$(get_val "$CAR" object_port)"
    # per-car dest_ip override 가능, 없으면 공통.
    DST_IP="$(get_val "$CAR" udp_dest_ip)";   DST_IP="${DST_IP:-$C_DST_IP}"
    SRC_IP="$(get_val "$CAR" udp_src_ip)";    SRC_IP="${SRC_IP:-$C_SRC_IP}"
    # raceline track/map: [CarN] 에 있으면 per-car, 없으면 [Common] 기본.
    TRACK_C="$(get_val "$CAR" track)";        TRACK_C="${TRACK_C:-$C_TRACK}"
    MAP_C="$(get_val "$CAR" map)";            MAP_C="${MAP_C:-$C_MAP}"

    if [[ -z "$PREFIX" || -z "$VSTATE" || -z "$CAM" || -z "$CTL" ]]; then
        echo "[err] $CAR 필수 키 누락 (prefix/vehicle_state_port/camera_port/control_port) — skip" >&2
        continue
    fi
    OBJ="${OBJ:-7076}"

    MODEL_ENV="$(MULTICAR_TRACK="$TRACK_C" MULTICAR_MAP="$MAP_C" emit_model_env "$PREFIX")" || { echo "[err] $CAR prefix 매핑 실패 — skip" >&2; continue; }

    CAR_DIR="$RUN_DIR/$NS"
    echo "--------------------------------------------------------------------"
    echo "[$CAR] ns=$NS prefix=$PREFIX  track=$TRACK_C map=$MAP_C  vstate=$VSTATE cam=$CAM ctl=$CTL obj=$OBJ  dst_ip=$DST_IP"
    echo "[$CAR] model env: $MODEL_ENV"

    # 1) 차별 simulation.ini
    make_car_ini "$CAR_DIR" "$VSTATE" "$CAM" "$CTL" "$OBJ" "$SRC_IP" "$DST_IP" \
                 "$C_LOC" "$C_PROJ" "$C_RLAT" "$C_RLON"

    # 2) moraibridge (cwd = CAR_DIR 이어야 IniParser 가 CAR_DIR/config/simulation.ini 를 읽음).
    #    getenv("PWD") 의존 → subshell 에서 cd 후 PWD export.
    (
        cd "$CAR_DIR" || exit 1
        export PWD="$CAR_DIR"
        exec roslaunch morai_bridge morai_bridge.launch \
            car_ns:="$NS" location:="$C_LOC" proj_mode:="$C_PROJ" \
            ref_lat:="$C_RLAT" ref_lon:="$C_RLON"
    ) > "$LOG_DIR/${NS}_bridge.log" 2>&1 &
    PIDS+=($!)
    echo "[$CAR] moraibridge PID=$! log=$LOG_DIR/${NS}_bridge.log"

    # 3) closed_loop 제어기 (CAR_NS=NS + 모델 env).
    (
        cd "$SCC_ROOT" || exit 1
        export CAR_NS="$NS"
        # shellcheck disable=SC2086
        export $MODEL_ENV
        exec "$PY" -u scripts/closed_loop_node.py \
            _debug_viz:=false _debug_viz_seg:=false _debug_viz_a2:=false
    ) > "$LOG_DIR/${NS}_ctrl.log" 2>&1 &
    PIDS+=($!)
    echo "[$CAR] closed_loop PID=$! log=$LOG_DIR/${NS}_ctrl.log"
done

echo "===================================================================="
echo "[multicar] ${#PIDS[@]} 프로세스 기동 완료. 로그: $LOG_DIR/"
echo "[multicar] Ctrl-C 로 일괄 종료."
echo "===================================================================="

# 모든 백그라운드 프로세스 대기 (하나라도 죽으면 trap 으로 정리).
wait

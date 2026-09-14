#!/usr/bin/env bash
# data/control/raceline_test/ 의 superellipse(p) × ay 변형 raceline 으로 prof635 closed-loop 실행.
#
# 사용:
#   bash scripts/run_closed_loop_prof635_rltest.sh speedway_mild_p1.5_ay24
#   bash scripts/run_closed_loop_prof635_rltest.sh sangam_high_p2.0_ay20 --viz
#   GATE_SMOOTH=beat  bash scripts/run_closed_loop_prof635_rltest.sh speedway_mild_p1.5_ay24
#
# 동작: spec="<track>_<map>_p<dim>_ay<NN>" → raceline_test/<spec>_<P>-<M>.csv 매칭.
#       data/control/<track>_<map>_<P>_<M>.csv (load_map glob 호환: underscore) 로 임시
#       심볼릭링크. 기존 default 파일은 ".bak" 로 잠시 옮겼다가 run_closed_loop_prof635.sh
#       호출, EXIT trap 으로 원복.
set -e

SCC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPEC="${1:?usage: $0 <track>_<map>_p<dim>_ay<NN> [run_closed_loop_prof635.sh flags...]}"
shift
TRACK="${SPEC%%_*}"; rest="${SPEC#*_}"
MAP="${rest%%_*}"

shopt -s nullglob
candidates=( "$SCC_ROOT"/data/control/raceline_test/"${SPEC}"_*.csv )
defaults=( "$SCC_ROOT"/data/control/"${TRACK}_${MAP}"_[0-9]*.csv )
shopt -u nullglob
(( ${#candidates[@]} )) || { echo "[err] raceline 없음: data/control/raceline_test/${SPEC}_*.csv" >&2; exit 1; }
SRC="${candidates[0]}"
PM_DASH="$(basename "${SRC%.csv}" | sed -E "s/^${SPEC}_//")"     # 00000-00710
PM="${PM_DASH/-/_}"                                              # 00000_00710
STAGE="$SCC_ROOT/data/control/${TRACK}_${MAP}_${PM}.csv"

cleanup() {
    rm -f "$STAGE"
    for f in "${defaults[@]}"; do [[ -f "$f.bak" ]] && mv -f "$f.bak" "$f"; done
}
trap cleanup EXIT INT TERM
for f in "${defaults[@]}"; do mv -f "$f" "$f.bak"; done
ln -sf "$SRC" "$STAGE"

echo "[rltest] spec=$SPEC  src=$(realpath --relative-to="$SCC_ROOT" "$SRC")  stage=$(basename "$STAGE")"
bash "$SCC_ROOT/scripts/run_closed_loop_prof635.sh" "$TRACK" "$MAP" "$@"

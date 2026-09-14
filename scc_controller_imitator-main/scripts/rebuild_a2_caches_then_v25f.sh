#!/bin/bash
# 6 dataset 의 a_2 cache 일괄 재생성 (build_a2_cache.py proj fix 적용 후) → DriverB prebake 완료 대기 → v25f launch
#
# 결과 cache 는 *_fixed.npz 로 저장 (broken 원본 보존). launch_v25f.sh 가 _fixed 를 사용하도록 변경.

set -e
PY=/home/ailab-12/miniforge3/envs/ml_con/bin/python
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
ROOT=/home/ailab-12/git/scc_controller_imitator
LOG=/tmp/a2_rebuild.log

cd "$ROOT"

# DriverB prebake 완료 대기 (file + done 마커 둘다 체크)
DRIVERB_PREBAKE=$D/prebake/f1_prebake_iy_DriverB_new_logged_fixed_prev20.npz
echo "[wait] DriverB prebake 완료 대기..." | tee -a "$LOG"
while [ ! -f "$DRIVERB_PREBAKE" ] || ! grep -q "  done." /tmp/driverb_prebake.log 2>/dev/null; do
  sleep 30
done
echo "[ok] DriverB prebake 완료 (file + done 마커 확인)" | tee -a "$LOG"

# 5 cache 재생성 (DriverB 제외 — 별도)
DATASETS=(
  "sangam_lead_0520|i1_tok_cache_sangam_lead_0520_prev20.npz|f1_prebake_iy_sangam_lead_0520_v1_brakeaug_prev20.npz"
  "sangam_lead_0521|i1_tok_cache_sangam_lead_0521_prev20.npz|f1_prebake_iy_sangam_lead_0521_v1_brakeaug_prev20.npz"
  "sangam_pertO_v2|i1_tok_cache_sangam_pertO_v2_prev20.npz|f1_prebake_iy_sangam_pertO_v2_prev20.npz"
  "sangam_pertX_v2|i1_tok_cache_sangam_pertX_v2_prev20.npz|f1_prebake_iy_sangam_pertX_v2_prev20.npz"
  "spd_v2|i1_tok_cache_spd_v2_prev20.npz|f1_prebake_iy_spd_streaming_brakeaug_prev20.npz"
  "DriverB_new|i1_tok_cache_DriverB_new_prev20.npz|f1_prebake_iy_DriverB_new_logged_fixed_prev20.npz"
)

for ds in "${DATASETS[@]}"; do
  IFS='|' read -r name i1 pb <<< "$ds"
  OUT="$D/cache/a2_out_cache_${name}_prev20_fixed.npz"
  if [ -f "$OUT" ]; then
    echo "[skip] $OUT 이미 존재" | tee -a "$LOG"
    continue
  fi
  echo "[build] $name → $OUT" | tee -a "$LOG"
  $PY -u scripts/build_a2_cache.py \
    --i1-cache "$D/cache/$i1" \
    --prebake  "$D/prebake/$pb" \
    --out      "$OUT" \
    --batch 256 2>&1 | tail -15 | tee -a "$LOG"
done

echo "[done] 6 cache 재생성 완료" | tee -a "$LOG"

# launch_v25f.sh 가 _fixed cache 사용하도록 sed 수정 — 일회성. 임시 _v25f.sh 만들기.
sed 's|a2_out_cache_\([a-zA-Z0-9_]*\)_prev20\.npz|a2_out_cache_\1_prev20_fixed.npz|g' \
  "$ROOT/scripts/launch_v25f.sh" > "$ROOT/scripts/launch_v25f_fixed_a2.sh"
chmod +x "$ROOT/scripts/launch_v25f_fixed_a2.sh"
echo "[ready] launch_v25f_fixed_a2.sh 준비 — 가동" | tee -a "$LOG"

bash "$ROOT/scripts/launch_v25f_fixed_a2.sh" | tee -a "$LOG"

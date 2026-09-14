#!/bin/bash
# i_2 v7b cache 전 frame 재생성 (build_i2_cache zero-fill fix 적용 후).
#   원인: 이전 build_i2_cache 는 prebake(a2inlane) ∩ cache 만 처리 → sangam/spd 가
#         prebake unique=728~3.4k 뿐이라 96% all-zero placeholder 였음.
#   fix : prebake 는 d_yaw/prev LUT(optional) 로만 쓰고, cache 전 frame 처리.
#         prebake 미커버 frame 은 d_yaw=0 + prev=tick-20 fallback, fx=640.
#   기존 prebake-covered frame 값은 보존됨(검증: max|diff|=0.004, float16 반올림).
#   a_2 ckpt = a2_v7b (배포 ckpt 무관). 출력은 기존 v7b cache 를 덮어씀.
set -e
PY=/home/ailab-12/miniforge3/envs/ml_con/bin/python
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
ROOT=/home/ailab-12/git/scc_controller_imitator
CKPT=$ROOT/data/ckpts/a2_v7b_d128_alpha075_best_loss.pt
cd "$ROOT"

# name|i1_cache|prebake(없으면 NONE → fallback 전체)
DATASETS=(
  "sangam_lead_0520|i1_tok_cache_sangam_lead_0520_prev20.npz|f1_prebake_iy_sangam_lead_0520_a2inlane_prev20.npz"
  "sangam_lead_0521|i1_tok_cache_sangam_lead_0521_prev20.npz|f1_prebake_iy_sangam_lead_0521_a2inlane_prev20.npz"
  "sangam_pertO_v2|i1_tok_cache_sangam_pertO_v2_prev20.npz|f1_prebake_iy_sangam_pertO_v2_a2inlane_prev20.npz"
  "sangam_pertX_v2|i1_tok_cache_sangam_pertX_v2_prev20.npz|f1_prebake_iy_sangam_pertX_v2_a2inlane_prev20.npz"
  "DriverB_new|i1_tok_cache_DriverB_new_prev20.npz|f1_prebake_iy_DriverB_new_a2inlane_prev20.npz"
  "spd_v2|i1_tok_cache_spd_v2_prev20.npz|NONE"
)
for ds in "${DATASETS[@]}"; do
  IFS='|' read -r name i1 pb <<< "$ds"
  OUT="$D/cache/i2_mem_cache_${name}_v7b_prev20.npz"
  echo "=== [build] $name -> $(basename "$OUT") ==="
  PBARG=""
  [ "$pb" != "NONE" ] && PBARG="--prebake $D/prebake/$pb"
  $PY -u scripts/build_i2_cache.py \
    --i1-cache "$D/cache/$i1" \
    $PBARG \
    --out "$OUT" \
    --a2-ckpt "$CKPT" --fx 640
done
echo "=== ALL DONE ==="

#!/bin/bash
# a_2 multi-frame fusion sweep (2026-05-26) — vrel(상대속도) 강화 검증.
#   프레임수 × 시간 span(10-tick granularity, ≤5프레임, span≤50) sweep.
#   tick spacing = 5 (frame 당 5 tick). offset 은 tick 단위 (-10=2frame, -20=4frame ...).
#   per-frame i1 cache (prev20) 에서 timestamp(tick-off) 인덱싱 → 새 GPU cache 불필요.
#   yaw_shift OFF (무의미 확정). 275,999 매칭 (B1/B2 fix). warm = a2_v7b.
#   recipe: d-out128, focal-alpha0.75, scale-v20, scale-dist40, lr1e-4, ep12, val0.05.
#
#   조합 (name → prev-offsets, 1f=offset 없음):
#     mf1f_curr        : (none)        baseline, temporal 없음 (n_frames=1)
#     mf2f_10 / _20 / _30 / _50        span 효과
#     mf3f_10_20 / _20_40 / _10_30     3프레임
#     mf4f_10_20_30 / _10_20_40        4프레임
#     mf5f_10_20_30_40 / _10_20_30_40_50  5프레임
set -e
source /home/ailab-12/miniforge3/etc/profile.d/conda.sh
conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
CKPT_INIT=data/ckpts/a2_v7b_d128_alpha075_best_loss.pt
OUT=outputs_multi/a2
mkdir -p "$OUT"

# 공통 데이터 인자 (275,999 매칭; launch_a2_multiframe.sh 와 동일 순서 — eval val-split 재현용)
DATA_ARGS=(
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0520_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0520"
  --i1-cache "$D/cache/i1_tok_cache_sangam_lead_0521_prev20.npz" --aux-dir "$D/aux_targets_sangam_lead_0521"
  --i1-cache "$D/cache/i1_tok_cache_DriverB_new_prev20.npz"      --aux-dir "$D/aux_targets_DriverB_new"
  --i1-cache "$D/cache/i1_tok_cache_spd_v2_prev20.npz"           --aux-dir "$D/aux_targets_spd_nolead_0520"
  --i1-cache "$D/cache/i1_tok_cache_sangam_pertO_v2_prev20.npz"  --aux-dir "$D/aux_targets_sangam_fx640"
  --i1-cache "$D/cache/i1_tok_cache_sangam_pertX_v2_prev20.npz"  --aux-dir "$D/aux_targets_sangam_fx640"
  --run-filter "" --run-filter "" --run-filter "" --run-filter "" --run-filter "pertO" --run-filter "pertX"
)

RECIPE=(
  --resume-from "$CKPT_INIT" --d-out 128 --epochs 12 --batch-size 192 --lr 1e-4
  --weight-decay 1e-4 --workers 4 --val-ratio 0.05 --w-obj 1.0 --w-reg 1.0
  --use-focal --focal-alpha 0.75 --focal-gamma 2.0 --scale-v 20.0 --scale-dist 40.0
  --lds --fx 640 --yaw-shift off --seed 0 --out-dir "$OUT"
)

# name:offsets ("-" = 1f baseline, prev-offsets 없음)
COMBOS=(
  "mf_sweep_1f_curr:-"
  "mf_sweep_2f_10:10"
  "mf_sweep_2f_20:20"
  "mf_sweep_2f_30:30"
  "mf_sweep_2f_50:50"
  "mf_sweep_3f_10_20:10,20"
  "mf_sweep_3f_20_40:20,40"
  "mf_sweep_3f_10_30:10,30"
  "mf_sweep_4f_10_20_30:10,20,30"
  "mf_sweep_4f_10_20_40:10,20,40"
  "mf_sweep_5f_10_20_30_40:10,20,30,40"
  "mf_sweep_5f_10_20_30_40_50:10,20,30,40,50"
)

echo "=== MF SWEEP START $(date) === ${#COMBOS[@]} combos"
for entry in "${COMBOS[@]}"; do
  NAME="${entry%%:*}"; OFFS="${entry##*:}"
  if [ -f "$OUT/$NAME/best_loss.pt" ]; then
    echo "[skip] $NAME (best_loss.pt exists)"; continue
  fi
  echo "--- [$(date +%H:%M:%S)] TRAIN $NAME  offsets='$OFFS' ---"
  if [ "$OFFS" = "-" ]; then
    # 1f: prev-offsets 없음 → n_frames=1. _MultiFrameFusion 은 KV=[ego] 만 (temporal 없음).
    python -u scripts/train_a2_multiframe.py "${DATA_ARGS[@]}" "${RECIPE[@]}" \
      --prev-offsets "" --n-frames 1 --name "$NAME"
  else
    python -u scripts/train_a2_multiframe.py "${DATA_ARGS[@]}" "${RECIPE[@]}" \
      --prev-offsets "$OFFS" --name "$NAME"
  fi
done
echo "=== MF SWEEP TRAIN DONE $(date) ==="

# === EVAL: 동일 held-out (val-split seed=0 5%, lead-rich) 에서 전 조합 비교 ===
echo "=== EVAL $(date) ==="
MF_ARGS=()
for entry in "${COMBOS[@]}"; do
  NAME="${entry%%:*}"; OFFS="${entry##*:}"
  CK="$OUT/$NAME/best_loss.pt"
  [ -f "$CK" ] || { echo "[eval-skip] $NAME missing ckpt"; continue; }
  if [ "$OFFS" = "-" ]; then
    MF_ARGS+=(--mf "$NAME" "$CK" 1 off "")
  else
    NF=$(( $(echo "$OFFS" | tr ',' '\n' | wc -l) + 1 ))
    MF_ARGS+=(--mf "$NAME" "$CK" "$NF" off "$OFFS")
  fi
done

python -u scripts/eval_a2_multiframe.py \
  "${DATA_ARGS[@]}" \
  --v7b data/ckpts/a2_v7b_d128_alpha075_best_loss.pt \
  --val-split --val-ratio 0.05 --seed 0 \
  "${MF_ARGS[@]}" \
  --out "$OUT/mf_sweep_eval.json"
echo "=== MF SWEEP ALL DONE $(date) ==="

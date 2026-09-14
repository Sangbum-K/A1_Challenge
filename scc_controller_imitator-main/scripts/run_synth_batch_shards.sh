#!/usr/bin/env bash
# 16-shard 병렬 synth_batch launcher.
# 사용:
#   bash scripts/run_synth_batch_shards.sh <frame_list> <raw_base> <out_dir> [N_SHARDS=16]

set -euo pipefail

FRAME_LIST="${1:?frame_list 필수}"
RAW_BASE="${2:?raw_base 필수}"
OUT_DIR="${3:?out_dir 필수}"
N="${4:-16}"

# v3 (2026-05-21): mpc/ fork (state ax constraint + jerk cap 5/10) — ax 폭주 fix
SYNTH=${SYNTH_OVERRIDE:-/home/ailab-12/git/scc_controller_imitator/mpc/devel/.private/scc_batch_synth/lib/scc_batch_synth/synth_batch}
SYNTH_CWD=${SYNTH_CWD:-/home/ailab-12/git/scc_controller_imitator/mpc}
echo "  synth bin: $SYNTH"
echo "  cwd:       $SYNTH_CWD"

# SCC MPC 합성 파라미터 — profile 별 제동 거동을 GT 에 심기 위해 env 로 전달.
#   TH = time headway (s),  SM = standstill gap s_min (m),  LH = lane_half (m, optional)
# 미지정 시 synth_batch default (t-headway 1.0 / s-min 3.0 / lane-half 2.0) 사용 → 기존 동작 보존.
TH="${TH:-}"; SM="${SM:-}"; LH="${LH:-}"
EXTRA_ARGS=()
[[ -n "$TH" ]] && EXTRA_ARGS+=(--t-headway "$TH")
[[ -n "$SM" ]] && EXTRA_ARGS+=(--s-min "$SM")
[[ -n "$LH" ]] && EXTRA_ARGS+=(--lane-half "$LH")
echo "  t-headway: ${TH:-<default 1.0>}   s-min: ${SM:-<default 3.0>}   lane-half: ${LH:-<default 2.0>}"

mkdir -p "$OUT_DIR"
LOG_DIR="$OUT_DIR/.shards_log"
mkdir -p "$LOG_DIR"

echo "[$(date +%H:%M:%S)] launching $N shards"
echo "  frame_list: $FRAME_LIST  ($(wc -l < "$FRAME_LIST") lines)"
echo "  raw_base:   $RAW_BASE"
echo "  out_dir:    $OUT_DIR"

PIDS=()
cd "$SYNTH_CWD"    # synth_batch 가 cwd 의존 (config/control.ini + resources/pedal_map 경로)
for ((i=0; i<N; i++)); do
  $SYNTH --frame-list "$FRAME_LIST" --raw-base "$RAW_BASE" --out "$OUT_DIR" \
         --shard "$i/$N" --warmup "${WARMUP:-20}" "${EXTRA_ARGS[@]}" > "$LOG_DIR/shard_$i.log" 2>&1 &
  PIDS+=($!)
done

echo "  shard pids: ${PIDS[*]}"

# wait all + report
fail=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "  shard pid=$pid FAILED"
    fail=$((fail+1))
  fi
done

echo "[$(date +%H:%M:%S)] all shards done. failed=$fail / $N"
# aggregate stats
echo "=== shard summaries ==="
for ((i=0; i<N; i++)); do
  tail -2 "$LOG_DIR/shard_$i.log" | grep "shard.*done" | sed "s/^/  [$i] /"
done

n_out=$(ls "$OUT_DIR" | grep -c "\.scc\.txt$" || true)
echo "total scc files: $n_out"

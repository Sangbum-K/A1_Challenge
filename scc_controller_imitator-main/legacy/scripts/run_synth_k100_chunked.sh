#!/usr/bin/env bash
# K=100 random trajectory 합성. chunk subprocess 분할 (segfault 방어 + incremental save).
# 사용: bash run_synth_k100_chunked.sh [chunk_size] [K] [start] [end]
set -u
CHUNK="${1:-5}"
K="${2:-100}"
START_FRAME="${3:-0}"
END_FRAME="${4:-6877}"
OUT_DIR="/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_k100_v1"
mkdir -p "$OUT_DIR"
SCRIPT="/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/scripts/synthesize_scc_kn_random.py"

# Override OUT_DIR for the script via env (script's OUT_DIR is module-level)
export SCC_OUT_DIR_OVERRIDE="$OUT_DIR"

for ((start=START_FRAME; start<END_FRAME; start+=CHUNK)); do
  end=$((start+CHUNK))
  [ "$end" -gt "$END_FRAME" ] && end="$END_FRAME"
  out_name=$(printf "scc_synth_k100_chunk_%05d_%05d.pkl" "$start" "$end")
  out_file="$OUT_DIR/$out_name"
  if [ -f "$out_file" ]; then
    echo "[skip] $out_name"
    continue
  fi
  echo "[chunk] start=$start end=$end K=$K"
  /usr/bin/python3 "$SCRIPT" --k "$K" --start "$start" --end "$end" \
    --out "$out_name" --out-dir-override "$OUT_DIR" 2>&1 | \
    grep -vE "^cc |^rm |^make|GAUSS|If you|Note:|OR the|^$|WARNING|MPC not solved|MPC get init|Vehicle state:|Cost:|^[0-9]+ Fail:|output_time|status: NaN|status: QP|steering_angle:|^-L|SQP_RTI" | \
    tail -10
done
echo "[done]"
ls "$OUT_DIR" | wc -l

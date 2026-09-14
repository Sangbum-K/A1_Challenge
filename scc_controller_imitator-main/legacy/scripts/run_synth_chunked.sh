#!/usr/bin/env bash
# acados solver 가 수백 frame 후 segfault → chunk 단위 subprocess. 각 chunk 마다 신선한 process.
# 사용: bash run_synth_chunked.sh [chunk_size]
set -u
CHUNK="${1:-500}"
TOTAL=6877
OUT_DIR="/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_v1"
mkdir -p "$OUT_DIR"
SCRIPT="/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/scripts/synthesize_scc_k1.py"

for ((start=0; start<TOTAL; start+=CHUNK)); do
  end=$((start+CHUNK))
  suffix=$(printf "_chunk_%05d_%05d" "$start" "$end")
  out_file="$OUT_DIR/scc_synth_k1_v1${suffix}.pkl"
  if [ -f "$out_file" ]; then
    echo "[skip] $out_file exists"
    continue
  fi
  echo "[chunk] $start..$end"
  /usr/bin/python3 "$SCRIPT" --start "$start" --end "$end" --out-suffix "$suffix" 2>&1 | \
    grep -vE "^cc |^rm |^make|GAUSS|If you|Note:|OR the|^$|WARNING|MPC not solved|MPC get init|Vehicle state:|Cost:|^[0-9]+ Fail:|output_time|status: NaN|status: QP|steering_angle:|^-L|SQP_RTI" | \
    tail -30
done

echo "[done] all chunks processed"
ls -la "$OUT_DIR"/scc_synth_k1_v1_chunk_*.pkl | head -20

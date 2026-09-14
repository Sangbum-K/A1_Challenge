#!/usr/bin/env bash
# K=N random trajectory 합성. chunk subprocess 분할.
set -u
CHUNK="${1:-500}"
K="${2:-5}"
TOTAL=6877
OUT_DIR="/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_kn_v1"
mkdir -p "$OUT_DIR"
SCRIPT="/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/scripts/synthesize_scc_kn_random.py"

for ((start=0; start<TOTAL; start+=CHUNK)); do
  end=$((start+CHUNK))
  out_name=$(printf "scc_synth_kn_v1_chunk_%05d_%05d.pkl" "$start" "$end")
  out_file="$OUT_DIR/$out_name"
  if [ -f "$out_file" ]; then
    echo "[skip] $out_name"
    continue
  fi
  echo "[chunk] start=$start end=$end K=$K"
  /usr/bin/python3 "$SCRIPT" --k "$K" --start "$start" --end "$end" --out "$out_name" 2>&1 | \
    grep -vE "^cc |^rm |^make|GAUSS|If you|Note:|OR the|^$|WARNING|MPC not solved|MPC get init|Vehicle state:|Cost:|^[0-9]+ Fail:|output_time|status: NaN|status: QP|steering_angle:|^-L|SQP_RTI" | \
    tail -15
done
echo "[done]"
ls "$OUT_DIR"/scc_synth_kn_v1_chunk_*.pkl | wc -l

#!/usr/bin/env bash
# fwd50 K=100 sequential CHUNK=2. correct transform (decision 0023).
set -u
K=100
CHUNK=2
SEED=${1:-1000}
OUT_DIR="/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_k100_v1"
SCRIPT="/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/scripts/synthesize_scc_kn_random.py"
IDX_FILE="$OUT_DIR/fwd50_frame_indices.txt"

LINES=($(cat "$IDX_FILE"))
N=${#LINES[@]}
echo "[fwd50 K=$K seq] total=$N chunk=$CHUNK seed=$SEED"

i=0
while (( i < N )); do
  sub_file=$(mktemp)
  for ((j=0; j<CHUNK && i+j<N; j++)); do
    echo "${LINES[$((i+j))]}" >> "$sub_file"
  done
  start_idx="${LINES[$i]}"
  end_i=$((i + CHUNK - 1)); (( end_i >= N )) && end_i=$((N-1))
  end_idx="${LINES[$end_i]}"
  out_name=$(printf "fwd50_%05d_%05d.pkl" "$start_idx" "$end_idx")
  out_file="$OUT_DIR/$out_name"
  if [ -f "$out_file" ]; then rm "$sub_file"; i=$((i + CHUNK)); continue; fi
  /usr/bin/python3 "$SCRIPT" --k "$K" --seed "$SEED" --frame-indices-file "$sub_file" \
    --out "$out_name" --out-dir-override "$OUT_DIR" 2>&1 | \
    grep -vE "^cc |^rm |^make|GAUSS|If you|Note:|OR the|^$|WARNING|MPC not solved|MPC get init|Vehicle state:|Cost:|^[0-9]+ Fail:|output_time|status: NaN|status: QP|steering_angle:|^-L|SQP_RTI" | \
    tail -3
  rm "$sub_file"
  i=$((i + CHUNK))
  if (( i % 20 == 0 )); then echo "[progress] $i/$N"; fi
done

echo "[fwd50 K=$K seq] done $i/$N"
ls "$OUT_DIR"/fwd50_*.pkl 2>/dev/null | wc -l

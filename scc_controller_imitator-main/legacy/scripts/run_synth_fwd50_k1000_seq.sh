#!/usr/bin/env bash
# fwd50 K=1000 sequential, CHUNK=1 (frame per subprocess).
# missing frame indices file 만 처리. acados shared lib race 회피.
set -u
K=1000
SEED=${1:-1000}
OUT_DIR="/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_k100_v1"
SCRIPT="/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/scripts/synthesize_scc_kn_random.py"
IDX_FILE="${IDX_FILE:-$OUT_DIR/fwd50_frame_indices.txt}"

LINES=($(cat "$IDX_FILE"))
N=${#LINES[@]}
echo "[fwd50 K=$K seq] missing=$N seed=$SEED"

processed=0
for idx in "${LINES[@]}"; do
  sub_file=$(mktemp)
  echo "$idx" > "$sub_file"
  out_name=$(printf "fwd50_seq_%05d.pkl" "$idx")
  out_file="$OUT_DIR/$out_name"
  if [ -f "$out_file" ]; then
    rm "$sub_file"; processed=$((processed+1)); continue
  fi
  /usr/bin/python3 "$SCRIPT" --k "$K" --seed "$SEED" --frame-indices-file "$sub_file" \
    --out "$out_name" --out-dir-override "$OUT_DIR" 2>&1 | \
    grep -vE "^cc |^rm |^make|GAUSS|If you|Note:|OR the|^$|WARNING|MPC not solved|MPC get init|Vehicle state:|Cost:|^[0-9]+ Fail:|output_time|status: NaN|status: QP|steering_angle:|^-L|SQP_RTI" | \
    tail -3
  rm "$sub_file"
  processed=$((processed+1))
  if (( processed % 10 == 0 )); then
    echo "[progress] $processed/$N"
  fi
done

echo "[fwd50 K=$K seq] done $processed/$N"
ls "$OUT_DIR"/fwd50_seq_*.pkl 2>/dev/null | wc -l

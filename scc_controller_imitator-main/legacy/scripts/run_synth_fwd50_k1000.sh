#!/usr/bin/env bash
# Forward-50m lead frames (ex<50, |ey|<5), K=1000 random trajectories per frame.
# 843 frame × K=1000 = 843,000 solves. chunk=2 (frame). prefix=fwd50_.
# 3 partition (병렬 실행용 — 시작 시 OFFSET argument 로 분할). 단일 process 면 OFFSET=0, STEP=1.
set -u
OFFSET=${1:-0}    # 0, 1, 2 (3 partition 시)
STEP=${2:-1}      # 1 = no partition, 3 = parallel 3
SEED=${3:-1000}
K=1000
CHUNK=2
OUT_DIR="/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_k100_v1"
SCRIPT="/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/scripts/synthesize_scc_kn_random.py"
IDX_FILE="$OUT_DIR/fwd50_frame_indices.txt"

# OFFSET partition (line-modulo)
ALL_LINES=($(cat "$IDX_FILE"))
N=${#ALL_LINES[@]}
echo "[fwd50 K=$K] total=$N  partition: offset=$OFFSET step=$STEP  seed=$SEED"

i=$OFFSET
while (( i < N )); do
  sub_file=$(mktemp)
  written=0
  for ((j=0; j<CHUNK && i + j*STEP < N; j++)); do
    echo "${ALL_LINES[$((i + j*STEP))]}" >> "$sub_file"
    written=$((written+1))
  done
  if [ $written -eq 0 ]; then rm "$sub_file"; break; fi
  start_idx="${ALL_LINES[$i]}"
  end_i=$((i + (CHUNK-1)*STEP)); (( end_i >= N )) && end_i=$((N-1))
  end_idx="${ALL_LINES[$end_i]}"
  prefix="fwd50_o${OFFSET}"
  out_name=$(printf "${prefix}_%05d_%05d.pkl" "$start_idx" "$end_idx")
  out_file="$OUT_DIR/$out_name"
  if [ -f "$out_file" ]; then
    rm "$sub_file"; i=$((i + CHUNK*STEP)); continue
  fi
  /usr/bin/python3 "$SCRIPT" --k "$K" --seed "$SEED" --frame-indices-file "$sub_file" \
    --out "$out_name" --out-dir-override "$OUT_DIR" 2>&1 | \
    grep -vE "^cc |^rm |^make|GAUSS|If you|Note:|OR the|^$|WARNING|MPC not solved|MPC get init|Vehicle state:|Cost:|^[0-9]+ Fail:|output_time|status: NaN|status: QP|steering_angle:|^-L|SQP_RTI" | \
    tail -5
  rm "$sub_file"
  i=$((i + CHUNK*STEP))
done

echo "[fwd50 K=$K o=$OFFSET] done"

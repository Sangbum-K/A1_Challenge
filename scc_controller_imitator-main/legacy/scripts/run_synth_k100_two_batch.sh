#!/usr/bin/env bash
# Batch 1: lead-only × K=100 (모두). Batch 2: no-lead × K=100 (max 100 frames sample).
# chunk_size=2 (segfault 방어, K=100 무거움). incremental save active.
set -u
K=100
CHUNK=2
OUT_DIR="/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_k100_v1"
mkdir -p "$OUT_DIR"
SCRIPT="/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/scripts/synthesize_scc_kn_random.py"

echo "=================="
echo "BATCH 1: lead-only frames × K=$K"
echo "=================="
# 6877 frame 전체를 chunk=2 로 돈다 (각 chunk 안에서 lead filter 적용 → lead 없는 chunk 는 빠르게 끝남)
for ((start=0; start<6877; start+=CHUNK)); do
  end=$((start+CHUNK))
  out_name=$(printf "lead_chunk_%05d_%05d.pkl" "$start" "$end")
  out_file="$OUT_DIR/$out_name"
  if [ -f "$out_file" ]; then
    echo "[skip] $out_name"
    continue
  fi
  /usr/bin/python3 "$SCRIPT" --k "$K" --start "$start" --end "$end" \
    --lead-filter lead-only --out "$out_name" --out-dir-override "$OUT_DIR" 2>&1 | \
    grep -vE "^cc |^rm |^make|GAUSS|If you|Note:|OR the|^$|WARNING|MPC not solved|MPC get init|Vehicle state:|Cost:|^[0-9]+ Fail:|output_time|status: NaN|status: QP|steering_angle:|^-L|SQP_RTI" | \
    tail -8
done

echo ""
echo "=================="
echo "BATCH 2: no-lead frames × K=$K (chunk 분할, ~100 frames total)"
echo "=================="
# 6877 range 를 chunk=140 으로 나눈 ~50 chunk × max-frames=2 = ~100 no-lead frames.
# 각 chunk seed 다양 (시작 frame 기반).
NL_CHUNK=140
for ((start=0; start<6877; start+=NL_CHUNK)); do
  end=$((start+NL_CHUNK))
  out_name=$(printf "no_lead_chunk_%05d_%05d.pkl" "$start" "$end")
  out_file="$OUT_DIR/$out_name"
  if [ -f "$out_file" ]; then
    echo "[skip] $out_name"
    continue
  fi
  /usr/bin/python3 "$SCRIPT" --k "$K" --start "$start" --end "$end" \
    --lead-filter no-lead --max-frames 2 --seed "$start" \
    --out "$out_name" --out-dir-override "$OUT_DIR" 2>&1 | \
    grep -vE "^cc |^rm |^make|GAUSS|If you|Note:|OR the|^$|WARNING|MPC not solved|MPC get init|Vehicle state:|Cost:|^[0-9]+ Fail:|output_time|status: NaN|status: QP|steering_angle:|^-L|SQP_RTI" | \
    tail -5
done

echo "[done]"
ls "$OUT_DIR" | wc -l

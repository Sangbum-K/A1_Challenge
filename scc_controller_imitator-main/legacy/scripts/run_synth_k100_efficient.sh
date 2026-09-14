#!/usr/bin/env bash
# 효율적 K=100 합성: lead/no-lead frame index 미리 추출 → 그것만 chunk process
set -u
K=100
CHUNK_FRAMES=2   # frame 당 K=100 → chunk 당 200 solve (안전)
OUT_DIR="/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_k100_v1"
mkdir -p "$OUT_DIR"
SCRIPT="/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/scripts/synthesize_scc_kn_random.py"

LEAD_IDX_FILE="$OUT_DIR/lead_frame_indices.txt"
NOLEAD_IDX_FILE="$OUT_DIR/nolead_frame_indices_sample100.txt"

echo "[step 1] extract frame indices (lead / no-lead sample) ..."
/home/ailab-12/miniforge3/envs/ml_con/bin/python <<PYEOF
import json, numpy as np
META = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/multiagnet_compact_meta.json"
NPZ = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/multiagnet_compact.npz"
BEV_X_MAX, BEV_Y_MIN, BEV_CELL = 80.0, -25.0, 0.5
def bbox_to_xy(b):
    cx, cy, _, _ = b
    return BEV_X_MAX - cy * BEV_CELL, BEV_Y_MIN + cx * BEV_CELL

meta = json.load(open(META))
z = np.load(NPZ)
lead_idx, nolead_idx = [], []
for m in meta:
    i = m["i"]
    ego = z[f"s{i}__ego_pose"]
    if float(ego[3]) < 2.0:
        continue
    has_lead = False
    for o in m["objects"]:
        xm, ym = bbox_to_xy(o["bbox"])
        if xm > 2.0 and abs(ym) < 3.5 and xm < 120.0:
            has_lead = True; break
    (lead_idx if has_lead else nolead_idx).append(i)
print(f"lead frames: {len(lead_idx)}, no-lead frames: {len(nolead_idx)}")
with open("$LEAD_IDX_FILE", "w") as f:
    for i in lead_idx: f.write(f"{i}\n")
rng = np.random.RandomState(0)
sel = sorted(rng.choice(nolead_idx, min(100, len(nolead_idx)), replace=False))
with open("$NOLEAD_IDX_FILE", "w") as f:
    for i in sel: f.write(f"{int(i)}\n")
print(f"wrote: {len(lead_idx)} lead idx, {len(sel)} no-lead idx (sampled 100)")
PYEOF

NLEAD=$(wc -l < "$LEAD_IDX_FILE")
NNOLEAD=$(wc -l < "$NOLEAD_IDX_FILE")
echo "lead: $NLEAD, no-lead sample: $NNOLEAD"

process_indices_file() {
  local idx_file=$1
  local prefix=$2
  local count=$(wc -l < "$idx_file")
  echo ""
  echo "[step] process $prefix ($count frames) chunk=$CHUNK_FRAMES"
  local lines=($(cat "$idx_file"))
  local n=${#lines[@]}
  local i=0
  while (( i < n )); do
    local sub_file=$(mktemp)
    for ((j=0; j<CHUNK_FRAMES && i+j<n; j++)); do
      echo "${lines[$((i+j))]}" >> "$sub_file"
    done
    local start_idx="${lines[$i]}"
    local end_i=$((i + CHUNK_FRAMES - 1))
    (( end_i >= n )) && end_i=$((n - 1))
    local end_idx="${lines[$end_i]}"
    local out_name=$(printf "${prefix}_%05d_%05d.pkl" "$start_idx" "$end_idx")
    local out_file="$OUT_DIR/$out_name"
    if [ -f "$out_file" ]; then
      rm "$sub_file"; i=$((i + CHUNK_FRAMES)); continue
    fi
    /usr/bin/python3 "$SCRIPT" --k "$K" --frame-indices-file "$sub_file" \
      --out "$out_name" --out-dir-override "$OUT_DIR" 2>&1 | \
      grep -vE "^cc |^rm |^make|GAUSS|If you|Note:|OR the|^$|WARNING|MPC not solved|MPC get init|Vehicle state:|Cost:|^[0-9]+ Fail:|output_time|status: NaN|status: QP|steering_angle:|^-L|SQP_RTI" | \
      tail -5
    rm "$sub_file"
    i=$((i + CHUNK_FRAMES))
  done
}

echo ""
echo "=================="
echo "BATCH 1: lead-only × K=$K (efficient)"
echo "=================="
process_indices_file "$LEAD_IDX_FILE" "lead"

echo ""
echo "=================="
echo "BATCH 2: no-lead × K=$K (100 sample)"
echo "=================="
process_indices_file "$NOLEAD_IDX_FILE" "nolead"

echo "[done]"
ls "$OUT_DIR" | wc -l

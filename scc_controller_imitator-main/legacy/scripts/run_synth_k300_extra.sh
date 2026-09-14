#!/usr/bin/env bash
# Phase 5: lead K +100 more (seed=2, prefix lead3_) + no-lead 400 new frame K=100 (seed=3, prefix nolead3_).
# Phase 4 와 병렬 실행 안전 (다른 prefix). 두 phase 합치면 lead K=300 + no-lead 600 frame.
set -u
K=100
CHUNK=2
OUT_DIR="/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_k100_v1"
mkdir -p "$OUT_DIR"
SCRIPT="/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/scripts/synthesize_scc_kn_random.py"

LEAD_IDX_FILE="$OUT_DIR/lead_frame_indices.txt"
NOLEAD_NEW_IDX_FILE="$OUT_DIR/nolead_frame_indices_extra400.txt"

echo "[step] generate 400 NEW no-lead frame indices (exclude existing 100+200=300) ..."
/home/ailab-12/miniforge3/envs/ml_con/bin/python <<PYEOF
import json, numpy as np, os
META = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/multiagnet_compact_meta.json"
NPZ = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/multiagnet_compact.npz"
EXISTING_FILES = [
    "$OUT_DIR/nolead_frame_indices_sample100.txt",
    "$OUT_DIR/nolead_frame_indices_extra200.txt",
]
OUT = "$NOLEAD_NEW_IDX_FILE"

BEV_X_MAX, BEV_Y_MIN, BEV_CELL = 80.0, -25.0, 0.5
def bbox_to_xy(b):
    cx, cy, _, _ = b
    return BEV_X_MAX - cy * BEV_CELL, BEV_Y_MIN + cx * BEV_CELL

existing = set()
for fp in EXISTING_FILES:
    if os.path.exists(fp):
        with open(fp) as f:
            for ln in f: existing.add(int(ln.strip()))
print(f"existing exclude: {len(existing)}")

meta = json.load(open(META))
z = np.load(NPZ)
nolead_idx = []
for m in meta:
    i = m["i"]
    if i in existing: continue
    ego = z[f"s{i}__ego_pose"]
    if float(ego[3]) < 2.0: continue
    has_lead = False
    for o in m["objects"]:
        xm, ym = bbox_to_xy(o["bbox"])
        if xm > 2.0 and abs(ym) < 3.5 and xm < 120.0:
            has_lead = True; break
    if not has_lead: nolead_idx.append(i)
rng = np.random.RandomState(3)
sel = sorted(rng.choice(nolead_idx, min(400, len(nolead_idx)), replace=False))
with open(OUT, "w") as f:
    for i in sel: f.write(f"{int(i)}\n")
print(f"wrote {len(sel)} extra no-lead indices to {OUT}")
PYEOF

process_indices_file() {
  local idx_file=$1; local prefix=$2; local seed=$3
  local lines=($(cat "$idx_file"))
  local n=${#lines[@]}
  echo "[step] process $prefix ($n frames) seed=$seed chunk=$CHUNK"
  local i=0
  while (( i < n )); do
    local sub_file=$(mktemp)
    for ((j=0; j<CHUNK && i+j<n; j++)); do
      echo "${lines[$((i+j))]}" >> "$sub_file"
    done
    local start_idx="${lines[$i]}"
    local end_i=$((i + CHUNK - 1)); (( end_i >= n )) && end_i=$((n - 1))
    local end_idx="${lines[$end_i]}"
    local out_name=$(printf "${prefix}_%05d_%05d.pkl" "$start_idx" "$end_idx")
    local out_file="$OUT_DIR/$out_name"
    if [ -f "$out_file" ]; then
      rm "$sub_file"; i=$((i + CHUNK)); continue
    fi
    /usr/bin/python3 "$SCRIPT" --k "$K" --seed "$seed" --frame-indices-file "$sub_file" \
      --out "$out_name" --out-dir-override "$OUT_DIR" 2>&1 | \
      grep -vE "^cc |^rm |^make|GAUSS|If you|Note:|OR the|^$|WARNING|MPC not solved|MPC get init|Vehicle state:|Cost:|^[0-9]+ Fail:|output_time|status: NaN|status: QP|steering_angle:|^-L|SQP_RTI" | \
      tail -5
    rm "$sub_file"
    i=$((i + CHUNK))
  done
}

echo ""
echo "=================="
echo "BATCH 1C: lead K=100 extra (seed=2) → 합치면 K=300"
echo "=================="
process_indices_file "$LEAD_IDX_FILE" "lead3" 2

echo ""
echo "=================="
echo "BATCH 2C: no-lead 400 NEW frame K=100"
echo "=================="
process_indices_file "$NOLEAD_NEW_IDX_FILE" "nolead3" 3

echo "[done phase 5]"
ls "$OUT_DIR" | wc -l

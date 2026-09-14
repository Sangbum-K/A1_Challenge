#!/bin/bash
# a_2-on-b0 ablation: b0 backbone 으로 i1 token cache 4-set 재생성.
#   각 cache 는 동명 v2_s cache 와 동일 key 집합/순서 (build_i1_tok_cache_b0.py 가 복제).
#   subset = spd_v2 + sangam_lead_0520 + sangam_lead_0521 + DriverB_new (작업지시 대표셋).
set -e
source /home/ailab-12/miniforge3/etc/profile.d/conda.sh
conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
RAW_SSD=/home/ailab-12/AILabSSD/02_Data/57_A1_E2E/00_raw
RAW_DRVB=/home/ailab-12/AILabDataset/02_Custom_Dataset/57_A1_E2E/00_raw
SEG_B0=outputs_multi/seg_pretrain_lite_b0/best_miou_backbone.pt
mkdir -p "$D/cache/b0"
BS=24; W=12

build () {
  local v2s=$1 aux=$2 raw=$3 out=$4
  if [ -f "$out" ]; then echo "[skip] $(basename $out) exists"; return; fi
  echo "=== [$(date +%H:%M:%S)] build $(basename $out) ==="
  python -u scripts/build_i1_tok_cache_b0.py \
    --v2s-cache "$v2s" --aux-dir "$aux" --raw-base "$raw" \
    --seg-ckpt "$SEG_B0" --out "$out" --batch-size $BS --workers $W
}

build "$D/cache/i1_tok_cache_sangam_lead_0520_prev20.npz" "$D/aux_targets_sangam_lead_0520" "$RAW_SSD" \
      "$D/cache/b0/i1_tok_cache_b0_sangam_lead_0520_prev20.npz"
build "$D/cache/i1_tok_cache_sangam_lead_0521_prev20.npz" "$D/aux_targets_sangam_lead_0521" "$RAW_SSD" \
      "$D/cache/b0/i1_tok_cache_b0_sangam_lead_0521_prev20.npz"
build "$D/cache/i1_tok_cache_spd_v2_prev20.npz" "$D/aux_targets_spd_nolead_0520" "$RAW_SSD" \
      "$D/cache/b0/i1_tok_cache_b0_spd_v2_prev20.npz"
build "$D/cache/i1_tok_cache_DriverB_new_prev20.npz" "$D/aux_targets_DriverB_new" "$RAW_DRVB" \
      "$D/cache/b0/i1_tok_cache_b0_DriverB_new_prev20.npz"
echo "=== ALL B0 CACHES DONE $(date) ==="

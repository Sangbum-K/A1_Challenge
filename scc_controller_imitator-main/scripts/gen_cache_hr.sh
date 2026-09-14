#!/bin/bash
# stride16 80-token 포함 cache (--hr) 재생성 — bin80 ablation 입력.
# manifest = {raw_base, samples} dict (meta list 를 감쌈). raw=AILabSSD, seg=fx640_v3.
# seg(GPU) 공존 위해 batch 작게. 출력: data/cache/i1_tok_cache_{X}_hr.npz
set -e
source /home/ailab-12/miniforge3/etc/profile.d/conda.sh; conda activate ml_con
cd /home/ailab-12/git/scc_controller_imitator
D=/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data
RAW=/home/ailab-12/AILabSSD/02_Data/57_A1_E2E/00_raw
SEG=outputs_multi/seg_pretrain_fx640_v3/best_miou_backbone.pt
mkdir -p /tmp/hr_manifest

# meta(list) → manifest({raw_base, samples}) 변환. spd 는 cache 명 spd_v2 로.
python3 - <<PY
import json
RAW="$RAW"; D="$D"
for ds,out in [("sangam_lead_0520","sangam_lead_0520"),("sangam_lead_0521","sangam_lead_0521"),("spd_nolead_0520","spd_v2")]:
    meta=json.load(open(f"{D}/meta_{ds}.json"))
    json.dump({"raw_base":RAW,"samples":meta}, open(f"/tmp/hr_manifest/{out}.json","w"))
    print(ds,"->",out,"n=",len(meta))
PY

for X in sangam_lead_0520 sangam_lead_0521 spd_v2; do
  echo "### cache --hr: $X $(date)"
  python -u scripts/compute_i1_tok_cache.py --hr \
    --manifest /tmp/hr_manifest/$X.json \
    --seg-ckpt "$SEG" \
    --out "$D/cache/i1_tok_cache_${X}_hr.npz" \
    --prev-dtick 5 20 --batch-size 32 --workers 4
done
echo "### cache --hr ALL DONE $(date)"

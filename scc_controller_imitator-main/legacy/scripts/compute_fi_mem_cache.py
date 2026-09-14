"""fi_mem 캐시 사전계산 (DataLoader 병렬 I/O 버전).

각 frame 의 image 를 DataLoader workers 로 병렬 load, GPU 에서 batched forward.
저장: data/fi_mem_cache_v1.npz {keys, fi_mem (N, 40, D) float16, meta}.

사용:
  python -u scripts/compute_fi_mem_cache.py --seg-ckpt ... --batch-size 64 --workers 8
"""
from __future__ import annotations
import os, sys, json, argparse, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.datamodule.scc_synth_pair_dataset import (
    PREV_DTICK, _load_roi, _img_to_tensor, _parse_ego, _yaw_shift,
    DEFAULT_MANIFEST,
)
from src.model.image_models.image_scc_models_v2 import SCCControllerImitatorV2

DEG2RAD = np.pi / 180.0


class CacheFrameDataset(Dataset):
    def __init__(self, keys, key_meta, raw_base):
        self.keys = keys
        self.key_meta = key_meta
        self.raw_base = Path(raw_base)

    def __len__(self): return len(self.keys)

    def __getitem__(self, i):
        key = self.keys[i]
        s = self.key_meta[key]
        run = s["run_id"]; tick = s["tick"]
        run_dir = self.raw_base / s["track"] / s["policy"] / run
        ci = run_dir / "CAMERA_1" / f"{tick}_Intensity.jpeg"
        pi = run_dir / "CAMERA_1" / f"{tick - PREV_DTICK}_Intensity.jpeg"
        ce = run_dir / "EGO_INFO" / f"{tick}.txt"
        pe = run_dir / "EGO_INFO" / f"{tick - PREV_DTICK}.txt"
        if not ci.exists():
            # skip — return None marker
            return {"key": key, "ok": False,
                    "img_c": torch.zeros(3, 250, 1280),
                    "img_p": torch.zeros(3, 250, 1280)}
        ego_c = _parse_ego(ce); ego_p = _parse_ego(pe) if pe.exists() else None
        d_yaw = (((ego_c["orientation"][2] if ego_c else 0)
                  - (ego_p["orientation"][2] if ego_p else 0)) * DEG2RAD
                  if (ego_c and ego_p) else 0.0)
        roi_c = _load_roi(ci)
        if pi.exists():
            roi_p = _yaw_shift(_load_roi(pi), d_yaw)
        else:
            roi_p = roi_c.copy()
        return {"key": key, "ok": True,
                "img_c": _img_to_tensor(roi_c),
                "img_p": _img_to_tensor(roi_p)}


def collate_cache(batch):
    return {
        "key": [b["key"] for b in batch],
        "ok":  [b["ok"]  for b in batch],
        "img_c": torch.stack([b["img_c"] for b in batch], dim=0),
        "img_p": torch.stack([b["img_p"] for b in batch], dim=0),
    }


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-ckpt", required=True)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out", default="/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/fi_mem_cache_v1.npz")
    ap.add_argument("--backbone", default="tf_efficientnetv2_s")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f"[cache] manifest from {args.manifest}", flush=True)
    m = json.load(open(args.manifest))
    keys = sorted({s["key"] for s in m["samples"]})
    if args.limit: keys = keys[:args.limit]
    key_meta = {}
    for s in m["samples"]:
        if s["key"] not in key_meta: key_meta[s["key"]] = s
    print(f"  unique keys: {len(keys)}", flush=True)

    ds = CacheFrameDataset(keys, key_meta, m["raw_base"])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                     num_workers=args.workers, pin_memory=True, collate_fn=collate_cache,
                     persistent_workers=args.workers > 0)

    model = SCCControllerImitatorV2(
        img_backbone=args.backbone, d_model=args.d_model, traj_seq_len=40,
    ).to(args.device)
    sd = torch.load(args.seg_ckpt, map_location="cpu")
    own = model.img_encoder.backbone.state_dict()
    n_loaded = 0
    for k, v in sd.items():
        k2 = k.replace("backbone.", "")
        if k2 in own and own[k2].shape == v.shape:
            own[k2] = v; n_loaded += 1
    model.img_encoder.backbone.load_state_dict(own)
    print(f"  seg_pretrain loaded: {n_loaded} keys", flush=True)
    model.eval()

    N = len(keys); D = args.d_model
    fi_mem_all = np.zeros((N, 40, D), dtype=np.float16)
    ok_mask = np.zeros(N, dtype=bool)

    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for batch_i, batch in enumerate(dl):
            cb = batch["img_c"].to(args.device, non_blocking=True)
            pb = batch["img_p"].to(args.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=True):
                tc = model.img_encoder(cb)
                tp = model.img_encoder(pb)
                fm = model.img_fusion(tc, tp)   # (B, 40, D)
            fm_np = fm.float().cpu().numpy().astype(np.float16)
            for k_off, key in enumerate(batch["key"]):
                if not batch["ok"][k_off]: continue
                gi = keys.index(key) if batch_i == 0 else None
                # cached
            # 위 lookup 비효율적 — sequential 이므로 직접 매핑
            base = batch_i * args.batch_size
            for k_off, ok in enumerate(batch["ok"]):
                gi = base + k_off
                if gi >= N: break
                if not ok: continue
                fi_mem_all[gi] = fm_np[k_off]
                ok_mask[gi] = True
            n_done += cb.size(0)
            if (batch_i + 1) % 5 == 0 or n_done >= N:
                dt = time.time() - t0
                rate = n_done / max(dt, 1e-3)
                eta = (N - n_done) / max(rate, 1e-3)
                print(f"  [{n_done}/{N}] rate={rate:.1f}/s  ETA={eta:.0f}s", flush=True)

    n_ok = int(ok_mask.sum())
    print(f"\n[done] {n_ok}/{N} frames cached", flush=True)
    out_keys = [k for i, k in enumerate(keys) if ok_mask[i]]
    out_fi = fi_mem_all[ok_mask]
    print(f"  saving {out_fi.shape} ({out_fi.nbytes/1e6:.1f} MB) → {args.out}", flush=True)
    np.savez_compressed(args.out,
                         keys=np.array(out_keys),
                         fi_mem=out_fi,
                         meta=json.dumps({
                             "d_model": args.d_model,
                             "backbone": args.backbone,
                             "seed": args.seed,
                             "seg_ckpt": args.seg_ckpt,
                         }))
    print(f"  done.", flush=True)


if __name__ == "__main__":
    main()

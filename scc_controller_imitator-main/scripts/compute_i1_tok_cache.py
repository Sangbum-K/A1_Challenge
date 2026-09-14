"""i_1 token cache (per-image, fi_mem cache 의 v3 — i_y token shift 와 짝).

(현재) compute_fi_mem_cache.py 는 (curr, prev_pixel_warp) → i_1 + i_2 fusion → fi_mem (40, D)
을 frame 단위로 cache. prev_dt 바뀌면 무효.

(이 script) per-image 단위 i_1 token cache:
  - 모든 unique (run, tick) 이미지 (curr 와 prev 양쪽) 를 한 번씩 i_1 forward
  - 결과 fi_tok (40, C_b) 저장 (no fusion, no yaw warp)
  - 학습 시점에 dataset 이 tok_curr / tok_prev_raw 따로 load, model 내부에서
    i_y (np.roll by dx bins) + i_2 (image fusion) 수행
  - prev_dt 무관 — 한 번 cache 로 다양한 Δt 실험 가능

출력: AILabDatasetLocal/scc_controller_imitator/data/cache/i1_tok_cache_<tag>.npz
  - keys: list[str] "run__t{tick:07d}"
  - tokens: (N_keys, 40, C_b) float16
  - meta: dict {backbone, C_b, src_camera_fx (참고용), seg_ckpt}

사용:
  python -u scripts/compute_i1_tok_cache.py \
    --seg-ckpt data/ckpts/seg_pretrain_v2s_full_best_backbone.pt \
    --manifest .../scc_synth_pair_manifest.json \
    --prev-dtick 5 20  # 캐시할 prev offset 들 (curr 외 추가로 필요한 image tick)
"""
from __future__ import annotations
import os, sys, json, argparse, time
from pathlib import Path
import numpy as np
import torch
import timm
from torch.utils.data import Dataset, DataLoader
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_MANIFEST = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_pair_manifest.json"
DEFAULT_OUT = "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/cache/i1_tok_cache_v1.npz"

ROI_Y0, ROI_Y1 = 235, 485
IMG_H, IMG_W = 720, 1280
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_roi_tensor(path: str) -> torch.Tensor:
    img = np.array(Image.open(path).convert("RGB"))
    roi = img[ROI_Y0:ROI_Y1]
    arr = roi.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


class ImagePathDataset(Dataset):
    def __init__(self, keys, key_to_path):
        self.keys = keys
        self.k2p = key_to_path

    def __len__(self): return len(self.keys)

    def __getitem__(self, i):
        key = self.keys[i]
        path = self.k2p[key]
        if not os.path.exists(path):
            return {"key": key, "ok": False, "img": torch.zeros(3, 250, 1280)}
        try:
            return {"key": key, "ok": True, "img": _load_roi_tensor(path)}
        except Exception:
            return {"key": key, "ok": False, "img": torch.zeros(3, 250, 1280)}


def collate(batch):
    return {
        "key": [b["key"] for b in batch],
        "ok":  [b["ok"]  for b in batch],
        "img": torch.stack([b["img"] for b in batch], dim=0),
    }


def get_args():
    ap = argparse.ArgumentParser()
    # registry 의 best seg_backbone 을 default 로 (CURRENT_BEST.json 참조)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.utils.ckpt_registry import get_best
        _seg_best = get_best("seg_backbone")
        _default_seg = _seg_best["path"] if _seg_best else None
    except Exception:
        _default_seg = None
    ap.add_argument("--seg-ckpt", default=_default_seg,
                    help="default: registry (data/ckpts/CURRENT_BEST.json) 의 seg_backbone.path. "
                         "명시 시 그 값 우선.")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--backbone", default="tf_efficientnetv2_s")
    ap.add_argument("--prev-dtick", nargs="*", type=int, default=[5, 20],
                    help="cache 할 prev tick offset 들 (curr 외 추가). 예: 5 20 = 50ms+200ms prev image 도 cache")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--hr", action="store_true",
                    help="stride16 80-col token(tokens_s16) 추가 저장 (bin80 1D-merge 용). off=기존 stride32만")
    return ap.parse_args()


def main():
    args = get_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)

    print(f"[cache] manifest {args.manifest}", flush=True)
    m = json.load(open(args.manifest))
    raw_p = Path(m["raw_base"])

    # unique image keys + paths (curr + prev_dtick offsets)
    keys_set = set(); k2p = {}
    samples = m["samples"]
    if args.limit: samples = samples[:args.limit]
    for s in samples:
        run = s["run_id"]; tick = int(s["tick"])
        track = s["track"]; policy = s["policy"]
        ticks = [tick] + [tick - dt for dt in args.prev_dtick]
        for t in ticks:
            key = f"{run}__t{t:07d}"
            if key in keys_set: continue
            path = raw_p / track / policy / run / "CAMERA_1" / f"{t}_Intensity.jpeg"
            keys_set.add(key); k2p[key] = str(path)
    keys = sorted(keys_set)
    print(f"  unique image keys: {len(keys)} (curr + prev_dtick {args.prev_dtick})", flush=True)

    # i_1 only — backbone + AdaptiveAvgPool + Conv1d(C_b → C_b) — 단순화 위해
    # SCCControllerImitatorV2.img_encoder 와 호환되도록 backbone + proj 둘다 cache
    # 하지만 proj 은 model-side 학습 가능 → backbone 직후 (C_b, 1, 40) 까지만 cache
    # 그러면 학습 시 model 의 proj (Conv1d C_b → D) 를 거치도록.
    _oi = [-2, -1] if args.hr else [-1]
    backbone = timm.create_model(
        args.backbone, pretrained=False, features_only=True, out_indices=_oi
    ).to(args.device).eval()
    # seg pretrain load
    sd = torch.load(args.seg_ckpt, map_location=args.device, weights_only=False)
    own = backbone.state_dict()
    n_loaded = 0
    for k, v in sd.items():
        k2 = k.replace("backbone.", "")
        if k2 in own and own[k2].shape == v.shape:
            own[k2] = v; n_loaded += 1
    backbone.load_state_dict(own)
    print(f"  seg backbone loaded: {n_loaded} keys, C_b={backbone.feature_info.channels()[-1]}", flush=True)

    C_b = backbone.feature_info.channels()[-1]
    C16 = backbone.feature_info.channels()[-2] if args.hr else None
    N = len(keys)
    tok_all = np.zeros((N, 40, C_b), dtype=np.float16)
    tok16_all = np.zeros((N, 80, C16), dtype=np.float16) if args.hr else None
    if args.hr:
        print(f"  [hr] stride16 추가: C16={C16} → tokens_s16 (N,80,{C16})", flush=True)
    ok_mask = np.zeros(N, dtype=bool)

    ds = ImagePathDataset(keys, k2p)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                     num_workers=args.workers, pin_memory=True, collate_fn=collate,
                     persistent_workers=args.workers > 0)

    t0 = time.time(); n_done = 0
    with torch.no_grad():
        for batch_i, b in enumerate(dl):
            x = b["img"].to(args.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=True):
                feats = backbone(x)
                f32 = feats[-1]                                       # (B, C_b, ~8, ~40)
                tok = torch.nn.functional.adaptive_avg_pool2d(f32, (1, 40)).squeeze(2).transpose(1, 2)  # (B,40,C_b)
                if args.hr:
                    f16 = feats[-2]                                  # (B, C16, ~16, ~80)
                    tok16 = torch.nn.functional.adaptive_avg_pool2d(f16, (1, 80)).squeeze(2).transpose(1, 2)  # (B,80,C16)
            tok_np = tok.float().cpu().numpy().astype(np.float16)
            tok16_np = tok16.float().cpu().numpy().astype(np.float16) if args.hr else None
            base = batch_i * args.batch_size
            for k_off, ok in enumerate(b["ok"]):
                gi = base + k_off
                if gi >= N: break
                if not ok: continue
                tok_all[gi] = tok_np[k_off]
                if args.hr: tok16_all[gi] = tok16_np[k_off]
                ok_mask[gi] = True
            n_done += x.size(0)
            if (batch_i + 1) % 20 == 0 or n_done >= N:
                dt = time.time() - t0
                rate = n_done / max(dt, 1e-3)
                eta = (N - n_done) / max(rate, 1e-3)
                print(f"  [{n_done}/{N}] rate={rate:.0f}/s ETA={eta:.0f}s", flush=True)

    n_ok = int(ok_mask.sum())
    print(f"\n[done] {n_ok}/{N} images cached", flush=True)
    out_keys = [k for i, k in enumerate(keys) if ok_mask[i]]
    out_tok = tok_all[ok_mask]
    out_tok16 = tok16_all[ok_mask] if args.hr else None
    print(f"  saving {out_tok.shape} ({out_tok.nbytes/1e6:.1f} MB)"
          + (f" + s16 {out_tok16.shape} ({out_tok16.nbytes/1e6:.1f} MB)" if args.hr else "")
          + f" → {args.out}", flush=True)
    # registry 의 metric 도 함께 저장 (downstream consistency check 용)
    _bk_metrics = {}
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.utils.ckpt_registry import get_best
        _r = get_best("seg_backbone")
        if _r and _r.get("path") == args.seg_ckpt:
            _bk_metrics = {k: v for k, v in _r.items() if k not in ("path", "name", "date", "note")}
    except Exception: pass
    _save_kw = dict(keys=np.array(out_keys), tokens=out_tok,
                    meta=json.dumps({
                        "backbone": args.backbone,
                        "C_b": int(C_b),
                        "C16": (int(C16) if args.hr else None),
                        "hr": bool(args.hr),
                        "prev_dtick": args.prev_dtick,
                        "seg_ckpt": args.seg_ckpt,
                        "seg_metrics": _bk_metrics,
                    }))
    if args.hr:
        _save_kw["tokens_s16"] = out_tok16
    np.savez_compressed(args.out, **_save_kw)
    print("  done.", flush=True)


if __name__ == "__main__":
    main()

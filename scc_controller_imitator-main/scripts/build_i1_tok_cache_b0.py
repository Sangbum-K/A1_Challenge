"""b0 backbone 으로 i_1 token cache 재생성 (a_2-on-b0 ablation 용).

핵심: 기존 v2_s i1 cache 와 **동일한 key 집합·동일 순서** 로 b0 토큰을 만든다.
  → dataset join / val-split(seed=0) 이 v2_s 와 1:1 일치 → 공정 비교.

key→raw image path 복원은 aux 디렉토리 파일명에서 ({track}__{policy}__{run}__t{tick}.npz)
  + raw_base(데이터셋별) 로 수행. v2_s cache 의 key 순서를 그대로 유지.

b0 last channel = 192 (≠ v2_s 256) → 출력 tokens (N,40,192).

사용:
  python scripts/build_i1_tok_cache_b0.py \
    --v2s-cache <v2s i1 cache.npz> \
    --aux-dir <aux dir> --raw-base <raw_base> \
    --seg-ckpt outputs_multi/seg_pretrain_lite_b0/best_miou_backbone.pt \
    --out <out b0 cache.npz>
"""
from __future__ import annotations
import os, sys, json, argparse, time, glob, re
from pathlib import Path
import numpy as np
import torch
import timm
from torch.utils.data import Dataset, DataLoader
from PIL import Image

ROI_Y0, ROI_Y1 = 235, 485
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
AUX_PAT = re.compile(r"^(.+?)__(.+?)__([^_].*?)__t(\d+)\.npz$")


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

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        key = self.keys[i]
        path = self.k2p.get(key)
        if not path or not os.path.exists(path):
            return {"key": key, "ok": False, "img": torch.zeros(3, 250, 1280)}
        try:
            return {"key": key, "ok": True, "img": _load_roi_tensor(path)}
        except Exception:
            return {"key": key, "ok": False, "img": torch.zeros(3, 250, 1280)}


def collate(batch):
    return {
        "key": [b["key"] for b in batch],
        "ok": [b["ok"] for b in batch],
        "img": torch.stack([b["img"] for b in batch], dim=0),
    }


def build_key2path(aux_dir, raw_base):
    """aux 파일명 → key (run__t{tick:07d}) → raw image path."""
    k2p = {}
    for f in glob.glob(str(Path(aux_dir) / "*.npz")):
        name = Path(f).name
        m = AUX_PAT.match(name)
        if not m:
            continue
        tr, pol, run, tick = m.group(1), m.group(2), m.group(3), int(m.group(4))
        key = f"{run}__t{tick:07d}"
        p = Path(raw_base) / tr / pol / run / "CAMERA_1" / f"{tick}_Intensity.jpeg"
        k2p[key] = str(p)
    return k2p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2s-cache", required=True, help="동일 key 집합/순서 복제 원본")
    ap.add_argument("--aux-dir", required=True)
    ap.add_argument("--raw-base", required=True)
    ap.add_argument("--seg-ckpt", required=True, help="b0 seg backbone ckpt")
    ap.add_argument("--backbone", default="tf_efficientnetv2_b0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-keys", type=int, default=0,
                    help="0=전부. >0 이면 v2s cache 에서 균등 stride 로 N key 만 forward (NFS 느릴 때 subset). "
                         "전체 run 을 대표(편중 회피). eval 시 v2s 도 동일 key set 으로 제한해야 공정.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)

    z = np.load(args.v2s_cache, allow_pickle=True)
    keys = z["keys"].tolist()
    if args.max_keys and args.max_keys < len(keys):
        # run-block 샘플링: run 단위 통째 선택(temporal prev offset 보존) + 전체 run 에서 균등 stride 로 run 선택(다양성).
        from collections import OrderedDict
        run2keys = OrderedDict()
        for k in keys:
            r = k.split("__t")[0]
            run2keys.setdefault(r, []).append(k)
        run_list = list(run2keys.keys())
        # run 들을 stride 로 순회하며 max_keys 도달까지 통째 추가
        picked = []
        rstride = max(1, len(run_list) // max(1, (args.max_keys // max(1, len(keys) // len(run_list)))))
        ri = 0
        while len(picked) < args.max_keys and ri < len(run_list):
            for r in run_list[ri::rstride]:
                picked.extend(run2keys[r])
                if len(picked) >= args.max_keys:
                    break
            ri += 1
            if rstride == 1:
                break
        keys = picked
        n_runs_used = len(set(k.split("__t")[0] for k in keys))
        print(f"[b0-cache] SUBSET run-block {len(keys):,} keys / {n_runs_used} runs "
              f"(temporal-preserving, --max-keys)", flush=True)
    print(f"[b0-cache] v2s cache keys={len(keys):,} <- {Path(args.v2s_cache).name}", flush=True)

    k2p = build_key2path(args.aux_dir, args.raw_base)
    n_cov = sum(1 for k in keys if k in k2p)
    print(f"  key2path covers {n_cov:,}/{len(keys):,} (aux={Path(args.aux_dir).name})", flush=True)
    if n_cov < 0.99 * len(keys):
        raise RuntimeError(f"[FAIL] aux key coverage {n_cov}/{len(keys)} < 99% — wrong aux/raw_base")

    backbone = timm.create_model(
        args.backbone, pretrained=False, features_only=True, out_indices=[-1]
    ).to(args.device).eval()
    sd = torch.load(args.seg_ckpt, map_location=args.device, weights_only=False)
    own = backbone.state_dict()
    n_loaded = 0
    for k, v in sd.items():
        k2 = k.replace("backbone.", "")
        if k2 in own and own[k2].shape == v.shape:
            own[k2] = v; n_loaded += 1
    backbone.load_state_dict(own)
    C_b = backbone.feature_info.channels()[-1]
    print(f"  {args.backbone} loaded {n_loaded} keys, C_b={C_b}", flush=True)

    N = len(keys)
    tok_all = np.zeros((N, 40, C_b), dtype=np.float16)
    ok_mask = np.zeros(N, dtype=bool)

    ds = ImagePathDataset(keys, k2p)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                    pin_memory=True, collate_fn=collate, persistent_workers=args.workers > 0)

    t0 = time.time(); n_done = 0
    with torch.no_grad():
        for bi, b in enumerate(dl):
            x = b["img"].to(args.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=True):
                feat = backbone(x)[-1]
                feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 40))
                tok = feat.squeeze(2).transpose(1, 2)   # (B,40,C_b)
            tok_np = tok.float().cpu().numpy().astype(np.float16)
            base = bi * args.batch_size
            for j, ok in enumerate(b["ok"]):
                gi = base + j
                if gi >= N:
                    break
                if not ok:
                    continue
                tok_all[gi] = tok_np[j]; ok_mask[gi] = True
            n_done += x.size(0)
            if (bi + 1) % 50 == 0 or n_done >= N:
                dt = time.time() - t0; rate = n_done / max(dt, 1e-3)
                print(f"  [{n_done}/{N}] rate={rate:.0f}/s ETA={(N-n_done)/max(rate,1e-3):.0f}s", flush=True)

    n_ok = int(ok_mask.sum())
    print(f"[done] {n_ok}/{N} cached", flush=True)
    # v2_s 와 1:1 비교 위해 key 순서·집합 유지. ok 안 된 것은 그대로 0 토큰 (거의 없어야).
    np.savez_compressed(
        args.out,
        keys=np.array(keys),
        tokens=tok_all,
        meta=json.dumps({
            "backbone": args.backbone,
            "C_b": int(C_b),
            "seg_ckpt": args.seg_ckpt,
            "src_v2s_cache": args.v2s_cache,
            "n_ok": n_ok,
        }))
    print(f"  saved {tok_all.shape} ({tok_all.nbytes/1e6:.1f} MB) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

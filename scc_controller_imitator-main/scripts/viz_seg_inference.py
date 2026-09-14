"""seg pretrain ckpt inference 시각화.

각 sample row: [원본 image | GT mask | pred mask | overlay]
컴포지트 PNG 1장 + index.html 생성.

사용:
  python scripts/viz_seg_inference.py \
      --ckpt outputs_multi/seg_pretrain_fx640_v3/best_miou_full.pt \
      --n 12 --out outputs_multi/seg_pretrain_fx640_v3/inference_demo
"""
from __future__ import annotations
import os, sys, json, argparse, random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib; matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.datamodule.seg_pretrain_dataset import (
    IMAGENET_MEAN, IMAGENET_STD, ROI_Y0, ROI_Y1,
    _load_image_roi, _img_to_tensor, _mask_to_label, IGNORE_IDX,
)
from src.model.image_models.seg_pretrain_model import SegPretrainModel

PALETTE = np.array([
    [0, 0, 0], [80, 80, 80], [255, 80, 80], [80, 200, 255]
], dtype=np.uint8)

DATASETS = [
    {
        "name":    "speedway",
        "meta":    "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/speedway_agent1_meta.json",
        "mask":    "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/seg_masks_spd",
        "raw":     "/home/ailab-12/AILabSSD/02_Data/57_A1_E2E/00_raw",
    },
    {
        "name":    "sangam_fx640",
        "meta":    "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/sangam_fx640_meta.json",
        "mask":    "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/seg_masks_sangam_fx640",
        "raw":     "/home/ailab-12/AILabSSD/02_Data/57_A1_E2E/00_raw",
    },
]


def _denorm(img_t: torch.Tensor) -> np.ndarray:
    arr = img_t.detach().cpu().numpy().transpose(1, 2, 0)
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _label_to_rgb(label: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    valid = label >= 0
    rgb[valid] = PALETTE[label[valid]]
    return rgb


def _overlay(img, mask_rgb, alpha=0.5):
    return (img.astype(np.float32) * (1 - alpha) +
            mask_rgb.astype(np.float32) * alpha).astype(np.uint8)


def collect_samples(n_per_dataset: int, seed: int = 0):
    rng = random.Random(seed)
    samples = []
    for ds in DATASETS:
        meta = json.load(open(ds["meta"]))
        mask_dir = Path(ds["mask"])
        with_mask = [m for m in meta if (mask_dir / f"{m['frame_id']}.npz").exists()]
        rng.shuffle(with_mask)
        picks = with_mask[:n_per_dataset]
        for m in picks:
            samples.append({**m, "_ds": ds["name"], "_mask_dir": str(mask_dir), "_raw": ds["raw"]})
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="tf_efficientnetv2_s")
    ap.add_argument("--n", type=int, default=12, help="dataset 별 sample 수 (total = 2 × n)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[viz] loading {args.ckpt}")
    model = SegPretrainModel(backbone=args.backbone, n_classes=4).to(args.device).eval()
    sd = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
    m, u = model.load_state_dict(sd, strict=False)
    print(f"  loaded (missing={len(m)} unexpected={len(u)})")

    samples = collect_samples(args.n, args.seed)
    print(f"[viz] {len(samples)} samples")

    # build composite (N rows × 4 cols)
    H, W = ROI_Y1 - ROI_Y0, 1280
    PAD = 4
    N = len(samples)
    panel = np.zeros((N * (H + PAD) - PAD, 4 * W + 3 * PAD, 3), dtype=np.uint8)

    for i, s in enumerate(samples):
        img_path = f"{s['_raw']}/{s['track']}/{s['policy']}/{s['run_id']}/CAMERA_1/{s['tick']}_Intensity.jpeg"
        if not os.path.exists(img_path):
            print(f"  skip {s['frame_id']}: img missing")
            continue
        roi = _load_image_roi(img_path)
        img_t = _img_to_tensor(roi).unsqueeze(0).to(args.device)
        with torch.no_grad():
            logits = model(img_t)               # (1, 4, H, W)
            pred = logits.argmax(dim=1)[0].cpu().numpy()
        # GT
        mask_arr = np.load(f"{s['_mask_dir']}/{s['frame_id']}.npz")["mask"]
        gt = _mask_to_label(mask_arr)
        # ROI image
        img_np = _denorm(img_t[0])
        gt_rgb = _label_to_rgb(gt)
        pr_rgb = _label_to_rgb(pred)
        ov = _overlay(img_np, pr_rgb, alpha=0.45)

        y0 = i * (H + PAD)
        x0 = 0
        panel[y0:y0+H, x0:x0+W]                   = img_np
        panel[y0:y0+H, x0+W+PAD:x0+2*W+PAD]       = gt_rgb
        panel[y0:y0+H, x0+2*(W+PAD):x0+3*W+2*PAD] = pr_rgb
        panel[y0:y0+H, x0+3*(W+PAD):x0+4*W+3*PAD] = ov

    out_png = args.out / "composite.png"
    Image.fromarray(panel).save(out_png, optimize=True)
    print(f"[viz] saved composite → {out_png}")

    # per-sample images + html
    comp_dir = args.out / "rows"
    comp_dir.mkdir(exist_ok=True)
    rows_html = []
    for i, s in enumerate(samples):
        row_h = H
        row = panel[i * (H + PAD):i * (H + PAD) + row_h]
        rp = comp_dir / f"{i:02d}_{s['_ds']}_{s['frame_id']}.png"
        Image.fromarray(row).save(rp, optimize=True)
        rows_html.append(
            f'<div class="row"><div class="lab">[{s["_ds"]}] {s["frame_id"]}</div>'
            f'<img src="rows/{rp.name}" loading="lazy"></div>')

    html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>Seg Pretrain v3 Inference Demo</title>
<style>
body {{ font-family: sans-serif; background:#1c1c1c; color:#ddd; margin:12px; }}
.row {{ margin-bottom:12px; background:#2a2a2a; padding:6px; border-radius:6px; }}
.lab {{ font-size:11px; color:#888; padding:4px; }}
.row img {{ width:100%; display:block; }}
h2 {{ font-weight:300; }}
</style></head><body>
<h2>Seg Pretrain v3 ({args.ckpt}) — columns: image | GT | pred | overlay</h2>
<p>color: <span style="color:#5c5">road=gray</span>, <span style="color:#f55">vehicle=red</span>, <span style="color:#5cf">wall=cyan</span>, bg=black</p>
{''.join(rows_html)}
</body></html>"""
    (args.out / "index.html").write_text(html)
    print(f"[viz] index → {args.out}/index.html")


if __name__ == "__main__":
    main()

"""a_2 ckpt inference 시각화 — 객체 많은 frame 우선 선택.

각 row: [image | GT obj overlay | pred obj overlay | per-channel bars]

사용:
  python scripts/viz_a2_inference.py \
      --ckpt outputs_multi/a2_fx640_v3_local/best_loss.pt \
      --pretrained outputs_multi/seg_pretrain_fx640_v3/best_miou_backbone.pt \
      --min-obj 3 --n 8 \
      --out outputs_multi/a2_fx640_v3_local/inference_many_obj
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.datamodule.a2_pair_dataset import (
    A2PairDataset, IMAGENET_MEAN, IMAGENET_STD, ROI_Y0, ROI_Y1,
)
from src.model.image_models.a2_model import A2Model


SPEEDWAY = {
    "meta": "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/speedway_agent1_meta.json",
    "a2":   "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/aux_targets_spd_fx640",
    "raw":  "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/raw_cache",
}


def _denorm(img_t):
    arr = img_t.detach().cpu().numpy().transpose(1, 2, 0)
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",       required=True)
    ap.add_argument("--pretrained", default=None)
    ap.add_argument("--backbone",   default="tf_efficientnetv2_s")
    ap.add_argument("--min-obj",    type=int, default=3, help="GT obj bin count ≥ 이만큼인 frame 만")
    ap.add_argument("--n",          type=int, default=8)
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--out",        required=True, type=Path)
    ap.add_argument("--prev-dtick", type=int, default=20)
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("[viz] loading speedway dataset (require_obj=True)...")
    ds = A2PairDataset(meta_path=SPEEDWAY["meta"], a2_dir=SPEEDWAY["a2"],
                        raw_base=SPEEDWAY["raw"], require_obj=True,
                        prev_dtick=args.prev_dtick)
    print(f"  {len(ds)} samples with obj")

    # quick scan: load only target (npz read) per sample to count obj bins
    rng = np.random.RandomState(args.seed)
    idxs = list(range(len(ds))); rng.shuffle(idxs)
    selected = []
    for i in idxs:
        s = ds.samples[i]
        try:
            tgt = np.load(s["ap"])["target"]
            n_obj = int((tgt[:, 0] > 0.5).sum())
        except Exception:
            continue
        if n_obj >= args.min_obj:
            selected.append((i, n_obj))
            if len(selected) >= args.n: break
    if not selected:
        print(f"[viz] no samples found with obj_bin >= {args.min_obj}, lowering...")
        selected = [(i, 0) for i in idxs[:args.n]]
    selected.sort(key=lambda x: -x[1])
    print(f"[viz] picked {len(selected)} samples, max obj_bin={selected[0][1]}")

    # model
    print(f"[viz] loading model {args.ckpt}")
    model = A2Model(img_backbone=args.backbone).to(args.device).eval()
    if args.pretrained:
        info = model.load_seg_pretrain(args.pretrained)
        print(f"  pretrained: {info}")
    sd = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
    m, u = model.load_state_dict(sd, strict=False)
    print(f"  loaded (missing={len(m)} unexpected={len(u)})")

    # composite: rows = samples, each row has 5 sub-panels stacked horizontally
    # (1) image (250 × 1280)
    # (2) GT obj bin overlay (250 × 1280) — red boxes per obj bin
    # (3) pred obj bin overlay (250 × 1280) — green boxes per pred obj bin (sigmoid > 0.5)
    # (4) bar chart: GT vs pred for [obj, vL_rel, dist] (×3 stacked, height 240)
    H, W = ROI_Y1 - ROI_Y0, 1280
    BARH = 180
    PAD = 6
    PANEL_W = W

    from PIL import ImageDraw

    rows = []
    for idx, (i, n_obj) in enumerate(selected):
        item = ds[i]
        img_c = item["image_curr"].unsqueeze(0).to(args.device)
        img_p = item["image_prev_rot"].unsqueeze(0).to(args.device)
        ego   = item["ego_motion"].unsqueeze(0).to(args.device)
        with torch.no_grad():
            logits = model(img_c, img_p, ego)[0]      # (40, 7)
        target = item["target"].numpy()                # (40, 7)
        pred = logits.cpu().numpy()
        obj_gt = target[:, 0] > 0.5
        obj_pr = 1.0 / (1.0 + np.exp(-pred[:, 0])) > 0.5

        # (1) image
        img_rgb = _denorm(item["image_curr"])
        # (2) GT overlay
        gt_img = Image.fromarray(img_rgb.copy())
        d = ImageDraw.Draw(gt_img)
        bin_w = W / 40
        for b in range(40):
            if obj_gt[b]:
                d.rectangle([b*bin_w, 0, (b+1)*bin_w-1, H-1], outline=(255, 60, 60), width=3)
        # (3) pred overlay
        pr_img = Image.fromarray(img_rgb.copy())
        d = ImageDraw.Draw(pr_img)
        for b in range(40):
            if obj_pr[b]:
                d.rectangle([b*bin_w, 0, (b+1)*bin_w-1, H-1], outline=(60, 200, 60), width=3)

        # (4) bar chart: matplotlib → np
        fig, axes = plt.subplots(3, 1, figsize=(W/100, BARH*3/100), dpi=100)
        x = np.arange(40)
        # obj
        axes[0].bar(x-0.2, target[:, 0], width=0.4, color="red", alpha=0.7, label="GT")
        axes[0].bar(x+0.2, 1.0/(1.0+np.exp(-pred[:, 0])), width=0.4, color="lime", alpha=0.7, label="pred")
        axes[0].set_title(f"obj (GT n={int(obj_gt.sum())}, pred n={int(obj_pr.sum())})", fontsize=8)
        axes[0].set_ylim(0, 1.1); axes[0].legend(fontsize=6, loc="upper right")
        # vL_rel (ch 3)
        axes[1].bar(x-0.2, target[:, 3], width=0.4, color="red", alpha=0.7)
        axes[1].bar(x+0.2, pred[:, 3]*20.0, width=0.4, color="lime", alpha=0.7)   # pred scaled back
        axes[1].set_title("vL_rel (GT red, pred green, m/s)", fontsize=8)
        # dist (ch 6)
        axes[2].bar(x-0.2, target[:, 6], width=0.4, color="red", alpha=0.7)
        axes[2].bar(x+0.2, pred[:, 6]*40.0, width=0.4, color="lime", alpha=0.7)
        axes[2].set_title("dist (GT red, pred green, m)", fontsize=8)
        for ax in axes:
            ax.set_xlim(-0.5, 39.5); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        # render to numpy
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        canvas = FigureCanvasAgg(fig); canvas.draw()
        bar_np, (bw, bh) = canvas.print_to_buffer()
        bar_np = np.frombuffer(bar_np, dtype=np.uint8).reshape(bh, bw, 4)[..., :3]
        plt.close(fig)
        if bar_np.shape[1] != W:
            bar_im = Image.fromarray(bar_np).resize((W, bar_np.shape[0] * W // bar_np.shape[1]))
            bar_np = np.array(bar_im)

        # stack vertically: image + gt + pred + bar
        sub_h = H + H + H + bar_np.shape[0] + 3*PAD
        row = np.full((sub_h, W, 3), 255, dtype=np.uint8)
        y = 0
        row[y:y+H] = img_rgb; y += H + PAD
        row[y:y+H] = np.array(gt_img); y += H + PAD
        row[y:y+H] = np.array(pr_img); y += H + PAD
        row[y:y+bar_np.shape[0]] = bar_np

        rp = args.out / f"{idx:02d}_obj{n_obj}_{item['frame_id']}.png"
        Image.fromarray(row).save(rp, optimize=True)
        rows.append((rp.name, n_obj, item["frame_id"]))
        print(f"  [{idx+1}/{len(selected)}] saved {rp.name}")

    # html
    rows_html = "\n".join(
        f'<div class="card"><div class="cap">[{n}obj] {fid}</div>'
        f'<img src="{name}" loading="lazy"></div>'
        for (name, n, fid) in rows)
    html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>a_2 v3 inference (many-obj)</title>
<style>
body {{ font-family: sans-serif; background:#1c1c1c; color:#ddd; margin:12px; }}
.card {{ background:#2a2a2a; padding:6px; border-radius:6px; margin-bottom:12px; }}
.cap {{ font-size:12px; color:#aaa; padding:4px; }}
.card img {{ width:100%; display:block; }}
h2 {{ font-weight:300; }}
</style></head><body>
<h2>a_2 v3 inference — many-obj samples</h2>
<p>각 row: image / GT bin overlay (red) / pred bin overlay (green) / bar charts (obj, vL_rel, dist)</p>
{rows_html}
</body></html>"""
    (args.out / "index.html").write_text(html)
    print(f"\n[viz] index → {args.out}/index.html")


if __name__ == "__main__":
    main()

"""SAM mask + 전방 15m 이내 차량 케이스 시각화.

각 frame 에 대해:
  - 상단: full image (ROI 박스 + lead 의 image-plane 투영 박스)
  - 중단: ROI + SAM mask overlay (기존 npz 우선, 없으면 신규 추론)
  - 하단: BEV — ego + 모든 객체 + closest-lead 강조 + lane (|n|<2m) 가이드

사용:
  ~/miniforge3/envs/ml_con/bin/python scripts/viz_sam_close_leads.py [--limit 6] [--out outputs_multi/sam_close]
"""
from __future__ import annotations
import os, sys, json, argparse, pickle, time
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# IPM helpers (jiwon_e2e)
E2E_ROOT = Path("/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing")
sys.path.insert(0, str(E2E_ROOT))
from src.data.ipm import _invert_extrinsic, _world_to_image
K_CAM = np.load(str(E2E_ROOT / "calibration/CAMERA_1__intrinsic.npy"))
E_CAM = np.load(str(E2E_ROOT / "calibration/CAMERA_1__extrinsic.npy"))
R_V2C, T_V2C = _invert_extrinsic(E_CAM)

# BEV bbox → ego (synth_k1 convention)
BEV_X_MAX = 80.0; BEV_Y_MIN = -25.0; BEV_CELL = 0.5
def bbox_to_ego(b):
    cx, cy, w_cells, h_cells = b
    return (BEV_X_MAX - cy*BEV_CELL, BEV_Y_MIN + cx*BEV_CELL,
            h_cells*BEV_CELL, w_cells*BEV_CELL)

META_JSON = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/multiagnet_compact_meta.json"
RAW_BASE = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/data/raw/sangam_track/MultiAgnet"
SEG_DIR = Path("/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/seg_masks_v1")

# from generate_image_seg_masks
ROI_Y_TOP, ROI_Y_BOT = 235, 485

CLASS_COLORS = np.array([
    [  0,   0,   0],   # 0 bg
    [ 80, 200,  80],   # 1 road
    [255,  80,  80],   # 2 vehicle
    [ 80, 100, 255],   # 3 wall
    [200, 200,  60],   # 4 ignore
], dtype=np.uint8)
CLASS_NAMES = ["bg", "road", "vehicle", "wall", "ignore"]


def project_box(ex, ey, L, W, h=1.5):
    corners = np.array([
        [ex+L/2, ey+W/2, -0.13], [ex+L/2, ey-W/2, -0.13],
        [ex-L/2, ey-W/2, -0.13], [ex-L/2, ey+W/2, -0.13],
        [ex+L/2, ey+W/2, -0.13+h], [ex+L/2, ey-W/2, -0.13+h],
        [ex-L/2, ey-W/2, -0.13+h], [ex-L/2, ey+W/2, -0.13+h],
    ])
    return _world_to_image(corners, K_CAM, R_V2C, T_V2C)


def colorize_mask(mask):
    K, H, W = mask.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for c in [1, 4, 3, 2]:
        if c >= K: continue
        m = mask[c].astype(bool)
        out[m] = CLASS_COLORS[c]
    return out


def overlay_image(rgb, mask_rgb, alpha=0.5):
    blend = alpha * mask_rgb.astype(np.float32) + (1-alpha) * rgb.astype(np.float32)
    return np.clip(blend, 0, 255).astype(np.uint8)


def get_seg_mask(frame_id, img_path, gd_sam):
    """기존 npz 있으면 로드, 없으면 GD+SAM2 신규 추론."""
    npz_path = SEG_DIR / f"{frame_id}.npz"
    if npz_path.exists():
        return np.load(npz_path)["mask"], "cached"
    if gd_sam is None:
        return None, "no_sam"
    gd_model, sam2_predictor = gd_sam
    from generate_image_seg_masks import process_frame
    mask = process_frame(str(img_path), gd_model, sam2_predictor)
    return mask, "fresh"


def visualize_frame(meta_entry, gd_sam, out_path):
    fid = meta_entry["frame_id"]
    run = meta_entry["run_id"]
    tick = int(meta_entry["tick"])
    img_path = Path(RAW_BASE) / run / "CAMERA_1" / f"{tick}_Intensity.jpeg"
    if not img_path.exists():
        return False, "image missing"

    img_full = np.array(Image.open(img_path).convert("RGB"))
    img_roi  = img_full[ROI_Y_TOP:ROI_Y_BOT]
    mask, src_kind = get_seg_mask(fid, img_path, gd_sam)

    all_objs = []
    for o in meta_entry["objects"]:
        ex, ey, L, W = bbox_to_ego(o["bbox"])
        all_objs.append((ex, ey, L, W))
    closest = min((o for o in all_objs if 1.5 < o[0] < 50 and abs(o[1]) < 8),
                   key=lambda o: o[0], default=None)

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1.6], width_ratios=[2, 1.2],
                            hspace=0.25, wspace=0.10)

    # (0, :) full image with object box projections
    ax_img = fig.add_subplot(gs[0, :])
    ax_img.imshow(img_full)
    ax_img.add_patch(Rectangle((0, ROI_Y_TOP), 1280, ROI_Y_BOT-ROI_Y_TOP,
                                  ec='lime', fc='none', lw=2, label='ROI'))
    for (ex, ey, L, W) in all_objs:
        if not (1.5 < ex < 80 and abs(ey) < 20): continue
        uv = project_box(ex, ey, L, W)
        if (uv[:, 2] <= 0).any(): continue
        is_closest = (closest is not None and abs(ex-closest[0]) < 0.01 and abs(ey-closest[1]) < 0.01)
        color = "red" if is_closest else "yellow"
        lw = 3 if is_closest else 1
        poly = uv[:4, :2]
        ax_img.plot(np.append(poly[:,0], poly[0,0]),
                     np.append(poly[:,1], poly[0,1]),
                     color=color, lw=lw)
        top = uv[4:, :2]
        ax_img.plot(np.append(top[:,0], top[0,0]),
                     np.append(top[:,1], top[0,1]),
                     color=color, lw=lw-0.5)
        for j in range(4):
            ax_img.plot([uv[j,0], uv[j+4,0]], [uv[j,1], uv[j+4,1]],
                          color=color, lw=lw-0.5)
    title_extra = f"closest lead: ({closest[0]:.1f}, {closest[1]:+.1f}) m" if closest else "no close lead"
    ax_img.set_title(f"{fid}   {title_extra}", fontsize=11)
    ax_img.axis("off")

    # (1, :) ROI + SAM overlay
    ax_seg = fig.add_subplot(gs[1, :])
    if mask is not None:
        mask_rgb = colorize_mask(mask)
        overlay = overlay_image(img_roi, mask_rgb, alpha=0.5)
        ax_seg.imshow(overlay)
        px_sum = " ".join(f"{CLASS_NAMES[c]}={int(mask[c].sum())}" for c in range(mask.shape[0]))
        ax_seg.set_title(f"SAM overlay ({src_kind})   {px_sum}", fontsize=10)
    else:
        ax_seg.imshow(img_roi)
        ax_seg.set_title("SAM mask 없음 (cache+sam 모두 미사용)", fontsize=10)
    ax_seg.axis("off")

    # (2, 0) BEV
    ax_bev = fig.add_subplot(gs[2, 0])
    ax_bev.add_patch(Rectangle((-2.5, -1.0), 5, 2, ec='k', fc='lightgray', alpha=0.7))
    # lane filter ±2m (직선 가정 — 진짜 lane 모양은 path 따라가지만 가시화 단순화)
    ax_bev.axhline(+2.0, color='gray', ls=':', lw=1, alpha=0.5)
    ax_bev.axhline(-2.0, color='gray', ls=':', lw=1, alpha=0.5, label='|n|=2m')
    for (ex, ey, L, W) in all_objs:
        if not (-2 < ex < 80 and abs(ey) < 25): continue
        is_closest = closest is not None and abs(ex-closest[0])<0.01 and abs(ey-closest[1])<0.01
        ax_bev.add_patch(Rectangle((ex-L/2, ey-W/2), L, W,
                                       ec='red' if is_closest else 'gray',
                                       fc='salmon' if is_closest else 'lightgray',
                                       alpha=0.8 if is_closest else 0.5,
                                       lw=2 if is_closest else 1))
        if is_closest or (1.5 < ex < 30 and abs(ey) < 8):
            ax_bev.text(ex, ey+W/2+0.3, f"({ex:.1f},{ey:+.1f})",
                          ha='center', fontsize=9,
                          color='red' if is_closest else 'dimgray',
                          fontweight='bold' if is_closest else 'normal')
    ax_bev.set_xlim(-5, 35)
    ax_bev.set_ylim(-12, 12)
    ax_bev.set_aspect("equal")
    ax_bev.grid(True, alpha=0.3)
    ax_bev.set_xlabel("x [m] forward")
    ax_bev.set_ylabel("y [m] left")
    ax_bev.set_title("BEV (ego frame, lane ±2m gray dotted)")
    ax_bev.legend(loc='upper right', fontsize=9)

    # (2, 1) summary text
    ax_info = fig.add_subplot(gs[2, 1])
    ax_info.axis("off")
    lines = [f"frame: {fid}", f"closest lead: ({closest[0]:.2f}, {closest[1]:+.2f}) m" if closest else "no lead",
             "", f"objects in scene: {len(all_objs)}"]
    if closest:
        lane_in = abs(closest[1]) <= 2.0
        lines.append(f"lane filter (|n|≤2m): {'✓ IN' if lane_in else '× OUT'}")
        lines.append(f"longitudinal distance: {closest[0]:.1f} m")
    if mask is not None:
        lines.append(f"\nSAM source: {src_kind}")
        lines.append(f"vehicle px: {int(mask[2].sum())}")
    ax_info.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=11,
                   family="monospace", transform=ax_info.transAxes)

    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True, src_kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=6, help="처음 N 개 frame")
    ap.add_argument("--out", type=str, default="outputs_multi/sam_close_leads")
    ap.add_argument("--max-x", type=float, default=15.0, help="lead 전방 거리 cutoff (m)")
    ap.add_argument("--no-fresh-sam", action="store_true",
                    help="SAM 신규 추론 안 함 (cached npz 만)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"meta 로딩 ...")
    meta = json.load(open(META_JSON))

    near = []
    for m in meta:
        for o in m["objects"]:
            ex, ey, _, _ = bbox_to_ego(o["bbox"])
            if 1.5 < ex < args.max_x and abs(ey) < 8.0:
                near.append(m); break
    print(f"전방 {args.max_x:.0f}m 이내 차량 있는 frame: {len(near)} / {len(meta)}")

    near = near[:args.limit]

    # cached 우선, 필요시만 GD+SAM2 lazy load
    gd_sam = None
    n_fresh_needed = sum(1 for m in near if not (SEG_DIR / f'{m["frame_id"]}.npz').exists())
    if not args.no_fresh_sam and n_fresh_needed > 0:
        print(f"신규 SAM 추론 {n_fresh_needed} frame — model 로딩 ...")
        from generate_image_seg_masks import (
            GD_CONFIG, GD_WEIGHT, SAM2_CONFIG, SAM2_WEIGHT)
        from groundingdino.util.inference import load_model
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        gd_model = load_model(GD_CONFIG, str(GD_WEIGHT), device="cuda")
        sam2_model = build_sam2(SAM2_CONFIG, str(SAM2_WEIGHT), device="cuda")
        gd_sam = (gd_model, SAM2ImagePredictor(sam2_model))
        print("loaded.\n")

    t0 = time.time()
    n_done = 0
    for m in near:
        out_path = out_dir / f"{m['frame_id']}.png"
        ok, src = visualize_frame(m, gd_sam, out_path)
        if ok:
            n_done += 1
            print(f"  [{n_done}/{len(near)}] {m['frame_id']} → {out_path.name} ({src})")
    print(f"\ndone {n_done}/{len(near)} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

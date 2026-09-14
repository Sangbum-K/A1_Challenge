"""SAM seg mask 검증 도구 — 임의 입력에 대해 GroundingDINO + SAM2 실행 후 시각화.

용도:
  1. 새 데이터셋 image 가 들어왔을 때 pipeline 동작 확인
  2. 기존 seg_masks_v1/*.npz 의 결과 시각화
  3. 특정 image 의 GD 박스 + SAM2 mask 디버그

사용:
  # 단일 image: GD+SAM2 돌려서 결과 PNG 저장
  ~/miniforge3/envs/ml_con/bin/python scripts/inspect_seg_mask.py --image /path/to/img.jpg

  # 디렉토리 모든 image:
  ~/miniforge3/envs/ml_con/bin/python scripts/inspect_seg_mask.py --image-dir /path/to/dir --limit 10

  # 기존 mask npz 시각화 (SAM 안 돌림, 빠름):
  ~/miniforge3/envs/ml_con/bin/python scripts/inspect_seg_mask.py --mask-npz data/seg_masks_v1/<frame>.npz

  # 기존 mask npz + 그 frame 의 raw image 비교:
  ~/miniforge3/envs/ml_con/bin/python scripts/inspect_seg_mask.py --mask-npz <npz> --image <jpg>

  # 새 데이터셋의 첫 N 장 sample 시각화 (예: AILabDataset 의 새 폴더):
  ~/miniforge3/envs/ml_con/bin/python scripts/inspect_seg_mask.py --image-dir <new_dir> --limit 5 --out outputs_multi/seg_check/
"""
from __future__ import annotations
import os, sys, argparse, time
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# class colors: 0 bg, 1 road, 2 vehicle, 3 wall, 4 ignore
CLASS_COLORS = np.array([
    [  0,   0,   0],     # 0 bg (transparent)
    [ 80, 200,  80],     # 1 road  (green)
    [255,  80,  80],     # 2 vehicle (red)
    [ 80, 100, 255],     # 3 wall (blue)
    [200, 200,  60],     # 4 ignore (yellow)
], dtype=np.uint8)
CLASS_NAMES = ["bg", "road", "vehicle", "wall", "ignore"]


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """(K, H, W) uint8 → (H, W, 3) RGB overlay (binary OR across classes, last writes win)."""
    K, H, W = mask.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    # priority: vehicle > wall > road > ignore > bg
    for c in [1, 4, 3, 2]:   # 마지막에 vehicle 이 위에 오도록
        if c >= K: continue
        m = mask[c].astype(bool)
        out[m] = CLASS_COLORS[c]
    return out


def overlay_image(rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """(H, W, 3) + (H, W, 3) → blended (H, W, 3)."""
    blend = (alpha * mask_rgb.astype(np.float32)
             + (1 - alpha) * rgb.astype(np.float32))
    return np.clip(blend, 0, 255).astype(np.uint8)


def run_pipeline(img_path: Path, gd_model, sam2_predictor):
    """기존 generate_image_seg_masks.process_frame 와 동일 흐름."""
    from generate_image_seg_masks import (
        process_frame, ROI_Y_TOP, ROI_Y_BOT)
    mask = process_frame(str(img_path), gd_model, sam2_predictor)
    # also return ROI image
    img_full = np.array(Image.open(img_path).convert("RGB"))
    img_roi = img_full[ROI_Y_TOP:ROI_Y_BOT]
    return img_roi, mask, img_full


def viz(img_full, img_roi, mask, title, out_path):
    """4-panel: full image / ROI / mask overlay / per-class breakdown."""
    K = mask.shape[0]
    fig, axes = plt.subplots(K + 2, 1, figsize=(16, 2.0 * (K + 2)))

    axes[0].imshow(img_full)
    axes[0].add_patch(plt.Rectangle((0, 235), 1280, 250,
                                       fill=False, ec="lime", lw=2))
    axes[0].set_title(f"FULL  {title}  (lime box = ROI [235:485])", fontsize=10)
    axes[0].axis("off")

    mask_rgb = colorize_mask(mask)
    overlay = overlay_image(img_roi, mask_rgb, alpha=0.5)
    axes[1].imshow(overlay)
    px_summary = "  ".join(
        f"{CLASS_NAMES[c]}={int(mask[c].sum())}" for c in range(K))
    axes[1].set_title(f"ROI + overlay   {px_summary}", fontsize=10)
    axes[1].axis("off")

    for c in range(K):
        row = c + 2
        axes[row].imshow(img_roi, alpha=0.6)
        m = mask[c].astype(bool)
        if m.any():
            colored = np.zeros((*m.shape, 4), dtype=np.uint8)
            colored[..., :3] = CLASS_COLORS[c]
            colored[..., 3] = m.astype(np.uint8) * 200
            axes[row].imshow(colored)
        axes[row].set_title(
            f"class {c} ({CLASS_NAMES[c]})   px={int(mask[c].sum())}",
            fontsize=9)
        axes[row].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",     type=str, default=None,
                    help="단일 image 처리 (GD+SAM2 실행)")
    ap.add_argument("--image-dir", type=str, default=None,
                    help="디렉토리의 *.jpg/*.png 모두 처리")
    ap.add_argument("--mask-npz",  type=str, default=None,
                    help="기존 mask npz 만 시각화 (SAM 안 돌림). 같이 --image 주면 raw 와 비교")
    ap.add_argument("--limit",     type=int, default=None,
                    help="--image-dir 일때 처음 N 장만")
    ap.add_argument("--out",       type=str, default="outputs_multi/seg_check",
                    help="결과 PNG 저장 디렉토리")
    ap.add_argument("--save-mask", action="store_true",
                    help="--image / --image-dir 결과 mask 도 npz 로 저장")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # mask-npz only: 빠른 시각화, GD/SAM2 안 띄움
    if args.mask_npz and not args.image and not args.image_dir:
        mask = np.load(args.mask_npz)["mask"]
        fid = Path(args.mask_npz).stem
        # 같은 frame_id 의 raw image 추측 (sangam_track__MultiAgnet__run__tXXXXX)
        img_path = None
        if "MultiAgnet" in fid:
            parts = fid.split("__")
            if len(parts) >= 4:
                run = parts[2]
                tick = parts[3].lstrip("t").lstrip("0") or "0"
                cand = Path(f"/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/data/raw/sangam_track/MultiAgnet/{run}/CAMERA_1/{tick}_Intensity.jpeg")
                if cand.exists(): img_path = cand
        if img_path is None:
            print(f"raw image 추측 실패 — overlay 없이 mask only 표시")
            from generate_image_seg_masks import ROI_Y_TOP, ROI_Y_BOT
            img_roi = np.full((ROI_Y_BOT - ROI_Y_TOP, 1280, 3), 64, dtype=np.uint8)
            img_full = np.full((720, 1280, 3), 64, dtype=np.uint8)
        else:
            from generate_image_seg_masks import ROI_Y_TOP, ROI_Y_BOT
            img_full = np.array(Image.open(img_path).convert("RGB"))
            img_roi = img_full[ROI_Y_TOP:ROI_Y_BOT]
            print(f"raw image found: {img_path}")
        out = out_dir / f"{fid}_existing.png"
        viz(img_full, img_roi, mask, fid + " (from npz)", out)
        print(f"saved {out}")
        return

    # 실 SAM 파이프라인 실행 — 모델 로드
    print("loading GroundingDINO + SAM2.1 base+ ...")
    from generate_image_seg_masks import (
        GD_CONFIG, GD_WEIGHT, SAM2_CONFIG, SAM2_WEIGHT)
    from groundingdino.util.inference import load_model
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    gd_model = load_model(GD_CONFIG, str(GD_WEIGHT), device="cuda")
    sam2_model = build_sam2(SAM2_CONFIG, str(SAM2_WEIGHT), device="cuda")
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    print("loaded.\n")

    # image list 결정
    images = []
    if args.image:
        images.append(Path(args.image))
    if args.image_dir:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            images.extend(sorted(Path(args.image_dir).glob(ext)))
        if args.limit:
            images = images[:args.limit]
    if not images:
        print("ERROR: --image 또는 --image-dir 지정 필요")
        sys.exit(1)

    print(f"processing {len(images)} images → {out_dir}")
    t0 = time.time()
    for i, img_path in enumerate(images):
        if not img_path.exists():
            print(f"  skip (not found): {img_path}")
            continue
        try:
            img_roi, mask, img_full = run_pipeline(img_path, gd_model, sam2_predictor)
        except Exception as e:
            print(f"  fail {img_path.name}: {e}")
            continue
        out = out_dir / f"{img_path.stem}_seg.png"
        viz(img_full, img_roi, mask, img_path.name, out)
        if args.save_mask:
            np.savez_compressed(out_dir / f"{img_path.stem}_mask.npz", mask=mask)
        dt = time.time() - t0
        rate = (i + 1) / max(dt, 1e-3)
        print(f"  [{i+1}/{len(images)}] {img_path.name} → {out.name}   rate={rate:.2f}/s")
    print(f"\ndone in {(time.time()-t0):.1f}s")


if __name__ == "__main__":
    main()

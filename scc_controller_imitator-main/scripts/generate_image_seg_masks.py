"""SCC controller imitator Stage A 용 image plane semantic mask 생성.

jiwon_e2e 의 GroundingDINO + SAM2 pipeline 차용. CAMERA_1 (front) 만 처리.
ROI y=[235, 485) crop 후 (5 class, 250, 1280) uint8 mask 저장.

Classes (jiwon_e2e 와 동일):
  0 background, 1 road (track surface + lane line), 2 vehicle (race car),
  3 wall, 4 ignore

실행 (ml_con env, GPU):
  ~/miniforge3/envs/ml_con/bin/python scripts/generate_image_seg_masks.py [--limit N]

Output: data/seg_masks_v1/{frame_id}.npz
"""
from __future__ import annotations
import os, sys, argparse, json, time
from pathlib import Path

# jiwon_e2e 의 SAM2 pipeline 경로 추가.
# 클러스터(SLURM)에서는 home prefix 가 /home/ailab 라 다르므로 env 로 override 가능.
#   E2E_ROOT  : jiwon_e2e/e2e_racing 루트 (gt_pipeline + GD/SAM2 weight 위치)
#   GD_CONFIG_PATH : GroundingDINO SwinB cfg (env 미지정시 ml_con site-packages 기본)
E2E_ROOT = Path(os.environ.get(
    "SCC_E2E_ROOT",
    "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing"))
sys.path.insert(0, str(E2E_ROOT))

import numpy as np
import torch
from PIL import Image

from src.data.gt_pipeline import (
    phrase_to_class_id, is_ego_self, per_cam_masks_to_class_layers,
    N_CLASSES, CLASS_NAMES,
)

# Config — v2: GroundingDINO SwinB + V2 ensemble (PROMPT_A + PROMPT_B)
# SwinT → SwinB swap: race-car detection recall 큰 폭 ↑ (검증: outputs_multi/sam_swinb/).
# 단 thr=0.15 는 false positive 폭주 → thr=0.25 (sweet spot, precision 0.97+ 회복).
GD_CONFIG = os.environ.get(
    "SCC_GD_CONFIG",
    "/home/ailab-12/miniforge3/envs/ml_con/lib/python3.11/site-packages/groundingdino/config/GroundingDINO_SwinB_cfg.py")
GD_WEIGHT = E2E_ROOT / "checkpoints/grounded_sam2/groundingdino_swinb_cogcoor.pth"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_WEIGHT = E2E_ROOT / "checkpoints/grounded_sam2/sam2.1_hiera_base_plus.pt"

# 2 prompt ensemble: A 는 모든 class, B 는 vehicle 보강만 (synonym).
# B 의 vehicle box 중 A 와 IoU>0.5 인 건 dedup.
PROMPT_A = "race car. track surface. lane line. wall. barrier."
PROMPT_B = "car. vehicle. truck."
PROMPT = PROMPT_A    # 하위 호환 (외부 import 시)
BOX_THRESHOLD = 0.25   # SwinT 0.20 → SwinB 0.25 (precision 회복)
TEXT_THRESHOLD = 0.22

# Our ROI (DESIGN.md §4.1)
ROI_Y_TOP, ROI_Y_BOT = 235, 485
ROI_H = ROI_Y_BOT - ROI_Y_TOP  # 250
RAW_IMG_BASE = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/data/raw/sangam_track/MultiAgnet"
META_JSON = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/multiagnet_compact_meta.json"
INDEX_JSON = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/data/processed/index.json"
OUT_DIR = Path("/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/seg_masks_v2")


def gd_transform(image_pil):
    import groundingdino.datasets.transforms as T
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_t, _ = transform(image_pil, None)
    return img_t


def boxes_to_xyxy(boxes_cxcywh, W, H):
    boxes_px = boxes_cxcywh * np.array([W, H, W, H])
    out = np.zeros_like(boxes_px)
    out[:, 0] = boxes_px[:, 0] - boxes_px[:, 2] / 2
    out[:, 1] = boxes_px[:, 1] - boxes_px[:, 3] / 2
    out[:, 2] = boxes_px[:, 0] + boxes_px[:, 2] / 2
    out[:, 3] = boxes_px[:, 1] + boxes_px[:, 3] / 2
    return out


def _iou(b1, b2):
    x0 = max(b1[0], b2[0]); y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2]); y1 = min(b1[3], b2[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def _gd_predict(gd_model, img_pil, prompt):
    from groundingdino.util.inference import predict
    img_t = gd_transform(img_pil)
    boxes, logits, phrases = predict(
        model=gd_model, image=img_t, caption=prompt,
        box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, device="cuda",
    )
    return boxes, logits, phrases


def process_frame(img_path, gd_model, sam2_predictor):
    """RGB image (720×1280) → ROI crop (250×1280) → GD ensemble (A+B) + SAM2
        → (N_CLASSES, 250, 1280) uint8.

    Crop 먼저 후 GD/SAM2 적용해야 jiwon_e2e MultiAgnet 의 알려진 문제 (GD 가
    race car 박스를 ego hood 와 한꺼번에 이미지 전체로 잡는 현상) 를 회피.

    v2 ensemble: PROMPT_A 모든 class + PROMPT_B vehicle synonym 보강.
    B 의 vehicle box 가 A 의 vehicle box 와 IoU>0.5 면 중복 제거.
    """
    img_pil_full = Image.open(img_path).convert("RGB")
    img_np_full = np.array(img_pil_full)
    # ROI crop
    img_np = img_np_full[ROI_Y_TOP:ROI_Y_BOT].copy()   # (250, 1280, 3)
    img_pil = Image.fromarray(img_np)
    H, W = img_np.shape[:2]   # 250, 1280

    # ----- 두 prompt 로 GD 추론 -----
    boxes_A, _, phrases_A = _gd_predict(gd_model, img_pil, PROMPT_A)
    boxes_B, _, phrases_B = _gd_predict(gd_model, img_pil, PROMPT_B)

    xyA = boxes_to_xyxy(boxes_A.cpu().numpy(), W, H) if len(boxes_A) else np.zeros((0, 4))
    xyB = boxes_to_xyxy(boxes_B.cpu().numpy(), W, H) if len(boxes_B) else np.zeros((0, 4))
    classes_A = [phrase_to_class_id(p) for p in phrases_A]
    classes_B = [phrase_to_class_id(p) for p in phrases_B]

    # B 는 vehicle (c=2) 만 keep, A 와 IoU>0.5 면 dedup
    keep_B = [i for i, c in enumerate(classes_B) if c == 2]
    dedup_B = []
    for j in keep_B:
        is_dup = any(classes_A[i] == 2 and _iou(xyB[j], xyA[i]) > 0.5
                       for i in range(len(xyA)))
        if not is_dup:
            dedup_B.append(j)
    all_boxes = list(xyA) + [xyB[j] for j in dedup_B]
    all_classes = list(classes_A) + [2] * len(dedup_B)

    if len(all_boxes) == 0:
        out = np.zeros((N_CLASSES, H, W), dtype=np.uint8)
        out[0] = 1
        return out

    boxes_xyxy = np.stack(all_boxes)
    class_ids_raw = all_classes
    # class 4 (ignore phrase) 제거. vehicle (c=2) 의 over-extended box (ROI 가로
    # 85% 이상) 는 GroundingDINO 의 알려진 MultiAgnet 결함 — 여러 차를 한 박스로 묶음.
    def keep_box(b, c):
        if c == 4:
            return False
        bw = b[2] - b[0]
        if c == 2 and bw > 0.85 * W:
            return False
        return True
    keep = [(b, c) for b, c in zip(boxes_xyxy, class_ids_raw) if keep_box(b, c)]
    if not keep:
        out = np.zeros((N_CLASSES, H, W), dtype=np.uint8)
        out[0] = 1
        return out
    kept_boxes = np.stack([b for b, _ in keep])
    kept_classes = [c for _, c in keep]

    sam2_predictor.set_image(img_np)
    masks, _, _ = sam2_predictor.predict(
        point_coords=None, point_labels=None,
        box=kept_boxes, multimask_output=False,
    )
    if masks.ndim == 4:
        masks = masks[:, 0]
    masks_bool = masks.astype(bool)

    # vehicle 의 SAM2 mask 도 후처리: mask coverage 가 ROI 면적 50% 이상이면
    # GD box 가 정상 (bw<0.85W) 이어도 SAM2 가 background 까지 잡은 케이스.
    final_masks = []
    final_classes = []
    for m, c in zip(masks_bool, kept_classes):
        if c == 2 and m.mean() > 0.50:
            continue
        final_masks.append(m)
        final_classes.append(c)
    if not final_masks:
        out = np.zeros((N_CLASSES, H, W), dtype=np.uint8)
        out[0] = 1
        return out

    return per_cam_masks_to_class_layers(
        np.stack(final_masks), final_classes, (H, W), bottom_strip_classes=()
    )


def get_image_path(meta, run_dt_map, raw_img_base):
    """meta → CAMERA_1 image path.
    meta 에 track/policy 가 있으면 raw_img_base 가 root 인지 leaf 인지 자동 감지하여 경로 구성.
    """
    run = meta["run_id"]
    tick = int(meta["tick"])
    track = meta.get("track"); policy = meta.get("policy")
    if track and policy:
        # raw_img_base 가 이미 track/policy 까지인지 자동 감지
        if raw_img_base.endswith(f"{track}/{policy}") or raw_img_base.endswith(f"{track}/{policy}/"):
            return f"{raw_img_base}/{run}/CAMERA_1/{tick}_Intensity.jpeg"
        return f"{raw_img_base}/{track}/{policy}/{run}/CAMERA_1/{tick}_Intensity.jpeg"
    return f"{raw_img_base}/{run}/CAMERA_1/{tick}_Intensity.jpeg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=999999)
    ap.add_argument("--every", type=int, default=1,
                    help="N 번째 frame 만 처리 (subsample)")
    ap.add_argument("--filter", choices=["all", "lead-frames-only"], default="all",
                    help="lead-frames-only: K=100 합성에 사용된 lead frame 만")
    ap.add_argument("--frame-indices-file", default=None,
                    help="처리할 frame index list 파일 (1 줄 1 i). 지정시 --filter 무시")
    ap.add_argument("--meta", default=META_JSON, help="meta json path (frame_id, track, policy, run_id, tick)")
    ap.add_argument("--raw-img-base", default=RAW_IMG_BASE,
                    help="<raw>/<track>/<policy> 까지의 경로 (run 직전)")
    ap.add_argument("--out-dir",      default=str(OUT_DIR), type=Path, help="output dir override")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading meta {args.meta} ...")
    meta_list = json.loads(open(args.meta).read())
    if os.path.exists(INDEX_JSON):
        idx_data = json.loads(open(INDEX_JSON).read())
        run_dt = {e["run_id"]: float(e.get("dt_per_tick_ms", 10.0)) for e in idx_data}
    else:
        run_dt = {}

    if args.frame_indices_file:
        with open(args.frame_indices_file) as f:
            wanted = set(int(line.strip()) for line in f if line.strip())
        meta_list = [m for m in meta_list if m["i"] in wanted]
        print(f"  filtered to {len(meta_list)} frames from --frame-indices-file")
    elif args.filter == "lead-frames-only":
        # 우선순위: lead frame (synth pipeline 일치). 효율 ↑
        lead_idx_file = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_k100_v1/lead_frame_indices.txt"
        if os.path.exists(lead_idx_file):
            with open(lead_idx_file) as f:
                wanted = set(int(line.strip()) for line in f if line.strip())
            meta_list = [m for m in meta_list if m["i"] in wanted]
            print(f"  filtered to {len(meta_list)} lead frames")

    # 'i' 필드 없는 경우 (신규 meta) idx 부여
    for i, m in enumerate(meta_list):
        if "i" not in m: m["i"] = i
    meta_list = [m for m in meta_list if args.start <= m["i"] < args.end]
    if args.every > 1:
        meta_list = [m for i, m in enumerate(meta_list) if i % args.every == 0]
        print(f"  every-{args.every} subsample → {len(meta_list)}")
    if args.limit:
        meta_list = meta_list[:args.limit]
    print(f"  processing {len(meta_list)} frames")

    print("loading GroundingDINO + SAM2.1 base+ ...")
    from groundingdino.util.inference import load_model
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    gd_model = load_model(GD_CONFIG, str(GD_WEIGHT), device="cuda")
    sam2_model = build_sam2(SAM2_CONFIG, str(SAM2_WEIGHT), device="cuda")
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    print("loaded.\n")

    t0 = time.time()
    done = 0; skipped = 0; failed = 0
    for m in meta_list:
        out_path = out_dir / f"{m['frame_id']}.npz"
        if out_path.exists():
            skipped += 1
            continue
        img_path = get_image_path(m, run_dt, args.raw_img_base)
        if not os.path.exists(img_path):
            failed += 1
            continue
        try:
            mask = process_frame(img_path, gd_model, sam2_predictor)
            np.savez_compressed(out_path, mask=mask)
            done += 1
        except Exception as e:
            print(f"  fail {m['frame_id']}: {e}")
            failed += 1
        if (done + skipped + failed) % 20 == 0:
            elapsed = time.time() - t0
            rate = (done + skipped) / max(elapsed, 1e-3)
            print(f"  [{done + skipped + failed}/{len(meta_list)}] rate={rate:.1f}/s done={done} skip={skipped} fail={failed}")

    print(f"\ntotal: done={done}, skipped={skipped}, failed={failed}")
    print(f"time: {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()

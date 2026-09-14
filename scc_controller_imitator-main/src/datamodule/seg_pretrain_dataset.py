"""Image encoder seg-pretraining dataset.

Pairs:
  - input  : ROI image (3, 250, 1280) ImageNet-normed float32
  - target : label map (250, 1280) int64 (0..3 = bg/road/vehicle/wall; ignore = 4 → -100)

Sources:
  - SAM masks  : data/seg_masks_v2/{frame_id}.npz  ['mask'] shape (5, 250, 1280) uint8
  - raw images : RAW/{run_id}/CAMERA_1/{tick}_Intensity.jpeg  (720, 1280, 3) RGB

multiagnet_compact_meta.json 의 frame_id 와 run_id/tick 으로 양쪽 path 매칭.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

ROI_Y0, ROI_Y1 = 235, 485
ROI_W = 1280
IMG_H, IMG_W = 720, 1280

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_RAW = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/data/raw"
DEFAULT_MASK = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/seg_masks_v2"
DEFAULT_META = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/multiagnet_compact_meta.json"

# class ids 매핑: bg, road, vehicle, wall, ignore
# train 시 ignore_index=-100 으로 ignore channel 픽셀은 loss 제외 (현재는 항상 0)
N_CLASSES = 4
IGNORE_IDX = -100


def _load_image_roi(path: str) -> np.ndarray:
    """720x1280 RGB → 250x1280 ROI np.uint8."""
    img = np.array(Image.open(path).convert("RGB"))
    if img.shape != (IMG_H, IMG_W, 3):
        raise RuntimeError(f"unexpected image shape {img.shape} at {path}")
    return img[ROI_Y0:ROI_Y1].copy()


def _img_to_tensor(roi_u8: np.ndarray) -> torch.Tensor:
    arr = roi_u8.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())  # (3, H, W)


def _mask_to_label(mask: np.ndarray) -> np.ndarray:
    """(5, H, W) uint8 → (H, W) int64.
    각 픽셀의 최대 활성 채널을 class label 로. ignore 채널 (idx 4) 은 -100.
    아무 채널도 활성 안 된 픽셀은 bg(0) 로.
    """
    # bg/road/vehicle/wall 만 사용 (ch 0..3)
    m4 = mask[:N_CLASSES]   # (4, H, W)
    any_active = m4.any(axis=0)
    label = np.argmax(m4, axis=0).astype(np.int64)
    label[~any_active] = 0   # 그냥 bg 처리

    if mask.shape[0] > N_CLASSES:
        ignore = mask[N_CLASSES] > 0
        label[ignore] = IGNORE_IDX
    return label


class SegPretrainDataset(Dataset):
    """SAM mask supervision 용 image encoder pretraining dataset."""

    def __init__(self,
                 meta_path: str = DEFAULT_META,
                 mask_dir: str = DEFAULT_MASK,
                 raw_base: str = DEFAULT_RAW,
                 frame_ids: List[str] | None = None):
        meta = json.load(open(meta_path))
        mask_dir_p = Path(mask_dir)
        raw_base_p = Path(raw_base)

        self.samples: List[Tuple[str, str]] = []   # (img_path, mask_path)
        wanted = set(frame_ids) if frame_ids is not None else None
        for m in meta:
            fid = m["frame_id"]
            if wanted is not None and fid not in wanted: continue
            mp = mask_dir_p / f"{fid}.npz"
            if not mp.exists(): continue
            run = m["run_id"]; tick = int(m["tick"])
            # frame_id = "{track}__{policy}__{run_id}__t{tick:06d}"  (mask 측은 t6)
            track = m["track"]; policy = m["policy"]
            ip = raw_base_p / track / policy / run / "CAMERA_1" / f"{tick}_Intensity.jpeg"
            if not ip.exists(): continue
            self.samples.append((str(ip), str(mp)))

        if not self.samples:
            raise RuntimeError("no samples matched (image + mask both present)")

    def __len__(self): return len(self.samples)

    def __getitem__(self, i: int):
        img_path, mask_path = self.samples[i]
        roi = _load_image_roi(img_path)
        img = _img_to_tensor(roi)
        mask = np.load(mask_path)["mask"]
        label = _mask_to_label(mask)
        return {
            "image": img,
            "label": torch.from_numpy(label),
        }


def collate_seg(batch):
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
    }

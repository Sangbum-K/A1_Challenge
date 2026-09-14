"""SCC controller imitator dataset.

Loads `fwd50_*.pkl` (or any synth prefix) and yields per-sample dict:
- image_curr      (3, 250, 1280) float32 ImageNet-normed
- image_prev_rot  (3, 250, 1280) float32 — Δyaw 회전 보정
- ego_5d          (5,)   [vx, vy, ax, ay, yaw_rate]
- traj_30         (30, 4)  [x, y, yaw, v] ego-local
- target          (30, 2)  [steer/40, accel/m_s2]  (controller_imitator 컨벤션 호환)
- lead_info       (6,)
- lead_valid      bool
- frame_id        str
- traj_tag        str (logged | rand_*)

ROI: y∈[235,485), x∈[0,1280) → (250, 1280)
"""
from __future__ import annotations
import os, glob, pickle, hashlib
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

ROI_Y_TOP, ROI_Y_BOT = 235, 485
ROI_H = ROI_Y_BOT - ROI_Y_TOP   # 250
ROI_W = 1280
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# CAMERA_1 fx (= fy). 작은 Δyaw 에서 K·R_y(θ)·K⁻¹ ≈ horizontal pixel shift of fx·θ.
# 본 데이터의 |d_yaw| < 0.06 rad (max 3.4°) — perspective 효과는 edge 에서 ≤2 px → 단순
# shift 로 충분. warp 검증: outputs_multi/warp_sanity_zoom.png 참조.
CAM_FX = 320.0

# Steering loss-weight convention (controller_imitator)
STEER_DIV = 40.0


def _yaw_shift(img_np: np.ndarray, d_yaw: float, fx: float = CAM_FX) -> np.ndarray:
    """ego 의 d_yaw rad 회전을 image-plane horizontal shift 로 근사.

    shift_px = round(fx · d_yaw). 양수면 우측으로 (열 인덱스 증가 방향), 음수면 좌측으로.
    경계는 edge column replication (BORDER_REPLICATE 와 동일 거동).

    근거: pure-yaw rotation 의 homography K·R_y(θ)·K⁻¹ 는 작은 θ 에서 u' ≈ u + fx·θ
    (모든 픽셀 균일 수평 이동). 본 데이터의 |θ| ≤ 0.06 rad 에서 edge 픽셀과의 차이는 ≤2 px.
    """
    if abs(d_yaw) < 1e-5:
        return img_np
    shift = int(round(fx * d_yaw))
    if shift == 0:
        return img_np
    out = np.empty_like(img_np)
    if shift > 0:
        out[:, shift:] = img_np[:, :-shift]
        out[:, :shift] = img_np[:, :1]
    else:
        s = -shift
        out[:, :-s] = img_np[:, s:]
        out[:, -s:] = img_np[:, -1:]
    return out


def _load_image(path: str) -> np.ndarray:
    """Read full-frame jpeg as (720, 1280, 3) uint8 RGB. ROI crop is applied later in __getitem__."""
    img = np.array(Image.open(path).convert("RGB"))
    assert img.shape == (720, 1280, 3), f"unexpected image shape {img.shape} at {path}"
    return img


def _to_imagenet_tensor(roi: np.ndarray) -> torch.Tensor:
    """uint8 (H, W, 3) → float32 (3, H, W) ImageNet normalized."""
    t = roi.astype(np.float32) / 255.0
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(t.transpose(2, 0, 1))


def _pad_traj(arr: np.ndarray, n: int) -> np.ndarray:
    """Pad/truncate traj along first axis to length n. Pad with last row repeated.

    Truncates silently if arr 가 더 김 (예: synth pkl 이 40 이상). 향후 schema 변경 시
    assertion 추가 고려.
    """
    if arr.shape[0] >= n:
        return arr[:n]
    pad = np.tile(arr[-1:], (n - arr.shape[0], 1))
    return np.concatenate([arr, pad], axis=0)


def _make_mask(n_real: int, n_total: int) -> np.ndarray:
    """Boolean mask: True for first n_real entries, False for the rest."""
    m = np.zeros(n_total, dtype=np.bool_)
    m[:min(n_real, n_total)] = True
    return m


def _scan_pkls(data_dir: str, prefixes: Tuple[str, ...], min_size: int = 1000) -> List[Tuple[str, int]]:
    """Return list of (pkl_path, sample_idx) for all non-empty pkls.
    Scans pickles once at init for fast random access."""
    out = []
    for pre in prefixes:
        for fp in sorted(glob.glob(os.path.join(data_dir, f"{pre}_*.pkl"))):
            if os.path.getsize(fp) < min_size:
                continue
            try:
                d = pickle.load(open(fp, "rb"))
                n = len(d["samples"])
                for i in range(n):
                    out.append((fp, i))
            except Exception:
                continue
    return out


class SCCSynthDataset(Dataset):
    """Reads samples from `data/scc_synth_k100_v1/{prefix}_*.pkl`.

    Splits across pkls: deterministic by frame_id hash (train/val).
    """

    def __init__(self,
                 data_dir: str,
                 prefixes: Tuple[str, ...] = ("fwd50",),
                 split: str = "train",   # train | val | all
                 val_ratio: float = 0.1,
                 split_seed: int = 0,
                 cache_in_ram: bool = True):
        super().__init__()
        self.data_dir = data_dir
        self.prefixes = prefixes
        self.cache_in_ram = cache_in_ram
        idx = _scan_pkls(data_dir, prefixes)
        if not idx:
            raise FileNotFoundError(f"no non-empty pkls in {data_dir} with prefixes {prefixes}")

        # split deterministically by frame_id hash
        self._all_idx = idx
        if cache_in_ram:
            # group by pkl path to load once
            self._pkl_cache: Dict[str, dict] = {}

        keep = []
        for (fp, i) in idx:
            s = self._get_sample(fp, i)
            fid = s["frame_id"]
            h = int(hashlib.sha1((fid + str(split_seed)).encode()).hexdigest()[:8], 16)
            is_val = (h % 1000) / 1000.0 < val_ratio
            if split == "val" and is_val:
                keep.append((fp, i))
            elif split == "train" and not is_val:
                keep.append((fp, i))
            elif split == "all":
                keep.append((fp, i))
        self.idx = keep
        print(f"SCCSynthDataset {split}: {len(keep)}/{len(idx)} samples "
               f"({len(set(self._get_sample(*k)['frame_id'] for k in keep))} frames)")

    def _get_sample(self, fp: str, i: int) -> dict:
        if self.cache_in_ram:
            if fp not in self._pkl_cache:
                self._pkl_cache[fp] = pickle.load(open(fp, "rb"))
            return self._pkl_cache[fp]["samples"][i]
        else:
            d = pickle.load(open(fp, "rb"))
            return d["samples"][i]

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, k: int) -> Dict[str, torch.Tensor]:
        fp, i = self.idx[k]
        s = self._get_sample(fp, i)

        # images
        img_curr = _load_image(s["image_curr_path"])
        img_prev = _load_image(s["image_prev_path"])
        # Δyaw 회전 보정: ego_5d[4] = yaw_rate (rad/s), Δt = 0.1 s
        ego_5d_arr = np.asarray(s["ego_5d"], dtype=np.float32)
        d_yaw = float(ego_5d_arr[4]) * 0.1
        img_prev_rot = _yaw_shift(img_prev, -d_yaw)     # back-warp prev to curr frame (horiz shift)
        # ROI crop
        img_curr_roi = img_curr[ROI_Y_TOP:ROI_Y_BOT]
        img_prev_roi = img_prev_rot[ROI_Y_TOP:ROI_Y_BOT]

        out = {
            "image_curr": _to_imagenet_tensor(img_curr_roi),
            "image_prev_rot": _to_imagenet_tensor(img_prev_roi),
            "ego_5d": torch.from_numpy(ego_5d_arr),
            # controller_imitator pretrained 호환 (#654): seq_length=40, traj_dim=3 (x,y,v)
            # synth pkl 의 traj_30 는 logged=30, random=40. logged 는 0-pad to 40.
            "traj_30": torch.from_numpy(_pad_traj(np.asarray(s["traj_30"], dtype=np.float32), 40)),
            "lead_info": torch.from_numpy(np.asarray(s["lead_info"], dtype=np.float32)),
            "lead_valid": torch.tensor(bool(s["lead_valid"])),
            # targets: stack steer (deg→/40) and accel; pad to length 40
            "target": torch.from_numpy(_pad_traj(np.stack([
                np.asarray(s["gt_steer_30"], dtype=np.float32) / STEER_DIV,
                np.asarray(s["gt_accel_30"], dtype=np.float32),
            ], axis=1), 40)),   # (40, 2)
            # target_mask: True for real timesteps, False for padded (loss 에서 masked-mean)
            "target_mask": torch.from_numpy(_make_mask(len(s["gt_steer_30"]), 40)),  # (40,)
            "frame_id": s["frame_id"],
            "traj_tag": s["traj_tag"],
        }
        return out


def collate_scc(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    out: Dict = {}
    for k in batch[0].keys():
        if k in ("frame_id", "traj_tag"):
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out

"""i_1 + i_2 + a_2 학습용 dataset.

각 sample:
  image_curr     (3, 250, 1280)   ROI ImageNet-normed
  image_prev_rot (3, 250, 1280)   prev-tick image, ego Δyaw 만큼 horizontal shift
  ego_motion     (2,)             [vx, wz]  (ego forward m/s, yaw_rate rad/s)
  target         (40, 7)          a_2 GT  [obj, vL_abs, vY_abs, vL_rel, vY_rel, prox, dist]
  obj_mask       (40,)            target[:, 0] > 0.5 — 회귀 채널 loss mask

prev-tick 결정: curr_tick - PREV_DTICK (기본 5 = ~50ms before).
prev frame 의 image / ego 가 모두 있어야 sample 포함.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

DEG2RAD = np.pi / 180.0
ROI_Y0, ROI_Y1 = 235, 485
IMG_H, IMG_W = 720, 1280
PREV_DTICK = 20      # file_step = 5 tick → 20 tick = 200ms (고정, multi-dt 비활성)

CAM_FX = 320.0

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_RAW = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/data/raw"
DEFAULT_META = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/multiagnet_compact_meta.json"
DEFAULT_A2   = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/aux_targets_fx320"


def _yaw_shift(img: np.ndarray, d_yaw: float, fx: float = CAM_FX) -> np.ndarray:
    """small Δyaw → uniform horizontal pixel shift fx*θ."""
    if abs(d_yaw) < 1e-5: return img
    shift = int(round(fx * d_yaw))
    if shift == 0: return img
    out = np.empty_like(img)
    if shift > 0:
        out[:, shift:] = img[:, :-shift]; out[:, :shift] = img[:, :1]
    else:
        s = -shift; out[:, :-s] = img[:, s:]; out[:, -s:] = img[:, -1:]
    return out


def _load_roi(path: Path) -> np.ndarray:
    img = np.array(Image.open(path).convert("RGB"))
    if img.shape != (IMG_H, IMG_W, 3):
        raise RuntimeError(f"shape {img.shape} at {path}")
    return img[ROI_Y0:ROI_Y1].copy()


def _img_to_tensor(roi: np.ndarray) -> torch.Tensor:
    arr = roi.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def _parse_ego(ego_file: Path):
    if not ego_file.exists(): return None
    out = {}
    for line in ego_file.read_text().splitlines():
        if ":" not in line: continue
        k, _, v = line.partition(":")
        k = k.strip(); v = v.strip()
        if k in ("position", "orientation", "velocity", "angularVelocity"):
            try: nums = [float(t) for t in v.split()]
            except: continue
            if len(nums) >= 3: out[k] = nums
    for req in ("position", "orientation", "velocity", "angularVelocity"):
        if req not in out: return None
    return out


class A2PairDataset(Dataset):
    """fx-aware (per-sample) + multi-dt sampling + close-lead 가중 지원.

    prev_dtick: int 또는 list (예: [5, 20, 50] = 50/200/500ms). list 시 __getitem__ 마다 random sample.
    fx: a_2 config.json 의 fx (per-dataset). per-sample 출력 → pixel warp 자동 매칭.
    """

    def __init__(self,
                 meta_path: str = DEFAULT_META,
                 a2_dir: str = DEFAULT_A2,
                 raw_base: str = DEFAULT_RAW,
                 prev_dtick = PREV_DTICK,            # int 또는 list[int]
                 require_obj: bool = False,
                 close_lead_meta_path: str = None):
        self.prev_dtick_options = list(prev_dtick) if isinstance(prev_dtick, (list, tuple)) else [int(prev_dtick)]
        self.max_dtick = max(self.prev_dtick_options)

        meta = json.load(open(meta_path))
        a2_p = Path(a2_dir); raw_p = Path(raw_base)
        # a_2 config — fx 자동 추출
        cfg_p = a2_p / "config.json"
        self.cfg = json.load(open(cfg_p)) if cfg_p.exists() else {}
        self.fx = float(self.cfg.get("fx", CAM_FX))

        # NFS stat 가 매우 느려서 (한 sample 마다 4 stat) parallel 로 batch
        from concurrent.futures import ThreadPoolExecutor

        def _check(m):
            fid = m["frame_id"]; tick = int(m["tick"])
            run = m["run_id"]; track = m["track"]; policy = m["policy"]
            ap = a2_p / f"{fid}.npz"
            if not ap.exists(): return None
            if require_obj:
                try:
                    t = np.load(ap)["target"]
                    if t[:, 0].sum() == 0: return None
                except Exception: return None
            run_dir = raw_p / track / policy / run
            ci = run_dir / "CAMERA_1" / f"{tick}_Intensity.jpeg"
            ce = run_dir / "EGO_INFO" / f"{tick}.txt"
            ptick_max = tick - self.max_dtick
            pi_max = run_dir / "CAMERA_1" / f"{ptick_max}_Intensity.jpeg"
            pe_max = run_dir / "EGO_INFO" / f"{ptick_max}.txt"
            if not (ci.exists() and ce.exists() and pi_max.exists() and pe_max.exists()):
                return None
            return {"fid": fid, "tick": tick, "ci": ci, "ce": ce, "ap": ap, "run_dir": run_dir}

        with ThreadPoolExecutor(max_workers=32) as ex:
            results = list(ex.map(_check, meta, chunksize=16))
        self.samples = [r for r in results if r is not None]
        if not self.samples:
            raise RuntimeError("no a_2 pair samples (image_curr + image_prev + a2 npz)")

        # close-lead: optional per-sample min_dist 로딩 (frame_id 기반)
        self.min_dist = np.full(len(self.samples), float("inf"), dtype=np.float32)
        if close_lead_meta_path:
            zm = np.load(close_lead_meta_path)
            cl_keys = zm["keys"].tolist(); cl_dists = zm["dists"]
            # close_lead_meta 의 key 는 "run__t{7d}" — a_2 의 frame_id 끝 부분과 매칭
            cl_map = {k: d for k, d in zip(cl_keys, cl_dists)}
            n_hit = 0
            for i, s in enumerate(self.samples):
                # frame_id "track__policy__run__t{6d}" → run__t{7d}
                parts = s["fid"].split("__")
                if len(parts) >= 4 and parts[-1].startswith("t"):
                    try: tk = int(parts[-1][1:])
                    except ValueError: tk = None
                    if tk is not None:
                        key = f"{parts[-2]}__t{tk:07d}"
                        if key in cl_map:
                            self.min_dist[i] = cl_map[key]; n_hit += 1
            print(f"[a2-dataset] close-lead matched {n_hit}/{len(self.samples)} (fx={self.fx})")

    def __len__(self): return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        # multi-dt: 매 호출마다 random 선택
        dtick = self.prev_dtick_options[np.random.randint(len(self.prev_dtick_options))] \
                if len(self.prev_dtick_options) > 1 else self.prev_dtick_options[0]
        ptick = s["tick"] - dtick
        pi = s["run_dir"] / "CAMERA_1" / f"{ptick}_Intensity.jpeg"
        pe = s["run_dir"] / "EGO_INFO" / f"{ptick}.txt"
        # fallback: 못찾으면 smallest dtick
        if not pi.exists() or not pe.exists():
            dtick = self.prev_dtick_options[0]
            ptick = s["tick"] - dtick
            pi = s["run_dir"] / "CAMERA_1" / f"{ptick}_Intensity.jpeg"
            pe = s["run_dir"] / "EGO_INFO" / f"{ptick}.txt"

        ego_c = _parse_ego(s["ce"]); ego_p = _parse_ego(pe)
        if ego_c is None or ego_p is None:
            ego_c = ego_p or ego_c
            ego_p = ego_c
        d_yaw = (ego_c["orientation"][2] - ego_p["orientation"][2]) * DEG2RAD
        roi_c = _load_roi(s["ci"])
        roi_p = _yaw_shift(_load_roi(pi), d_yaw, fx=self.fx)

        ego_motion = np.array([
            ego_c["velocity"][0],
            ego_c["angularVelocity"][2] * DEG2RAD,
        ], dtype=np.float32)

        target = np.load(s["ap"])["target"].astype(np.float32)
        obj_mask = target[:, 0] > 0.5

        return {
            "image_curr":     _img_to_tensor(roi_c),
            "image_prev_rot": _img_to_tensor(roi_p),
            "ego_motion":     torch.from_numpy(ego_motion),
            "target":         torch.from_numpy(target),
            "obj_mask":       torch.from_numpy(obj_mask.astype(np.uint8)),
            "frame_id":       s["fid"],
            "fx":             torch.tensor(self.fx, dtype=torch.float32),
            "prev_dtick":     torch.tensor(dtick, dtype=torch.long),
            "min_dist":       torch.tensor(float(self.min_dist[i]), dtype=torch.float32),
        }


def collate_a2(batch):
    out = {
        "image_curr":     torch.stack([b["image_curr"] for b in batch], dim=0),
        "image_prev_rot": torch.stack([b["image_prev_rot"] for b in batch], dim=0),
        "ego_motion":     torch.stack([b["ego_motion"] for b in batch], dim=0),
        "target":         torch.stack([b["target"] for b in batch], dim=0),
        "obj_mask":       torch.stack([b["obj_mask"] for b in batch], dim=0).bool(),
        "frame_id":       [b["frame_id"] for b in batch],
    }
    if "fx" in batch[0]:
        out["fx"]         = torch.stack([b["fx"] for b in batch], dim=0)
        out["prev_dtick"] = torch.stack([b["prev_dtick"] for b in batch], dim=0)
        out["min_dist"]   = torch.stack([b["min_dist"] for b in batch], dim=0)
    return out

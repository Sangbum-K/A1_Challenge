"""f_1 학습용 dataset — SCC synth 결과 (logged + random) 통합 reader.

각 sample = 1 .scc.txt 파일 (해당 frame 의 1 trajectory variant 의 SCC 처리 결과):
  image_curr     (3, 250, 1280)   ROI ImageNet-normed
  image_prev_rot (3, 250, 1280)   prev tick image, Δyaw shift
  ego_5d         (5,)             [vx, vy, ax, ay, yaw_rate]
  traj_40        (40, 3)          INPUT trajectory (ego frame, [x, y, speed])
                                   logged: scc_logged_bank/{key}.bin
                                   random: scc_traj_bank/{key}.trajs.bin row k
  target_40      (40, 2)          long_traj 의 처음 40 row [steer (deg), ax (m/s²)]
  target_mask    (40,)            valid (long_traj 가 40 보다 짧으면 뒤 padded)

NOTE: traj_40 / target 모두 world→ego transform 후 dataset 에 노출.
"""
from __future__ import annotations
import json, struct
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

DEG2RAD = np.pi / 180.0
ROI_Y0, ROI_Y1 = 235, 485
IMG_H, IMG_W = 720, 1280
PREV_DTICK = 20     # file_step = 20 tick → ~200ms prev (4x training base 50ms)
CAM_FX = 320.0
TRAJ_N = 40    # input traj length
TARG_N = 40    # output target length

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_RAW = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/data/raw"
DEFAULT_META = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/multiagnet_compact_meta.json"
DEFAULT_INDEX = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/data/processed/index.json"
DEFAULT_SCC  = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_v2"
DEFAULT_RAND_BANK = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_traj_bank"
DEFAULT_LOG_BANK  = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_logged_bank"


def _yaw_shift(img: np.ndarray, d_yaw: float, fx: float = CAM_FX) -> np.ndarray:
    if abs(d_yaw) < 1e-5: return img
    shift = int(round(fx * d_yaw))
    if shift == 0: return img
    out = np.empty_like(img)
    if shift > 0:
        out[:, shift:] = img[:, :-shift]; out[:, :shift] = img[:, :1]
    else:
        s = -shift; out[:, :-s] = img[:, s:]; out[:, -s:] = img[:, -1:]
    return out


def _load_roi(p: Path) -> np.ndarray:
    img = np.array(Image.open(p).convert("RGB"))
    return img[ROI_Y0:ROI_Y1].copy()


def _img_to_tensor(roi: np.ndarray) -> torch.Tensor:
    arr = roi.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def _parse_ego(p: Path):
    if not p.exists(): return None
    out = {}
    for line in p.read_text().splitlines():
        if ":" not in line: continue
        k, _, v = line.partition(":")
        k = k.strip(); v = v.strip()
        if k in ("position","orientation","velocity","angularVelocity","acceleration"):
            try: nums = [float(t) for t in v.split()]
            except: continue
            if len(nums) >= 3: out[k] = nums
    for req in ("position","orientation","velocity","angularVelocity","acceleration"):
        if req not in out: return None
    return out


def _parse_scc_txt(p: Path):
    """{run}__t{tick}__{tag}.scc.txt → dict(long_traj=(M,8), lat_traj=(M,5), tag, in_lane_ids).
    long_traj cols: i time vx ax steer trq accel_pedal brake_pedal   ← steer 는 항상 ~0 (longitudinal 미사용)
    lat_traj  cols: i time vx ax steer                                ← steer 가 진짜 lateral cmd (rad)
    """
    out = {"long_traj": [], "lat_traj": [], "tag": "logged", "in_lane_ids": []}
    cur = None
    for line in p.read_text().splitlines():
        line = line.rstrip()
        if not line: continue
        if line.startswith("# run="):
            for kv in line.lstrip("# ").split():
                if kv.startswith("tag="): out["tag"] = kv.split("=",1)[1]
            continue
        if line.startswith("#"): continue
        if line.startswith("in_lane_obj_ids:"):
            out["in_lane_ids"] = [int(x) for x in line.split(":",1)[1].split()]
            continue
        if "=" in line and not line[0].isdigit():
            k, v = line.split("=", 1)
            if k.strip() == "long_traj_N": cur = "long_traj"; continue
            if k.strip() == "lat_traj_N":  cur = "lat_traj"; continue
            if k.strip() == "N":           cur = None; continue
            continue
        try:
            nums = [float(x) for x in line.split()]
        except: continue
        if cur in ("long_traj", "lat_traj"): out[cur].append(nums)
    out["long_traj"] = np.array(out["long_traj"], dtype=np.float32) if out["long_traj"] else np.zeros((0,8), np.float32)
    out["lat_traj"]  = np.array(out["lat_traj"],  dtype=np.float32) if out["lat_traj"]  else np.zeros((0,5), np.float32)
    return out


def _load_rand_traj(bank_path: Path, k: int):
    """scc_traj_bank/{key}.trajs.bin (header: int32 K_rand, int32 N_wp; data: float32[K*N*4])."""
    with open(bank_path, "rb") as f:
        h = f.read(8); K_rand, N_wp = struct.unpack("ii", h)
        if k < 0 or k >= K_rand: return None
        f.seek(8 + k * N_wp * 4 * 4)
        d = f.read(N_wp * 4 * 4)
    arr = np.frombuffer(d, dtype=np.float32).reshape(N_wp, 4)
    return arr   # [x, y, yaw, speed] world frame


def _load_logged_traj(bank_path: Path):
    """scc_logged_bank/{key}.bin (header: int32 N_wp; data: float32[N*4])."""
    with open(bank_path, "rb") as f:
        h = f.read(4); (N_wp,) = struct.unpack("i", h)
        d = f.read(N_wp * 4 * 4)
    return np.frombuffer(d, dtype=np.float32).reshape(N_wp, 4)


def _world_to_ego_traj(traj_world: np.ndarray, ex: float, ey: float, e_yaw: float):
    """(N, 4) world [x,y,yaw,speed] → (N, 3) ego [x_fwd, y_left, speed]."""
    c, s = np.cos(-e_yaw), np.sin(-e_yaw)
    dx = traj_world[:, 0] - ex; dy = traj_world[:, 1] - ey
    x_e = c * dx - s * dy
    y_e = s * dx + c * dy
    return np.stack([x_e, y_e, traj_world[:, 3]], axis=1).astype(np.float32)


DEFAULT_MANIFEST = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_pair_manifest.json"


def build_manifest(meta_path=DEFAULT_META, scc_dir=DEFAULT_SCC, raw_base=DEFAULT_RAW,
                    rand_bank=DEFAULT_RAND_BANK, log_bank=DEFAULT_LOG_BANK,
                    out_path=DEFAULT_MANIFEST, require_logged_bank=True):
    """Pre-scan .scc.txt + bank + meta → manifest json. file path 만 저장 (.exists() 호출 안함).

    실제 file existence 는 학습 시 getitem 에서 처리 — manifest 빌드 시는 trust.
    """
    print(f"[manifest] scanning {scc_dir} ...")
    scc_p = Path(scc_dir); raw_p = Path(raw_base)
    rand_p = Path(rand_bank); log_p = Path(log_bank)

    # bank file 목록 set 화 (memory 빠른 lookup)
    log_keys = {p.stem for p in log_p.glob("*.bin")}
    rand_keys = {p.stem.replace(".trajs","") for p in rand_p.glob("*.trajs.bin")}
    print(f"  log_bank keys: {len(log_keys)}, rand_bank keys: {len(rand_keys)}")

    meta = json.load(open(meta_path))
    m_by_key = {f"{m['run_id']}__t{int(m['tick']):07d}": m for m in meta}

    samples = []
    n_total = 0; n_skip_bank = 0; n_skip_meta = 0
    for p in scc_p.glob("*.scc.txt"):
        n_total += 1
        stem = p.stem.replace(".scc","")
        parts = stem.rsplit("__", 1)
        if len(parts) == 2 and parts[1] == "logged":
            key = parts[0]; tag = "logged"; rand_k = -1
        elif len(parts) == 2 and parts[1].startswith("rand_"):
            key = parts[0]; tag = "rand"; rand_k = int(parts[1].split("_")[1])
        else:
            key = stem; tag = "logged"; rand_k = -1
        m = m_by_key.get(key)
        if m is None: n_skip_meta += 1; continue
        if tag == "logged":
            if require_logged_bank and key not in log_keys: n_skip_bank += 1; continue
        else:
            if key not in rand_keys: n_skip_bank += 1; continue
        samples.append({
            "scc": str(p), "key": key, "tag": tag, "rand_k": rand_k,
            "frame_id": m["frame_id"],
            "run_id": m["run_id"], "tick": int(m["tick"]),
            "track": m["track"], "policy": m["policy"],
        })
    print(f"  total .scc.txt={n_total}  skipped(bank)={n_skip_bank}  skipped(meta)={n_skip_meta}  kept={len(samples)}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"samples": samples,
                   "raw_base": str(raw_p), "rand_bank": str(rand_p),
                   "log_bank": str(log_p)}, f)
    print(f"  manifest → {out_path}")
    return out_path, samples


class SCCSynthPairDataset(Dataset):
    def __init__(self,
                 manifest_path: str = DEFAULT_MANIFEST,
                 limit: int | None = None,
                 rebuild_if_missing: bool = True):
        if not Path(manifest_path).exists():
            if rebuild_if_missing:
                print(f"[dataset] manifest 없음 — build_manifest 호출")
                build_manifest(out_path=manifest_path)
            else:
                raise RuntimeError(f"manifest not found: {manifest_path}")
        m = json.load(open(manifest_path))
        self.raw_base = Path(m["raw_base"])
        self.rand_bank = Path(m["rand_bank"])
        self.log_bank  = Path(m["log_bank"])
        self.samples = m["samples"]
        if limit: self.samples = self.samples[:limit]
        if not self.samples:
            raise RuntimeError("manifest empty")
        print(f"[dataset] {len(self.samples)} samples from {manifest_path}")

    def __len__(self): return len(self.samples)

    def _paths(self, s):
        run = s["run_id"]; tick = s["tick"]
        run_dir = self.raw_base / s["track"] / s["policy"] / run
        return {
            "ci": run_dir / "CAMERA_1" / f"{tick}_Intensity.jpeg",
            "pi": run_dir / "CAMERA_1" / f"{tick - PREV_DTICK}_Intensity.jpeg",
            "ce": run_dir / "EGO_INFO" / f"{tick}.txt",
            "pe": run_dir / "EGO_INFO" / f"{tick - PREV_DTICK}.txt",
            "scc": Path(s["scc"]),
            "log_bank":  self.log_bank  / f"{s['key']}.bin",
            "rand_bank": self.rand_bank / f"{s['key']}.trajs.bin",
        }

    def __getitem__(self, i: int):
        s = self.samples[i]
        pp = self._paths(s)
        ego_c = _parse_ego(pp["ce"]); ego_p = _parse_ego(pp["pe"])
        if ego_c is None:
            # 거의 없음 (manifest 가 .scc.txt 있는 frame 만 = ego 있음). 안전 가드만.
            raise RuntimeError(f"ego_c missing {pp['ce']}")
        if ego_p is None:
            ego_p = ego_c   # Δyaw=0 fallback
        ex, ey = ego_c["position"][0], ego_c["position"][1]
        e_yaw = ego_c["orientation"][2] * DEG2RAD
        d_yaw = (e_yaw - (ego_p["orientation"][2] * DEG2RAD)) if ego_p else 0.0

        roi_c = _load_roi(pp["ci"])
        # prev 이미지 없으면 curr 로 fallback (Δyaw=0)
        if pp["pi"].exists():
            roi_p = _yaw_shift(_load_roi(pp["pi"]), d_yaw)
        else:
            roi_p = roi_c.copy()

        # 5d ego state
        ego_5d = np.array([
            ego_c["velocity"][0], ego_c["velocity"][1],
            ego_c["acceleration"][0], ego_c["acceleration"][1],
            ego_c["angularVelocity"][2] * DEG2RAD,
        ], dtype=np.float32)

        # input trajectory (world frame)
        if s["tag"] == "logged":
            traj_w = _load_logged_traj(pp["log_bank"])
        else:
            traj_w = _load_rand_traj(pp["rand_bank"], s["rand_k"])
            if traj_w is None:
                # bank entry missing — fall back to logged if exists
                if pp["log_bank"].exists():
                    traj_w = _load_logged_traj(pp["log_bank"])
                else:
                    raise RuntimeError(f"bank entry {s['rand_k']} missing for {s['key']}")
        traj_ego = _world_to_ego_traj(traj_w[:TRAJ_N], ex, ey, e_yaw)
        if traj_ego.shape[0] < TRAJ_N:
            pad = np.tile(traj_ego[-1:], (TRAJ_N - traj_ego.shape[0], 1))
            traj_ego = np.concatenate([traj_ego, pad], axis=0)

        # target = first TARG_N long_traj rows [steer (deg), ax]
        scc = _parse_scc_txt(pp["scc"])
        lt = scc["long_traj"]   # (M, 8): i time vx ax steer trq accel brake
        target = np.zeros((TARG_N, 2), dtype=np.float32)
        target_mask = np.zeros((TARG_N,), dtype=np.uint8)
        m_real = min(lt.shape[0], TARG_N)
        if m_real > 0:
            target[:m_real, 0] = lt[:m_real, 4]   # steer deg
            target[:m_real, 1] = lt[:m_real, 3]   # ax m/s^2
            target_mask[:m_real] = 1

        return {
            "image_curr":     _img_to_tensor(roi_c),
            "image_prev_rot": _img_to_tensor(roi_p),
            "ego_5d":         torch.from_numpy(ego_5d),
            "traj_40":        torch.from_numpy(traj_ego),
            "target":         torch.from_numpy(target),
            "target_mask":    torch.from_numpy(target_mask.astype(np.bool_)),
            "frame_id":       s["frame_id"],
            "tag":            s["tag"] + (f"_{s['rand_k']:05d}" if s["tag"]=="rand" else ""),
        }


def collate_scc_synth(batch):
    return {
        "image_curr":     torch.stack([b["image_curr"] for b in batch], dim=0),
        "image_prev_rot": torch.stack([b["image_prev_rot"] for b in batch], dim=0),
        "ego_5d":         torch.stack([b["ego_5d"] for b in batch], dim=0),
        "traj_40":        torch.stack([b["traj_40"] for b in batch], dim=0),
        "target":         torch.stack([b["target"] for b in batch], dim=0),
        "target_mask":    torch.stack([b["target_mask"] for b in batch], dim=0).bool(),
        "frame_id":       [b["frame_id"] for b in batch],
        "tag":            [b["tag"] for b in batch],
    }

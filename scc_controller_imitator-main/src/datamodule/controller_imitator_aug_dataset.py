"""controller_imitator v0301/v0302/etc → IyModel-호환 augmentation dataset.

용도: image 가 없는 controller_imitator .pt 데이터를 image_token=0 으로 마스킹하여
"객체 없음" 시나리오 학습 sample 로 사용. obj_gate 의 "객체 없으면 image 무시" 학습 강화.

각 .pt:
  trajectory_points_local (40, 3) — ego frame
  ego_info (6,) — [vx, vy, ax, ay, tire_angle, yaw_rate]
  target_steer (40,)              — deg
  target_acceleration (40,)        — m/s²

→ IyModel batch dict 변환:
  tok_curr     (40, C_b)  ← zeros
  tok_prev_raw (40, C_b)  ← zeros
  d_yaw        scalar      ← 0
  fx           scalar      ← 640
  ego_5d       (5,)         ← ego_info[:5]
  traj_40      (40, 3)
  target       (40, 2)      = [steer, ax_clipped]
  target_mask  (40,) bool   ← all True
  obj_target   (40,)        ← zeros (no obj)
  min_dist     scalar       ← +inf (no lead)
"""
from __future__ import annotations
import glob, pickle
from pathlib import Path
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset


class ControllerImitatorAugDataset(Dataset):
    """controller_imitator .pt → image-masked augmentation."""

    def __init__(self,
                 dirs: List[str],
                 C_b: int = 256,
                 fx: float = 640.0,
                 ax_clip: float = 20.0,     # MPC 합성 cap 과 동일
                 steer_clip: float = 30.0,  # deg
                 limit: int | None = None):
        self.C_b = int(C_b)
        self.fx  = float(fx)
        self.ax_clip = float(ax_clip)
        self.steer_clip = float(steer_clip)

        # 두 가지 source 지원:
        #   (a) .pt 풀린 dirs — sample 별 torch.load
        #   (b) all_normal_data.pkl — pickle 통합 (data: list of dict)
        self.files: list[str] = []         # (a)
        self.pkl_samples: list[dict] = []  # (b)
        for d in dirs:
            # 우선순위: npz (mmap, 즉시) > pkl (deserialize 5min) > .pt (sample 별 disk I/O 가장 느림)
            npz_p = Path(d) / "all_normal_data.npz"
            pkl_p = Path(d) / "all_normal_data.pkl"
            if npz_p.exists():
                print(f"  loading npz (mmap): {npz_p}")
                z = np.load(npz_p, mmap_mode='r', allow_pickle=False)
                ego_5d = z['ego_5d']; traj_40 = z['traj_40']; target = z['target']
                N = ego_5d.shape[0]
                for i in range(N):
                    self.pkl_samples.append({
                        "ego_info": ego_5d[i],
                        "trajectory_points_local": traj_40[i],
                        "target_steer": target[i, :, 0],
                        "target_acceleration": target[i, :, 1],
                    })
            elif pkl_p.exists():
                print(f"  loading pkl: {pkl_p}")
                p = pickle.load(open(pkl_p, "rb"))
                self.pkl_samples.extend(p["data"] if isinstance(p, dict) and "data" in p else p)
            else:
                # fallback: .pt 풀린 sample 별 파일 (가장 느림)
                pts = sorted(glob.glob(f"{d}/**/all_*.pt", recursive=True))
                if pts:
                    self.files.extend(pts)
        N = len(self.files) + len(self.pkl_samples)
        if limit:
            # split limit proportionally
            self.files = self.files[:limit]
            self.pkl_samples = self.pkl_samples[:max(0, limit - len(self.files))]
        print(f"[ctrl-imit-aug] {len(self.files)} .pt + {len(self.pkl_samples)} pkl = {len(self.files)+len(self.pkl_samples)} samples from {len(dirs)} dirs")

        # i_1 cache 대신 zeros 사용 — pre-allocated single tensor (re-use 효율)
        self._zero_tok = torch.zeros((40, self.C_b), dtype=torch.float32)
        self._zero_obj = torch.zeros(40, dtype=torch.float32)
        self._all_true = torch.ones(40, dtype=torch.bool)
        self._zero_dy  = torch.tensor(0.0, dtype=torch.float32)
        self._fx_t     = torch.tensor(self.fx, dtype=torch.float32)
        self._inf_d    = torch.tensor(float("inf"), dtype=torch.float32)

    def __len__(self): return len(self.files) + len(self.pkl_samples)

    def __getitem__(self, i: int):
        if i < len(self.files):
            f = self.files[i]
            d = torch.load(f, weights_only=False)
            key = Path(f).stem
        else:
            d = self.pkl_samples[i - len(self.files)]
            key = f"pkl_{i - len(self.files):06d}"
        ego = d["ego_info"]
        if isinstance(ego, torch.Tensor):
            ego5 = ego[:5].float()
        else:
            ego5 = torch.tensor(ego[:5], dtype=torch.float32)
        # ego outlier guard — vx/ax 도 합리 범위 clamp + NaN→0
        ego5 = torch.nan_to_num(ego5, nan=0.0, posinf=50.0, neginf=-50.0)
        ego5 = ego5.clamp(min=-50.0, max=80.0)
        traj = d["trajectory_points_local"]
        if isinstance(traj, torch.Tensor):
            traj40 = traj[:40, :3].float()
        else:
            traj40 = torch.tensor(np.asarray(traj)[:40, :3], dtype=torch.float32)
        traj40 = torch.nan_to_num(traj40, nan=0.0, posinf=1000.0, neginf=-1000.0).clamp(-2000.0, 2000.0)
        # target_steer/acceleration: tuple of floats
        steer = torch.tensor(np.asarray(d["target_steer"])[:40], dtype=torch.float32)
        ax    = torch.tensor(np.asarray(d["target_acceleration"])[:40], dtype=torch.float32)
        # NaN/Inf guard + clip — controller_imitator 가 가끔 큰 값 (시뮬 spike) 출력
        steer = torch.nan_to_num(steer, nan=0.0, posinf=self.steer_clip, neginf=-self.steer_clip)
        ax    = torch.nan_to_num(ax,    nan=0.0, posinf=self.ax_clip,    neginf=-self.ax_clip)
        steer = steer.clamp(-self.steer_clip, self.steer_clip)
        ax    = ax.clamp(-self.ax_clip,    self.ax_clip)
        target = torch.stack([steer, ax], dim=-1)   # (40, 2)
        return {
            "tok_curr":     self._zero_tok,
            "tok_prev_raw": self._zero_tok,
            "d_yaw":        self._zero_dy,
            "fx":           self._fx_t,
            "ego_5d":       ego5,
            "traj_40":      traj40,
            "target":       target,
            "target_mask":  self._all_true,
            "obj_target":   self._zero_obj,
            "min_dist":     self._inf_d,
            "closing_rate": torch.tensor(0.0, dtype=torch.float32),  # ci_aug 는 lead 정보 없음
            "inlane_dist":  self._inf_d,  # ci_aug no lead → inf (no augment)
            "inlane_closing": torch.tensor(0.0, dtype=torch.float32),
            "a2_feat":      torch.zeros(40, 7, dtype=torch.float32),  # ci_aug 는 a_2 features 없음
            "i2_mem":       torch.zeros(40, 128, dtype=torch.float32),  # ci_aug 는 image_mem 없음 (zeros)
            "a2_logits":    torch.cat([torch.full((40, 1), -10.0), torch.zeros(40, 6)], dim=-1),  # obj=-10 → gate≈0 (image 무관)
            "tag":          "ci_aug",
            "key":          key,
        }

"""Pre-baked f_1 dataset — 전부 in-memory, I/O 0.

Inputs:
  prebake npz (scripts/prebake_f1_dataset.py 결과)
  fi_mem npz (scripts/compute_fi_mem_cache.py 결과)

각 sample:
  fi_mem      (40, D) float32   — cached image fusion
  ego_5d      (5,)              — frame 별
  traj_40     (40, 3)           — input trajectory ego frame
  target      (40, 2)           — long_traj steer/ax
  mask        (40,) bool
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_PREBAKE = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/f1_prebake_v1.npz"
DEFAULT_CACHE   = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/fi_mem_cache_v1.npz"


class SCCSynthPrebakedDataset(Dataset):
    def __init__(self, prebake_path: str = DEFAULT_PREBAKE,
                 cache_path: str = DEFAULT_CACHE,
                 limit: int | None = None):
        print(f"[prebaked] loading prebake {prebake_path}")
        z = np.load(prebake_path, allow_pickle=True)
        self.frame_keys = z["frame_keys"].tolist()
        self.ego_5d = z["ego_5d"]                      # (N_frames, 5)
        self.sample_keys = z["sample_keys"].tolist()
        self.sample_tags = z["sample_tags"].tolist()
        self.sample_frame_idx = z["sample_frame_idx"]   # (N_samples,)
        self.traj   = z["traj"]                         # (N_samples, 40, 3)
        self.target = z["target"]                       # (N_samples, 40, 2)
        self.mask   = z["mask"]                         # (N_samples, 40) bool
        print(f"  prebake: samples={len(self.sample_keys)}  frames={len(self.frame_keys)}")

        print(f"[prebaked] loading fi_mem cache {cache_path}")
        zc = np.load(cache_path, allow_pickle=True)
        cache_keys = zc["keys"].tolist()
        self.fi_mem = zc["fi_mem"]                     # (N_cache, 40, D) float16
        cache_idx = {k: i for i, k in enumerate(cache_keys)}
        # frame_keys[i] → cache row
        self.frame_to_cache = np.array([cache_idx.get(k, -1)
                                          for k in self.frame_keys], dtype=np.int64)
        n_missing = int((self.frame_to_cache < 0).sum())
        print(f"  fi_mem: shape={self.fi_mem.shape}  missing cache for {n_missing} frames")

        # filter: drop samples whose frame has no cache or no valid traj/target
        valid_frame = self.frame_to_cache >= 0
        valid_target = self.mask.any(axis=1)        # target 이 1 step 이라도 valid
        # frame valid mapping
        sample_frame_valid = valid_frame[self.sample_frame_idx]
        keep = sample_frame_valid & valid_target
        self.valid_idx = np.where(keep)[0]
        print(f"  valid samples: {len(self.valid_idx)} / {len(self.sample_keys)}")
        if limit: self.valid_idx = self.valid_idx[:limit]

    def __len__(self): return len(self.valid_idx)

    def __getitem__(self, i: int):
        idx = int(self.valid_idx[i])
        fi_idx = int(self.frame_to_cache[self.sample_frame_idx[idx]])
        return {
            "fi_mem":  torch.from_numpy(self.fi_mem[fi_idx].astype(np.float32)),
            "ego_5d":  torch.from_numpy(self.ego_5d[self.sample_frame_idx[idx]].astype(np.float32)),
            "traj_40": torch.from_numpy(self.traj[idx]),
            "target":  torch.from_numpy(self.target[idx]),
            "target_mask": torch.from_numpy(self.mask[idx].astype(np.bool_)),
            "tag":     self.sample_tags[idx],
            "key":     self.sample_keys[idx],
        }


def collate_prebaked(batch):
    return {
        "fi_mem":      torch.stack([b["fi_mem"] for b in batch], dim=0),
        "ego_5d":      torch.stack([b["ego_5d"] for b in batch], dim=0),
        "traj_40":     torch.stack([b["traj_40"] for b in batch], dim=0),
        "target":      torch.stack([b["target"] for b in batch], dim=0),
        "target_mask": torch.stack([b["target_mask"] for b in batch], dim=0).bool(),
        "tag":         [b["tag"] for b in batch],
        "key":         [b["key"] for b in batch],
    }

"""Cached fi_mem 기반 f_1 학습 dataset.

이미지 forward 를 사전 캐시 (compute_fi_mem_cache.py 결과) 한 fi_mem 을 사용.
sample 당 cost: NPZ in-memory lookup + 1 ego file read + 1 bank read + 1 .scc.txt parse.

각 sample 출력:
  fi_mem      (40, D)         cached image memory (float32 변환)
  ego_5d      (5,)
  traj_40     (40, 3)         input trajectory (ego frame [x, y, speed])
  target      (40, 2)         long_traj 의 [steer (deg), ax (m/s²)] 처음 40 row
  target_mask (40,) bool
"""
from __future__ import annotations
import json, struct
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from src.datamodule.scc_synth_pair_dataset import (
    DEG2RAD, PREV_DTICK, TRAJ_N, TARG_N,
    _parse_ego, _parse_scc_txt, _load_rand_traj, _load_logged_traj, _world_to_ego_traj,
    DEFAULT_MANIFEST, DEFAULT_RAW,
)

DEFAULT_CACHE = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/fi_mem_cache_v1.npz"


class SCCSynthCachedDataset(Dataset):
    def __init__(self,
                 manifest_path: str = DEFAULT_MANIFEST,
                 cache_path: str = DEFAULT_CACHE,
                 limit: Optional[int] = None):
        print(f"[cached] loading manifest {manifest_path}")
        m = json.load(open(manifest_path))
        self.raw_base = Path(m["raw_base"])
        self.rand_bank = Path(m["rand_bank"])
        self.log_bank  = Path(m["log_bank"])

        print(f"[cached] loading fi_mem cache {cache_path}")
        z = np.load(cache_path, allow_pickle=True)
        keys = z["keys"].tolist()
        self.fi_mem = z["fi_mem"]    # (N_keys, 40, D) float16
        self.meta = json.loads(str(z["meta"]))
        print(f"  fi_mem: {self.fi_mem.shape} (D={self.fi_mem.shape[-1]})  cached_keys={len(keys)}")

        self.key2idx = {k: i for i, k in enumerate(keys)}
        cached_set = set(keys)
        all_samples = m["samples"]
        # filter: only samples whose key is in cache
        self.samples = [s for s in all_samples if s["key"] in cached_set]
        if limit: self.samples = self.samples[:limit]
        print(f"  samples after cache filter: {len(self.samples)} / {len(all_samples)}")
        if not self.samples:
            raise RuntimeError("no samples (cache empty?)")

    def __len__(self): return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        # fi_mem (cached)
        fi_mem = torch.from_numpy(self.fi_mem[self.key2idx[s["key"]]].astype(np.float32))

        run = s["run_id"]; tick = s["tick"]
        run_dir = self.raw_base / s["track"] / s["policy"] / run
        ego_c = _parse_ego(run_dir / "EGO_INFO" / f"{tick}.txt")
        if ego_c is None:
            # fallback: zeros
            ego_5d = np.zeros(5, dtype=np.float32)
            ex = ey = e_yaw = 0.0
        else:
            ex, ey = ego_c["position"][0], ego_c["position"][1]
            e_yaw = ego_c["orientation"][2] * DEG2RAD
            ego_5d = np.array([
                ego_c["velocity"][0], ego_c["velocity"][1],
                ego_c["acceleration"][0], ego_c["acceleration"][1],
                ego_c["angularVelocity"][2] * DEG2RAD,
            ], dtype=np.float32)

        # trajectory
        if s["tag"] == "logged":
            traj_w = _load_logged_traj(self.log_bank / f"{s['key']}.bin")
        else:
            traj_w = _load_rand_traj(self.rand_bank / f"{s['key']}.trajs.bin", s["rand_k"])
            if traj_w is None:
                traj_w = _load_logged_traj(self.log_bank / f"{s['key']}.bin")
        traj_ego = _world_to_ego_traj(traj_w[:TRAJ_N], ex, ey, e_yaw)
        if traj_ego.shape[0] < TRAJ_N:
            pad = np.tile(traj_ego[-1:], (TRAJ_N - traj_ego.shape[0], 1))
            traj_ego = np.concatenate([traj_ego, pad], axis=0)

        # target
        scc = _parse_scc_txt(Path(s["scc"]))
        lt = scc["long_traj"]
        target = np.zeros((TARG_N, 2), dtype=np.float32)
        target_mask = np.zeros((TARG_N,), dtype=np.uint8)
        m_real = min(lt.shape[0], TARG_N)
        if m_real > 0:
            target[:m_real, 0] = lt[:m_real, 4]   # steer deg
            target[:m_real, 1] = lt[:m_real, 3]   # ax
            target_mask[:m_real] = 1

        return {
            "fi_mem":      fi_mem,
            "ego_5d":      torch.from_numpy(ego_5d),
            "traj_40":     torch.from_numpy(traj_ego),
            "target":      torch.from_numpy(target),
            "target_mask": torch.from_numpy(target_mask.astype(np.bool_)),
            "frame_id":    s["frame_id"],
            "tag":         s["tag"] + (f"_{s['rand_k']:05d}" if s["tag"]=="rand" else ""),
        }


def collate_cached(batch):
    return {
        "fi_mem":      torch.stack([b["fi_mem"] for b in batch], dim=0),
        "ego_5d":      torch.stack([b["ego_5d"] for b in batch], dim=0),
        "traj_40":     torch.stack([b["traj_40"] for b in batch], dim=0),
        "target":      torch.stack([b["target"] for b in batch], dim=0),
        "target_mask": torch.stack([b["target_mask"] for b in batch], dim=0).bool(),
        "frame_id":    [b["frame_id"] for b in batch],
        "tag":         [b["tag"] for b in batch],
    }

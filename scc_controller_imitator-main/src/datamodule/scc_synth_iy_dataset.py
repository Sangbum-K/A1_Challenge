"""i_y refactor dataset — per-image i_1 token cache + token-level Δyaw shift.

각 sample:
  tok_curr        (40, C_b)  float32   curr image 의 i_1 token cache lookup
  tok_prev_raw    (40, C_b)  float32   prev image 의 i_1 token cache lookup (yaw warp 안됨)
  d_yaw           float32             curr_yaw - prev_yaw (rad). model 이 i_y shift 계산
  ego_5d          (5,)
  traj_40         (40, 3)
  target          (40, 2)
  target_mask     (40,) bool

cache 와 prebake 별도. prev_dt 변경 시 prebake 만 (또는 manifest 만) 재생성, cache 는 unaffected.
"""
from __future__ import annotations
import json, struct
from pathlib import Path
from typing import Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset

from src.datamodule.scc_synth_pair_dataset import (
    DEG2RAD, _parse_ego, _parse_scc_txt, _load_rand_traj, _load_logged_traj,
    _world_to_ego_traj, DEFAULT_MANIFEST, DEFAULT_RAW, TRAJ_N, TARG_N,
)

DEFAULT_PREBAKE = "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/prebake/f1_prebake_iy_v1.npz"
DEFAULT_I1_CACHE = "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/cache/i1_tok_cache_v1.npz"


class SCCSynthIyDataset(Dataset):
    """prebake 의 traj/target/ego/curr_key/prev_key 와 i_1 token cache 를 결합."""

    def __init__(self,
                 prebake_path: str = DEFAULT_PREBAKE,
                 i1_cache_path: str = DEFAULT_I1_CACHE,
                 close_lead_meta_path: Optional[str] = None,
                 a2_cache_path: Optional[str] = None,
                 i2_cache_path: Optional[str] = None,
                 inlane_gt_cache_path: Optional[str] = None,
                 limit: Optional[int] = None):
        print(f"[iy-dataset] prebake {prebake_path}")
        z = np.load(prebake_path, allow_pickle=True)
        self.sample_keys     = z["sample_keys"].tolist()
        self.sample_tags     = z["sample_tags"].tolist()
        self.curr_keys       = z["curr_keys"].tolist()
        self.prev_keys       = z["prev_keys"].tolist()
        self.d_yaws          = z["d_yaws"]                 # (N_samples,) rad
        self.ego_5d          = z["ego_5d"]                 # (N_samples, 5)
        self.traj            = z["traj"]                    # (N_samples, 40, 3)
        self.target          = z["target"]                  # (N_samples, 40, 2)
        self.mask            = z["mask"]                    # (N_samples, 40) bool
        # i_g obj_target (40,) — 신규 prebake (v3) 부터. 없으면 zeros (no gate supervision)
        if "obj_target" in z.files:
            self.obj_target = z["obj_target"].astype(np.float32)
            print(f"  obj_target column present (mean={self.obj_target.mean():.3f})")
        else:
            self.obj_target = np.zeros((len(self.sample_keys), 40), dtype=np.float32)
        # sample_fx: 신 prebake (DriverB_new 이후) 에만 존재. 없으면 320 default
        if "sample_fx" in z.files:
            self.sample_fx = z["sample_fx"].astype(np.float32)
            print(f"  fx column: present (uniq={sorted(set(self.sample_fx.tolist()))[:5]})")
        else:
            self.sample_fx = np.full(len(self.sample_keys), 320.0, dtype=np.float32)
        print(f"  samples={len(self.sample_keys)}")

        print(f"[iy-dataset] i_1 cache {i1_cache_path}")
        zc = np.load(i1_cache_path, allow_pickle=True)
        cache_keys = zc["keys"].tolist()
        self.tokens = zc["tokens"]       # (N_cache, 40, C_b) float16
        self.meta_cache = json.loads(str(zc["meta"]))
        self.C_b = self.tokens.shape[-1]
        self.k2idx = {k: i for i, k in enumerate(cache_keys)}
        print(f"  cache: {self.tokens.shape}  C_b={self.C_b}  meta={self.meta_cache}")

        # filter: curr_key + prev_key both in cache, target valid
        valid_t = self.mask.any(axis=1)
        valid_c = np.array([k in self.k2idx for k in self.curr_keys])
        valid_p = np.array([k in self.k2idx for k in self.prev_keys])
        keep = valid_t & valid_c & valid_p
        self.valid_idx = np.where(keep)[0]
        print(f"  valid samples: {len(self.valid_idx)} / {len(self.sample_keys)} "
              f"(t={valid_t.sum()} c={valid_c.sum()} p={valid_p.sum()})")
        if limit: self.valid_idx = self.valid_idx[:limit]

        # min_dist + closing_rate + inlane arrays 미리 초기화 (a_2 cache 가 inlane 채움)
        self.min_dist       = np.full(len(self.sample_keys), float("inf"), dtype=np.float32)
        self.closing_rate   = np.zeros(len(self.sample_keys), dtype=np.float32)
        self.inlane_dist    = np.full(len(self.sample_keys), float("inf"), dtype=np.float32)
        self.inlane_closing = np.zeros(len(self.sample_keys), dtype=np.float32)

        # GT inlane cache (우선) — aux GT 의 dist + (-vL_rel) 로 inlane 채움. cache 종류 무관.
        # predicted (a_2 cache, 근거리 과대평가) 보다 정확 → ttc-augment 가 GT 로 trigger.
        if inlane_gt_cache_path:
            print(f"[iy-dataset] inlane GT cache {inlane_gt_cache_path}")
            zg = np.load(inlane_gt_cache_path, allow_pickle=True)
            g_k2i = {k: i for i, k in enumerate(zg["keys"].tolist())}
            g_dist = zg["inlane_dist"]; g_clos = zg["inlane_closing"]
            n_hit = 0
            for i, k in enumerate(self.curr_keys):
                gi = g_k2i.get(k)
                if gi is not None:
                    self.inlane_dist[i] = g_dist[gi]
                    self.inlane_closing[i] = g_clos[gi]
                    if np.isfinite(g_dist[gi]): n_hit += 1
            print(f"  inlane GT 적용: {n_hit}/{len(self.curr_keys)} (in-lane lead 있음)")
            self._inlane_from_gt = True
        else:
            self._inlane_from_gt = False

        # v6 i_2 mem cache (optional): per-frame (40, 128) image_mem + a_2 logits (40, 7)
        self.i2_mem = None; self.i2_k2idx = None; self.i2_a2_logits = None
        if i2_cache_path:
            print(f"[iy-dataset] i_2 cache {i2_cache_path}")
            zi = np.load(i2_cache_path, allow_pickle=True)
            i2_keys = zi["keys"].tolist()
            self.i2_mem = zi["i2_mem"]
            if "a2_logits" in zi.files:
                self.i2_a2_logits = zi["a2_logits"]   # (N_frame, 40, 7) f16
            self.i2_k2idx = {k: i for i, k in enumerate(i2_keys)}
            print(f"  i_2 cache: {self.i2_mem.shape}  + a2_logits={self.i2_a2_logits.shape if self.i2_a2_logits is not None else 'N/A'}")

        # a_2 v4 cache (optional): per-frame (40, 7) features
        self.a2_out = None; self.a2_k2idx = None
        if a2_cache_path:
            print(f"[iy-dataset] a_2 cache {a2_cache_path}")
            za = np.load(a2_cache_path, allow_pickle=True)
            a2_keys = za["keys"].tolist()
            self.a2_out = za["a2_out"]   # (N_frame, 40, 7) f16
            self.a2_k2idx = {k: i for i, k in enumerate(a2_keys)}
            print(f"  a_2 cache: {self.a2_out.shape}  dtype={self.a2_out.dtype}")

            # in-lane dist + closing 사전 계산 (bin 16-25, obj logit > 0, dist × 40)
            BIN_LO, BIN_HI = 16, 25     # inclusive, 10 bins (center lane)
            OBJ_THR = 0.0               # sigmoid(logit) > 0.5 == logit > 0
            DIST_SCALE = 40.0           # a_2 train scale-dist
            DTICK_INLANE = 10
            DT_S_INLANE = 0.1
            n_inlane = 0
            for i, k in enumerate(self.curr_keys):
                if k not in self.a2_k2idx: continue
                a2_curr = self.a2_out[self.a2_k2idx[k]]            # (40, 7) f16
                obj_curr = a2_curr[BIN_LO:BIN_HI+1, 0].astype(np.float32)
                has_obj = obj_curr > OBJ_THR
                if not has_obj.any(): continue
                d_arr = a2_curr[BIN_LO:BIN_HI+1, 6].astype(np.float32) * DIST_SCALE
                d_inlane = d_arr[has_obj]
                d_now = float(d_inlane.min())
                if not np.isfinite(d_now) or d_now <= 0: continue
                self.inlane_dist[i] = d_now
                n_inlane += 1
                # closing: prev frame 의 same bin range
                try:
                    run, t_str = k.rsplit('__t', 1)
                    prev_k = f"{run}__t{int(t_str) - DTICK_INLANE:07d}"
                    if prev_k in self.a2_k2idx:
                        a2_prev = self.a2_out[self.a2_k2idx[prev_k]]
                        obj_prev = a2_prev[BIN_LO:BIN_HI+1, 0].astype(np.float32) > OBJ_THR
                        if obj_prev.any():
                            d_prev = float((a2_prev[BIN_LO:BIN_HI+1, 6].astype(np.float32) * DIST_SCALE)[obj_prev].min())
                            self.inlane_closing[i] = (d_prev - d_now) / DT_S_INLANE
                except Exception: pass
            print(f"  in-lane (bin 16-25, obj>0) dist computed: {n_inlane}/{len(self.curr_keys)} samples")

        # close-lead meta (optional): self.min_dist + self.closing_rate 채움 (위에 미리 초기화됨)
        if close_lead_meta_path:
            print(f"[iy-dataset] close-lead meta {close_lead_meta_path}")
            zm = np.load(close_lead_meta_path)
            cl_keys = zm["keys"].tolist(); cl_dists = zm["dists"]
            cl_map = {k: d for k, d in zip(cl_keys, cl_dists)}
            n_hit = 0
            DTICK_CLOSING = 10  # 1 tick = 10ms, 10 tick = 100ms 윈도
            DT_S = DTICK_CLOSING * 0.01   # 0.1s
            n_closing = 0
            for i, k in enumerate(self.curr_keys):
                if k in cl_map:
                    self.min_dist[i] = cl_map[k]; n_hit += 1
                    # closing rate: 이전 frame dist - 현재 dist (양수면 접근)
                    try:
                        run, t_str = k.rsplit('__t', 1)
                        prev_k = f"{run}__t{int(t_str) - DTICK_CLOSING:07d}"
                        if prev_k in cl_map:
                            d_prev = cl_map[prev_k]; d_now = cl_map[k]
                            if np.isfinite(d_prev) and np.isfinite(d_now):
                                self.closing_rate[i] = (d_prev - d_now) / DT_S
                                n_closing += 1
                    except Exception: pass
            print(f"  matched {n_hit}/{len(self.curr_keys)} samples to close-lead meta")
            print(f"  closing-rate computed: {n_closing} samples (dist sequence diff)")
            valid_dists = self.min_dist[self.valid_idx]
            n_lt30 = int((valid_dists < 30).sum())
            print(f"  close-lead distribution: <30m={n_lt30}/{len(valid_dists)}")

    def __len__(self): return len(self.valid_idx)

    def __getitem__(self, i: int):
        idx = int(self.valid_idx[i])
        ci = self.k2idx[self.curr_keys[idx]]
        pi = self.k2idx[self.prev_keys[idx]]
        # NaN/Inf/outlier guard — prebake 일부 sample 의 ego_5d 가 sensor noise
        ego = torch.from_numpy(self.ego_5d[idx].astype(np.float32))
        ego = torch.nan_to_num(ego, nan=0.0, posinf=80.0, neginf=-50.0).clamp(-50.0, 80.0)
        traj = torch.from_numpy(self.traj[idx])
        traj = torch.nan_to_num(traj, nan=0.0, posinf=300.0, neginf=-300.0).clamp(-300.0, 300.0)
        target = torch.from_numpy(self.target[idx])
        target = torch.nan_to_num(target, nan=0.0, posinf=25.0, neginf=-25.0).clamp(-90.0, 90.0)
        d_yaw = float(self.d_yaws[idx])
        if not (d_yaw == d_yaw) or abs(d_yaw) > 10:  # NaN check + clamp
            d_yaw = 0.0
        out = {
            "tok_curr":      torch.from_numpy(self.tokens[ci].astype(np.float32)),
            "tok_prev_raw":  torch.from_numpy(self.tokens[pi].astype(np.float32)),
            "d_yaw":         torch.tensor(d_yaw, dtype=torch.float32),
            "fx":            torch.tensor(float(self.sample_fx[idx]), dtype=torch.float32),
            "ego_5d":        ego,
            "traj_40":       traj,
            "target":        target,
            "target_mask":   torch.from_numpy(self.mask[idx].astype(np.bool_)),
            "obj_target":    torch.from_numpy(self.obj_target[idx]),
            "min_dist":      torch.tensor(float(self.min_dist[idx]), dtype=torch.float32),
            "closing_rate":  torch.tensor(float(self.closing_rate[idx]), dtype=torch.float32),
            "inlane_dist":   torch.tensor(float(self.inlane_dist[idx]), dtype=torch.float32),
            "inlane_closing":torch.tensor(float(self.inlane_closing[idx]), dtype=torch.float32),
            "tag":           self.sample_tags[idx],
            "key":           self.sample_keys[idx],
        }
        # a_2 v4 cache lookup (없으면 zeros — 학습은 "a2=0 → 정보없음" 으로 처리)
        if self.a2_out is not None:
            ck = self.curr_keys[idx]
            if ck in self.a2_k2idx:
                out["a2_feat"] = torch.from_numpy(self.a2_out[self.a2_k2idx[ck]].astype(np.float32))
            else:
                out["a2_feat"] = torch.zeros(40, self.a2_out.shape[-1], dtype=torch.float32)
        else:
            # a_2 cache 없어도 default zeros — ci_aug 등 다른 dataset 과 stack 호환
            out["a2_feat"] = torch.zeros(40, 7, dtype=torch.float32)
        # v6: i_2 cache lookup (image_mem 직접 — img_proj+img_fusion 우회)
        if self.i2_mem is not None:
            ck = self.curr_keys[idx]
            if ck in self.i2_k2idx:
                out["i2_mem"] = torch.from_numpy(self.i2_mem[self.i2_k2idx[ck]].astype(np.float32))
                if self.i2_a2_logits is not None:
                    out["a2_logits"] = torch.from_numpy(self.i2_a2_logits[self.i2_k2idx[ck]].astype(np.float32))
            else:
                out["i2_mem"] = torch.zeros(40, self.i2_mem.shape[-1], dtype=torch.float32)
                if self.i2_a2_logits is not None:
                    out["a2_logits"] = torch.zeros(40, self.i2_a2_logits.shape[-1], dtype=torch.float32)
        # default zeros (collate compat — a_2 v6 calibration mean ~-1.7 → sigmoid 0.16)
        if "a2_logits" not in out:
            out["a2_logits"] = torch.full((40, 7), -1.7, dtype=torch.float32)  # ci_aug 등 — gate ≈ 0.16
            out["a2_logits"][:, 0] = -10.0  # obj 는 매우 negative (gate ≈ 0)
        return out


def collate_iy(batch):
    out = {
        "tok_curr":     torch.stack([b["tok_curr"]     for b in batch], dim=0),
        "tok_prev_raw": torch.stack([b["tok_prev_raw"] for b in batch], dim=0),
        "d_yaw":        torch.stack([b["d_yaw"]        for b in batch], dim=0),
        "fx":           torch.stack([b["fx"]           for b in batch], dim=0),
        "ego_5d":       torch.stack([b["ego_5d"]       for b in batch], dim=0),
        "traj_40":      torch.stack([b["traj_40"]      for b in batch], dim=0),
        "target":       torch.stack([b["target"]       for b in batch], dim=0),
        "target_mask":  torch.stack([b["target_mask"]  for b in batch], dim=0).bool(),
        "obj_target":   torch.stack([b["obj_target"]   for b in batch], dim=0),
        "min_dist":     torch.stack([b["min_dist"]     for b in batch], dim=0),
        "closing_rate": torch.stack([b["closing_rate"] for b in batch], dim=0),
        "inlane_dist":  torch.stack([b["inlane_dist"]  for b in batch], dim=0),
        "inlane_closing":torch.stack([b["inlane_closing"] for b in batch], dim=0),
        "tag":          [b["tag"] for b in batch],
        "key":          [b["key"] for b in batch],
    }
    # optional keys: 모든 샘플에 있을 때만 stack (SCCSynth 는 i2-cache 없으면 i2_mem 미생성,
    # ci_aug 는 항상 생성 → mixed batch 에서 batch[0] 만 보면 KeyError. all() 로 robust 화).
    if all("a2_feat" in b for b in batch):
        out["a2_feat"] = torch.stack([b["a2_feat"] for b in batch], dim=0)
    if all("i2_mem" in b for b in batch):
        out["i2_mem"] = torch.stack([b["i2_mem"] for b in batch], dim=0)
    if all("a2_logits" in b for b in batch):
        out["a2_logits"] = torch.stack([b["a2_logits"] for b in batch], dim=0)
    return out

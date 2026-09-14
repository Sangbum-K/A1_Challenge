"""#635-format dataset: prebake → (traj6, ego6, i2_mem, steer_label, pedal_label).

main = controller_imitator #635 (frozen) → steer. f_2 = side → pedal.

#635 입력 (confirmed, validated steer corr 0.97 on model's own val):
  traj: 6점 (6,3) [-y_local, +x_local, abs_time], abs_time=[0.05,0.30,0.55,0.80,1.05,1.30]
        prebake traj_40 (0.1s, ego-frame [x_fwd,y_left,speed]) 를 target_times 에 선형보간.
  ego:  (6,) [vx, vy, ax, ay, tire_angle(rad), yaw_rate]
        prebake ego_5d=[vx,vy,ax,ay,yaw_rate] → tire_angle 은 EGO_INFO steer 없으면 kinematic
        atan(WB·yaw_rate/vx), |vx|<5 → 0.
출력 라벨:
  steer_label = target_steer(deg) / 40   (→ #635 out[0])
  pedal_label = pedal_map(speed, target_ax) 사전계산 cache (→ f_2 out)
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

T6 = np.array([0.05, 0.30, 0.55, 0.80, 1.05, 1.30], dtype=np.float32)
TSRC = np.arange(40, dtype=np.float32) * 0.1   # prebake traj_40 시간축 (0.1s)
WB = 2.35           # wheelbase (kinematic tire angle)
STEER_NORM = 40.0   # steer(deg) / 40 → [-1,1]


def build_traj6(traj40: np.ndarray, mode: str = "speed") -> np.ndarray:
    """(40,3)[x_fwd,y_left,speed] → (6,3)[-y, x, ch2] at T6 (linear interp).
    mode='speed' (#654): ch2 = interp speed. mode='abs_time' (#635): ch2 = T6.
    """
    xi = np.interp(T6, TSRC, traj40[:, 0])
    yi = np.interp(T6, TSRC, traj40[:, 1])
    if mode == "speed":
        ch2 = np.interp(T6, TSRC, traj40[:, 2])
    else:
        ch2 = T6
    return np.stack([-yi, xi, ch2], axis=1).astype(np.float32)


def build_ego6(ego5: np.ndarray) -> np.ndarray:
    """[vx,vy,ax,ay,yaw_rate] → [vx,vy,ax,ay,tire_angle,yaw_rate] (kinematic tire)."""
    vx, vy, ax, ay, yr = [float(x) for x in ego5]
    tire = float(np.arctan(WB * yr / vx)) if abs(vx) >= 5.0 else 0.0
    return np.array([vx, vy, ax, ay, tire, yr], dtype=np.float32)


# a_2 logit ch (predict_var=False): [obj, vL_abs, vY_abs, vL_rel, vY_rel, prox, dist]
A2_CH_OBJ, A2_CH_VLREL, A2_CH_DIST = 0, 3, 6
A2_BIN_LO, A2_BIN_HI = 16, 25       # in-lane (center) bins
A2_SCALE_V, A2_SCALE_DIST = 20.0, 40.0   # a_2 학습 normalize scale (denorm 용)


def _a2logits_to_closing(a2_logits, obj_thr=0.5):
    """(N,40,7) a2_logits → (N,) in-lane lead closing (m/s, 양수=접근).

    build_pred_vrel_cache.py 와 동일 aggregation: center bin 16-25 중 obj_prob>thr 인 col 중
    pred dist 최소(=최근접 lead)인 col 의 vL_rel → closing = -vL_rel*SCALE_V.
    lead 없으면 0. **20f a_2(mf20 cache)의 a2_logits 에서 직접** → 추론(closed_loop 20f a_2)=학습 일관.
    """
    a = a2_logits.astype(np.float32)
    obj = 1.0 / (1.0 + np.exp(-a[:, A2_BIN_LO:A2_BIN_HI + 1, A2_CH_OBJ]))   # (N,10) prob
    vl  = a[:, A2_BIN_LO:A2_BIN_HI + 1, A2_CH_VLREL] * A2_SCALE_V           # (N,10) m/s
    di  = a[:, A2_BIN_LO:A2_BIN_HI + 1, A2_CH_DIST] * A2_SCALE_DIST         # (N,10) m
    closing = np.zeros(len(a), dtype=np.float32)
    om = obj > obj_thr
    d_masked = np.where(om, di, np.inf)
    has_lead = om.any(axis=1)
    sel = np.argmin(d_masked, axis=1)                                      # 최근접 lead col
    rows = np.arange(len(a))
    closing[has_lead] = -vl[rows, sel][has_lead]                           # closing = -vL_rel
    return closing


class Prof635Dataset(Dataset):
    def __init__(self, prebake_path, i2_cache_path=None, pedal_label_path=None, inlane_gt_path=None,
                 gate_tau=0.5, vrel_from_a2logits=False, a2_obj_thr=0.5):
        z = np.load(prebake_path, allow_pickle=True)
        self.curr_keys = z["curr_keys"].tolist()
        self.traj = z["traj"]        # (N,40,3)
        self.ego5 = z["ego_5d"]      # (N,5)
        self.target = z["target"]    # (N,40,2) [steer_deg, ax]
        self.mask = z["mask"]        # (N,40)
        self.N = len(self.curr_keys)
        # i2_mem cache (key 정렬)
        self.i2 = None; self.a2 = None
        if i2_cache_path:
            zi = np.load(i2_cache_path, allow_pickle=True)
            k2i = {k: i for i, k in enumerate(zi["keys"].tolist())}
            self.i2 = zi["i2_mem"]
            self.a2 = zi["a2_logits"] if "a2_logits" in zi.files else None  # (N,40,7) gate용
            self.i2_idx = np.array([k2i.get(k, -1) for k in self.curr_keys], dtype=np.int64)
        # pedal label cache (key 정렬)
        self.pedal = None
        if pedal_label_path:
            zp = np.load(pedal_label_path, allow_pickle=True)
            k2p = {k: i for i, k in enumerate(zp["keys"].tolist())}
            self.pedal_arr = zp["pedal"]    # (Np,40)
            self.pedal_idx = np.array([k2p.get(k, -1) for k in self.curr_keys], dtype=np.int64)
        # [진단] GT 상대속도(closing). lead 없으면(NaN) 0. f_2 에 주입해 capacity 격리 테스트.
        # gate BCE target g* = (inlane_dist 유효=lead 인지) & (inlane_closing > gate_tau=접근 중).
        #   collapse 방지: no-lead frame(dist NaN/inf) → g*=0. lead 정속(closing≤τ) → g*=0.
        #   lead 접근(closing>τ) → g*=1. dist 유효 판정 = finite (inlane_gt 빌더가 no-lead 면 inf).
        self.vrel = None; self.gate_tgt = None; self.gate_valid = None
        if inlane_gt_path:
            zv = np.load(inlane_gt_path, allow_pickle=True)
            k2v = {k: i for i, k in enumerate(zv["keys"].tolist())}
            cl = zv["inlane_closing"]; di = zv["inlane_dist"]
            self.vrel = np.array([float(cl[k2v[k]]) if (k in k2v and np.isfinite(cl[k2v[k]])) else 0.0
                                  for k in self.curr_keys], dtype=np.float32)
            gt = np.zeros(self.N, dtype=np.float32); gv = np.zeros(self.N, dtype=np.float32)
            for j, k in enumerate(self.curr_keys):
                if k in k2v:
                    gv[j] = 1.0   # inlane_gt cache 커버 → gate BCE 적용
                    has_lead = np.isfinite(di[k2v[k]])
                    closing = float(cl[k2v[k]]) if np.isfinite(cl[k2v[k]]) else 0.0
                    gt[j] = 1.0 if (has_lead and closing > float(gate_tau)) else 0.0
            self.gate_tgt = gt; self.gate_valid = gv
        # a-pred: vrel 출처 = mf20 cache 의 a2_logits(20f a_2) in-lane closing. 추론=학습 일관.
        #   gate target(g*)은 위 inlane_gt(GT)에서 그대로 계산. vrel(주입값)만 a2_logits 예측으로 교체.
        if vrel_from_a2logits:
            if self.a2 is None:
                raise ValueError("vrel_from_a2logits=True 인데 i2 cache 에 a2_logits 없음")
            # i2_idx 정렬 적용 (curr_keys 순서). 없는 frame(idx<0)은 closing=0.
            cl_all = _a2logits_to_closing(self.a2, obj_thr=a2_obj_thr)      # (N_cache,)
            self.vrel = np.array([float(cl_all[self.i2_idx[i]]) if self.i2_idx[i] >= 0 else 0.0
                                  for i in range(self.N)], dtype=np.float32)

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        out = {
            "traj6":  torch.from_numpy(build_traj6(self.traj[i])),          # (6,3)
            "ego6":   torch.from_numpy(build_ego6(self.ego5[i])),           # (6,)
            "steer_label": torch.tensor(self.target[i, 0, 0] / STEER_NORM, dtype=torch.float32),
            "mask0":  torch.tensor(bool(self.mask[i, 0])),
            "key":    self.curr_keys[i],
        }
        if self.i2 is not None and self.i2_idx[i] >= 0:
            out["i2_mem"] = torch.from_numpy(self.i2[self.i2_idx[i]].astype(np.float32))  # (40,128)
            if self.a2 is not None:
                out["a2_logits"] = torch.from_numpy(self.a2[self.i2_idx[i]].astype(np.float32))  # (40,7)
        else:
            out["i2_mem"] = torch.zeros(40, 128, dtype=torch.float32)
        if "a2_logits" not in out:
            out["a2_logits"] = torch.full((40, 7), -10.0, dtype=torch.float32)  # obj=-10 → gate≈0
        if self.pedal is not None or hasattr(self, "pedal_idx"):
            pi = self.pedal_idx[i] if hasattr(self, "pedal_idx") else -1
            out["pedal_label"] = torch.tensor(
                float(self.pedal_arr[pi, 0]) if pi >= 0 else 0.0, dtype=torch.float32)
        if self.vrel is not None:
            out["gt_vrel"] = torch.tensor(self.vrel[i], dtype=torch.float32)
        if self.gate_tgt is not None:
            out["gate_tgt"]   = torch.tensor(self.gate_tgt[i], dtype=torch.float32)
            out["gate_valid"] = torch.tensor(self.gate_valid[i], dtype=torch.float32)
        return out

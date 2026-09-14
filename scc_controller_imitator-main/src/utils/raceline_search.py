"""Raceline waypoint extractor for the FROZEN controller_imitator main model
(SelfAttentionControlPredictor_Professor; the A1_E2E runtime ckpt
`model_0.05898784_202_...` == model #635, config steer_accel_brake_635.json).

I/O confirmed against authoritative training code
(controller_imitator/train_multi_output_split_loss_visualization_renew.py
::preprocess_data_for_preload) and validated on the model's own validation pkl
(steer corr 0.969, first-step 0.987) — see module docstring tail.

#635 EXACT input (per-sample):
  - raw waypoints are dt=0.1s in ego frame; upsample to dt=0.05s (linear midpoint).
  - select 6 points at input_pred_idx_list[1:] = [1, 6, 11, 16, 21, 26]
    of the 0.05s grid  ->  times [0.05, 0.30, 0.55, 0.80, 1.05, 1.30] s.
  - per-point feature, "absolute_trajectory":
        traj[:,0] = -y_local   (right-hand reorder; lateral, +left becomes -)
        traj[:,1] = +x_local   (longitudinal forward)
        traj[:,2] = abs_time   (== idx*0.05 == the 6 target times)   # #635
    NOTE #654 differs ONLY in channel 2: it uses speed (include_speed_profile=true,
    input_time_to_traj='none'). Channels 0/1 identical. Both traj_dim=3.
  - ego state (6): [vx, vy, ax, ay, tire_angle(rad), yaw_rate].
  - output: out[...,0]*40 -> steer (deg); out[...,1] -> accel_brake pedal (-1..1).

dt=0.1s raceline file format (snailpit *_dt0p1s.csv): '#'-comment header, then
semicolon cols [t(s), x, y, yaw_rad]. No speed -> derived from spacing/dt.
Optional 6-vertex pit polygon may be appended as the LAST 6 data rows (not dt-spaced);
ego inside polygon -> nearest search over pit waypoints, else track waypoints.

other behavior (advance, ego-frame transform) mirrors
scripts/closed_loop_node._sample_raceline_traj.
"""
from __future__ import annotations
import os
import re
import numpy as np

DT = 0.1  # raceline time step (s)

# controller_imitator #635/#654 input_pred_idx_list[1:] on the 0.05s grid.
PROF_TARGET_TIMES = (0.05, 0.30, 0.55, 0.80, 1.05, 1.30)

# Fallback nearest-search cutoff (exclusive) per track when not in filename.
# tail rows >= cutoff are the pit-merge overlap and are excluded from nearest search.
SEARCH_CUTOFF = {"speedway": 1088, "sangam": 1300}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _is_polygon_row(prev_t: float, t: float) -> bool:
    """Heuristic: a polygon row breaks the uniform 0.1s time grid (dt jump)."""
    return abs((t - prev_t) - DT) > 0.5 * DT


def load_raceline_dt(path: str, has_polygon: bool | None = None):
    """Load a dt0p1s raceline file.

    Returns (raceline (N,4)[t,x,y,yaw], polygon (6,2) or None).
    If `has_polygon` is None, auto-detect a trailing block whose time column
    breaks the uniform 0.1s grid (the appended polygon, not dt-sampled).
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            v = [float(p) for p in line.split(";") if p.strip()]
            if len(v) >= 3:
                rows.append(v[:4] if len(v) >= 4 else v + [0.0] * (4 - len(v)))
    arr = np.array(rows, dtype=np.float64)
    polygon = None

    if arr.shape[0] >= 7:
        # detect trailing rows that break the uniform dt grid
        t = arr[:, 0]
        break_at = None
        for i in range(arr.shape[0] - 1, 0, -1):
            if _is_polygon_row(t[i - 1], t[i]):
                break_at = i
                break
        n_tail = arr.shape[0] - break_at if break_at is not None else 0
        if has_polygon is True or (has_polygon is None and n_tail == 6):
            if break_at is not None and n_tail == 6:
                polygon = arr[break_at:, 1:3].copy()
                arr = arr[:break_at]
    return arr, polygon


def parse_cutoffs_from_filename(path: str):
    """Two 5-digit integers after the last underscore: '..._XXXXXYYYYY.csv'
    (XXXXX = pit-limit index, YYYYY = searchable cutoff index).
    Returns (pit_limit, searchable_cutoff) or (None, None) if absent.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    last = stem.rsplit("_", 1)[-1]
    m = re.fullmatch(r"(\d{5})(\d{5})", last)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def resolve_cutoff(path: str, track: str | None = None,
                   searchable_cutoff: int | None = None) -> int:
    """Resolve searchable cutoff: explicit arg > filename > track default."""
    if searchable_cutoff is not None:
        return int(searchable_cutoff)
    _, yy = parse_cutoffs_from_filename(path)
    if yy is not None:
        return yy
    key = (track or os.path.basename(path)).lower()
    for name, cut in SEARCH_CUTOFF.items():
        if name in key:
            return cut
    raise ValueError(f"cutoff unknown for {path!r}; pass searchable_cutoff or track")


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def point_in_polygon(px: float, py: float, poly: np.ndarray) -> bool:
    """ray-casting; poly (M,2)."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i, 0], poly[i, 1]
        xj, yj = poly[j, 0], poly[j, 1]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def _derive_speed(xy: np.ndarray) -> np.ndarray:
    """speed[i] = ||p[i+1]-p[i]|| / DT (last copies previous)."""
    if len(xy) < 2:
        return np.zeros(len(xy), dtype=np.float64)
    d = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])) / DT
    return np.concatenate([d, d[-1:]])


def nearest_searchable_idx(xy: np.ndarray, ex: float, ey: float,
                           cutoff: int, polygon: np.ndarray | None) -> int:
    """nearest raceline idx among candidates.
    candidates = (idx < cutoff) AND (polygon given -> wp_in_pit == ego_in_pit).
    """
    N = len(xy)
    cand = np.zeros(N, dtype=bool)
    cand[: min(cutoff, N)] = True
    if polygon is not None:
        ego_in = point_in_polygon(ex, ey, polygon)
        wp_in = np.array(
            [point_in_polygon(xy[i, 0], xy[i, 1], polygon) for i in range(N)]
        )
        cand &= (wp_in == ego_in)
        if not cand.any():
            cand[: min(cutoff, N)] = True  # fallback: cutoff only
    d2 = (xy[:, 0] - ex) ** 2 + (xy[:, 1] - ey) ** 2
    d2[~cand] = np.inf
    return int(np.argmin(d2))


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def sample_traj_ego(raceline: np.ndarray, ex: float, ey: float, e_yaw: float,
                    cutoff: int, polygon: np.ndarray | None = None,
                    target_times=PROF_TARGET_TIMES) -> np.ndarray:
    """Sample the raceline at `target_times` (s) ahead of the nearest point,
    in ego frame.  Returns (len(target_times), 3) = [x_fwd, y_left, speed].

    raceline is 0.1s-uniform, so time t maps to fractional row idx = i0 + t/DT,
    with linear interpolation between rows (parity with the 0.05s upsample the
    model was trained on: a midpoint at t=0.05 equals 0.5*(row0+row1)).
    """
    xy = raceline[:, 1:3]
    spd = _derive_speed(xy)
    N = len(xy)
    i0 = nearest_searchable_idx(xy, ex, ey, cutoff, polygon)
    c, s = np.cos(-e_yaw), np.sin(-e_yaw)
    out = np.zeros((len(target_times), 3), dtype=np.float32)
    for k, t in enumerate(target_times):
        f = i0 + t / DT
        i_lo = int(np.floor(f))
        frac = f - i_lo
        i_lo = min(i_lo, N - 1)
        i_hi = min(i_lo + 1, N - 1)
        px = xy[i_lo, 0] * (1 - frac) + xy[i_hi, 0] * frac
        py = xy[i_lo, 1] * (1 - frac) + xy[i_hi, 1] * frac
        v = spd[i_lo] * (1 - frac) + spd[i_hi] * frac
        dx, dy = px - ex, py - ey
        out[k, 0] = c * dx - s * dy   # x_fwd
        out[k, 1] = s * dx + c * dy   # y_left
        out[k, 2] = v
    return out


def to_prof654_traj(traj_ego: np.ndarray, mode: str = "abs_time",
                    target_times=PROF_TARGET_TIMES) -> np.ndarray:
    """ego-frame [x_fwd, y_left, speed] -> model input (N,3).

    Right-hand reorder for both #635 and #654:
        col0 = -y_left   (lateral)
        col1 = +x_fwd    (longitudinal)
    Channel 2 depends on the model:
        mode='abs_time' (#635): col2 = target_times (== idx*0.05)
        mode='speed'    (#654): col2 = speed
    """
    n = traj_ego.shape[0]
    out = np.zeros((n, 3), dtype=np.float32)
    out[:, 0] = -traj_ego[:, 1]
    out[:, 1] = traj_ego[:, 0]
    if mode == "speed":
        out[:, 2] = traj_ego[:, 2]
    elif mode == "abs_time":
        out[:, 2] = np.asarray(target_times, dtype=np.float32)[:n]
    else:
        raise ValueError(f"mode must be 'abs_time' (#635) or 'speed' (#654), got {mode!r}")
    return out


def sample_traj(raceline: np.ndarray, ex: float, ey: float, e_yaw: float,
                cutoff: int, polygon: np.ndarray | None = None,
                mode: str = "abs_time",
                target_times=PROF_TARGET_TIMES) -> np.ndarray:
    """Convenience: ego-frame sample + model reorder in one call.
    Returns the ready-to-feed model trajectory input, shape (len(target_times), 3).
    """
    ego = sample_traj_ego(raceline, ex, ey, e_yaw, cutoff, polygon, target_times)
    return to_prof654_traj(ego, mode=mode, target_times=target_times)

"""Vehicle position tracking on A₂ output (display/logging only; not used by the brake).

Measurements: every contiguous group of bins with σ(A[k,0]) ≥ threshold over all
40 bins.  Bearing comes from the group's peak bin, range from the
probability-weighted distance channel.  Each measurement becomes a bird's-eye
point in the camera frame:

    θ_k = atan((u_k − cx) / fx),  u_k = (k + 0.5)·1280/40,  fx = cx = 640
    x = d·sin θ  (lateral, + = right),   y = d·cos θ  (longitudinal, forward)

fx/cx are the A₂ *training* geometry (90° HFOV).  The physical camera is an
uncalibrated 1280×720 centre crop, so these positions inherit the same
geometry caveat as the distance channel.

Tracks: constant-velocity Kalman filter per vehicle, state [x, y, vx, vy]
(velocities relative to the camera), gated nearest-neighbour association,
confirmed after 3 hits, deleted after 1 s without a hit.  The "lead" is the
nearest confirmed track whose bearing lies inside the in-lane bins 16..25.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .a2_lead import (A2_SCALE_DIST, CH_DIST, CH_OBJ, CH_V_LONG_REL, IN_LANE_BIN0,
                      IN_LANE_BIN1, N_BINS)

IMAGE_W = 1280.0
FX = 640.0
CX = 640.0


def bin_bearing(k: float) -> float:
    """Bearing (rad, + = right) of the centre of bin k in the training geometry."""
    u = (k + 0.5) * IMAGE_W / N_BINS
    return math.atan((u - CX) / FX)


def bearing_to_u(theta: float) -> float:
    """Image column (px) for a bearing; inverse of the bin mapping."""
    return CX + FX * math.tan(theta)


# In-lane wedge = outer edges of bins 16..25.
IN_LANE_THETA = (math.atan((IN_LANE_BIN0 * IMAGE_W / N_BINS - CX) / FX),
                 math.atan((IN_LANE_BIN1 * IMAGE_W / N_BINS - CX) / FX))


@dataclass(frozen=True)
class TrackerConfig:
    obj_threshold: float = 0.5        # σ(logit) for a bin to count as object
    sigma_range_m: float = 6.0        # A₂ range jitter on a parked car: ±6 m at 45 m (10/07 live)
    sigma_bearing_rad: float = 0.05   # lead bin wanders 16..21 on a parked car (10/07 live)
    accel_noise_mps2: float = 3.0     # CV-model process noise
    gate_chi2: float = 9.21           # 2-dof 99 %
    confirm_hits: int = 3
    max_miss_s: float = 1.0
    tentative_max_miss_s: float = 0.3
    init_speed_sigma_mps: float = 10.0
    # Same-vehicle merge: A₂ range on one parked car jumps ±6 m at 45 m (2026-10-07
    # live, static image), which otherwise spawns a second track for the same car.
    merge_lateral_m: float = 2.5
    merge_range_frac: float = 0.25


@dataclass(frozen=True)
class Measurement:
    peak_bin: int
    probability: float
    distance_m: float
    v_rel_a2_mps: float
    x_m: float
    y_m: float
    theta_rad: float


def measurements_from_a2(raw: np.ndarray, threshold: float) -> list[Measurement]:
    a = np.asarray(raw, dtype=np.float64)
    prob = 1.0 / (1.0 + np.exp(-np.clip(a[:, CH_OBJ], -80.0, 80.0)))
    out = []
    k = 0
    while k < N_BINS:
        if prob[k] < threshold:
            k += 1
            continue
        start = k
        while k < N_BINS and prob[k] >= threshold:
            k += 1
        sl = slice(start, k)
        peak = start + int(np.argmax(prob[sl]))
        w = prob[sl] / prob[sl].sum()
        dist = float(np.sum(a[sl, CH_DIST] * A2_SCALE_DIST * w))
        if not (math.isfinite(dist) and dist > 0.0):
            continue
        v_rel = float(np.sum(a[sl, CH_V_LONG_REL] * 20.0 * w))
        theta = bin_bearing(peak)
        out.append(Measurement(peak, float(prob[peak]), dist, v_rel,
                               dist * math.sin(theta), dist * math.cos(theta), theta))
    return out


@dataclass
class Track:
    track_id: int
    x: np.ndarray            # [x, y, vx, vy]
    P: np.ndarray
    t_last: float
    t_hit: float
    hits: int = 1
    confirmed: bool = False
    trail: list = field(default_factory=list)   # [(t, x, y)]

    @property
    def range_m(self) -> float:
        return float(math.hypot(self.x[0], self.x[1]))

    @property
    def bearing_rad(self) -> float:
        return float(math.atan2(self.x[0], self.x[1]))

    def as_dict(self) -> dict:
        return {
            "id": self.track_id, "confirmed": self.confirmed, "hits": self.hits,
            "x_m": float(self.x[0]), "y_m": float(self.x[1]),
            "vx_mps": float(self.x[2]), "vy_mps": float(self.x[3]),
            "range_m": self.range_m, "bearing_deg": math.degrees(self.bearing_rad),
            "age_since_hit_s": self.t_last - self.t_hit,
        }


class VehicleTracker:
    def __init__(self, config: TrackerConfig = TrackerConfig(), trail_s: float = 5.0):
        self.config = config
        self.trail_s = trail_s
        self.tracks: list[Track] = []
        self._next_id = 1
        self._t = None

    def reset(self) -> None:
        self.tracks.clear()
        self._t = None

    def _meas_cov(self, m: Measurement) -> np.ndarray:
        c = self.config
        s, co = math.sin(m.theta_rad), math.cos(m.theta_rad)
        # d(x,y)/d(range, bearing)
        J = np.array([[s, m.distance_m * co], [co, -m.distance_m * s]])
        return J @ np.diag([c.sigma_range_m ** 2, c.sigma_bearing_rad ** 2]) @ J.T

    def _predict(self, track: Track, dt: float) -> None:
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = self.config.accel_noise_mps2 ** 2
        G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
        track.x = F @ track.x
        track.P = F @ track.P @ F.T + G @ (q * np.eye(2)) @ G.T

    def update(self, t_s: float, measurements: list[Measurement]) -> dict:
        c = self.config
        if self._t is not None and t_s <= self._t:
            self.reset()                       # clock jump / bag loop
        dt = 0.0 if self._t is None else t_s - self._t
        self._t = t_s
        for tr in self.tracks:
            if dt > 0.0:
                self._predict(tr, dt)
            tr.t_last = t_s

        H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        pairs = []
        for ti, tr in enumerate(self.tracks):
            for mi, m in enumerate(measurements):
                S = H @ tr.P @ H.T + self._meas_cov(m)
                r = np.array([m.x_m, m.y_m]) - H @ tr.x
                d2 = float(r @ np.linalg.solve(S, r))
                if d2 <= c.gate_chi2:
                    pairs.append((d2, ti, mi))
        used_t, used_m = set(), set()
        for d2, ti, mi in sorted(pairs):
            if ti in used_t or mi in used_m:
                continue
            used_t.add(ti)
            used_m.add(mi)
            tr, m = self.tracks[ti], measurements[mi]
            R = self._meas_cov(m)
            S = H @ tr.P @ H.T + R
            K = tr.P @ H.T @ np.linalg.inv(S)
            tr.x = tr.x + K @ (np.array([m.x_m, m.y_m]) - H @ tr.x)
            tr.P = (np.eye(4) - K @ H) @ tr.P
            tr.hits += 1
            tr.t_hit = t_s
            if tr.hits >= c.confirm_hits:
                tr.confirmed = True

        for mi, m in enumerate(measurements):
            if mi in used_m:
                continue
            P = np.zeros((4, 4))
            P[:2, :2] = self._meas_cov(m)
            P[2, 2] = P[3, 3] = c.init_speed_sigma_mps ** 2
            self.tracks.append(Track(self._next_id, np.array([m.x_m, m.y_m, 0.0, 0.0]),
                                     P, t_s, t_s))
            self._next_id += 1

        # Merge duplicates of one vehicle: keep the track with more hits.
        alive = sorted(self.tracks, key=lambda tr: (-tr.hits, tr.track_id))
        merged: list[Track] = []
        for tr in alive:
            dup = any(
                abs(tr.x[0] - keep.x[0]) <= c.merge_lateral_m
                and abs(tr.x[1] - keep.x[1]) <= c.merge_range_frac * max(keep.x[1], 1.0)
                for keep in merged)
            if not dup:
                merged.append(tr)
        self.tracks = sorted(merged, key=lambda tr: tr.track_id)

        kept = []
        for tr in self.tracks:
            limit = c.max_miss_s if tr.confirmed else c.tentative_max_miss_s
            if t_s - tr.t_hit <= limit:
                tr.trail.append((t_s, float(tr.x[0]), float(tr.x[1])))
                while tr.trail and t_s - tr.trail[0][0] > self.trail_s:
                    tr.trail.pop(0)
                kept.append(tr)
        self.tracks = kept

        lead = self.lead()
        return {
            "measurements": [
                {"bin": m.peak_bin, "p": m.probability, "d_m": m.distance_m,
                 "x_m": m.x_m, "y_m": m.y_m} for m in measurements],
            "tracks": [tr.as_dict() for tr in self.tracks if tr.confirmed],
            "lead_track": lead.as_dict() if lead is not None else None,
        }

    def lead(self) -> Track | None:
        lo, hi = IN_LANE_THETA
        cands = [tr for tr in self.tracks
                 if tr.confirmed and lo <= tr.bearing_rad <= hi and tr.x[1] > 0.0]
        return min(cands, key=lambda tr: tr.range_m) if cands else None

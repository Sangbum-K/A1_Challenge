"""ḋ: closing speed from the A₂ distance channel alone (식 0 교차 확인).

Least-squares slope of d(t) over a sliding time window.  Used only through
v_rel,used = min(v_rel, ḋ), so a wrong velocity channel can never make the
brake weaker than the distance trend says it should be.
"""

from __future__ import annotations

from collections import deque
import math

import numpy as np


class DistanceRate:
    def __init__(self, window_s: float = 1.0, min_samples: int = 8,
                 min_span_s: float = 0.5):
        if window_s <= 0.0 or min_samples < 2:
            raise ValueError("window_s must be > 0 and min_samples >= 2")
        self.window_s = window_s
        self.min_samples = min_samples
        self.min_span_s = min_span_s
        self._samples: deque[tuple[float, float]] = deque()

    def reset(self) -> None:
        self._samples.clear()

    def update(self, t_s: float, distance_m: float | None) -> float | None:
        """Add one sample (None = no valid distance) and return ḋ in m/s or None."""
        if distance_m is not None and math.isfinite(distance_m):
            if self._samples and t_s <= self._samples[-1][0]:
                # Timestamps went backwards (bag loop / clock jump): start over.
                self._samples.clear()
            self._samples.append((t_s, distance_m))
        while self._samples and t_s - self._samples[0][0] > self.window_s:
            self._samples.popleft()
        return self.rate()

    def rate(self) -> float | None:
        if len(self._samples) < self.min_samples:
            return None
        t = np.fromiter((s[0] for s in self._samples), dtype=np.float64)
        d = np.fromiter((s[1] for s in self._samples), dtype=np.float64)
        if t[-1] - t[0] < self.min_span_s:
            return None
        t = t - t.mean()
        denom = float(np.dot(t, t))
        if denom <= 0.0:
            return None
        return float(np.dot(t, d - d.mean()) / denom)

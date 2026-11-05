"""Vehicle tracker on synthetic A₂ outputs."""

from __future__ import annotations

import math

import numpy as np
import pytest

from a1_brake.a2_lead import A2_SCALE_DIST, A2_SCALE_V
from a1_brake.tracking import (IN_LANE_THETA, TrackerConfig, VehicleTracker, bearing_to_u,
                               bin_bearing, measurements_from_a2)


def frame(objects, background=-6.0):
    """objects: list of (bins, distance_m, v_rel)."""
    raw = np.zeros((40, 7), dtype=np.float64)
    raw[:, 0] = background
    for bins, dist, v_rel in objects:
        for k in bins:
            raw[k, 0] = 3.0
            raw[k, 6] = dist / A2_SCALE_DIST
            raw[k, 3] = v_rel / A2_SCALE_V
    return raw


def test_bin_geometry_matches_training_fov():
    assert bin_bearing(19.5) == pytest.approx(0.0)
    assert math.degrees(bin_bearing(-0.5)) == pytest.approx(-45.0)
    assert bearing_to_u(bin_bearing(10)) == pytest.approx(10.5 * 32)
    assert IN_LANE_THETA[0] < 0.0 < IN_LANE_THETA[1]


def test_groups_become_measurements():
    ms = measurements_from_a2(frame([((19, 20, 21), 25.0, -3.0), ((5, 6), 40.0, 0.0)]), 0.5)
    assert len(ms) == 2
    near = min(ms, key=lambda m: m.distance_m)
    assert near.distance_m == pytest.approx(25.0)
    assert abs(near.x_m) < 1.0 and near.y_m == pytest.approx(25.0, abs=0.1)
    far = max(ms, key=lambda m: m.distance_m)
    assert far.x_m < -15.0


def test_track_follows_approaching_lead_and_estimates_speed():
    tr = VehicleTracker(TrackerConfig())
    rng = np.random.default_rng(0)
    out = None
    for i in range(63):                      # 3 s at 21 Hz, closing at 4 m/s
        t = i / 21.0
        d = 40.0 - 4.0 * t + rng.normal(0, 1.5)
        out = tr.update(t, measurements_from_a2(frame([((19, 20), d, -4.0)]), 0.5))
    lead = out["lead_track"]
    assert lead is not None and lead["confirmed"]
    assert lead["y_m"] == pytest.approx(28.0, abs=3.0)
    assert lead["vy_mps"] == pytest.approx(-4.0, abs=1.5)
    assert len(out["tracks"]) == 1          # one id kept through the noise


def test_adjacent_lane_vehicle_is_tracked_but_not_lead():
    tr = VehicleTracker(TrackerConfig())
    out = None
    for i in range(10):
        out = tr.update(i / 21.0, measurements_from_a2(frame([((27, 28, 29), 20.0, 0.0)]), 0.5))
    assert len(out["tracks"]) == 1
    assert out["lead_track"] is None


def test_track_dropped_after_miss_time():
    tr = VehicleTracker(TrackerConfig(max_miss_s=1.0))
    for i in range(10):
        tr.update(i / 21.0, measurements_from_a2(frame([((20,), 20.0, 0.0)]), 0.5))
    out = tr.update(0.5 + 1.2, [])
    assert out["tracks"] == [] and out["lead_track"] is None


def test_range_jitter_on_one_car_does_not_split_into_two_tracks():
    """10/07 live: one parked car read alternately near 38 m and 50 m."""
    tr = VehicleTracker(TrackerConfig())
    rng = np.random.default_rng(1)
    counts = []
    for i in range(210):                     # 10 s
        d = 44.0 + rng.choice([-6.0, 6.0]) + rng.normal(0, 1.0)
        out = tr.update(i / 21.0, measurements_from_a2(frame([((18, 19, 20), d, 12.0)]), 0.5))
        counts.append(len(out["tracks"]))
    assert max(counts[30:]) == 1
    assert out["lead_track"] is not None

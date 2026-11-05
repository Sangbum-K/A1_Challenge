"""식 0~3 behaviour on synthetic A₂ outputs."""

from __future__ import annotations

import math

import numpy as np
import pytest

from a1_brake.a2_lead import A2_SCALE_DIST, A2_SCALE_V, extract_lead
from a1_brake.brake import BrakeConfig, BrakeController, config_from_dict
from a1_brake.closing_rate import DistanceRate
from a1_brake.ego_speed import decode_wheel_info
from a1_brake.gate import GateConfig, RuleGate
from a1_brake.p5 import P5Config, min_gap_eff, p5_accel


def a2_frame(*, bin_=20, logit=3.0, distance=20.0, v_rel=0.0, background=-6.0):
    raw = np.zeros((40, 7), dtype=np.float32)
    raw[:, 0] = background
    raw[:, 6] = 60.0 / A2_SCALE_DIST
    raw[bin_, 0] = logit
    raw[bin_, 3] = v_rel / A2_SCALE_V
    raw[bin_, 6] = distance / A2_SCALE_DIST
    return raw


def test_lead_is_argmax_logit_inside_bins_16_to_25():
    raw = a2_frame(bin_=22, logit=2.0, distance=18.0, v_rel=-4.0)
    raw[5, 0] = 9.0   # stronger object outside the in-lane bins is ignored
    lead = extract_lead(raw)
    assert lead.bin == 22
    assert lead.score_logit == pytest.approx(2.0)
    assert lead.distance_m == pytest.approx(18.0, abs=1e-5)
    assert lead.v_rel_mps == pytest.approx(-4.0, abs=1e-5)
    assert lead.distance_valid


def test_negative_distance_is_not_a_lead():
    lead = extract_lead(a2_frame(distance=-3.0))
    assert not lead.distance_valid


def test_gate_matches_senior_rule():
    gate = RuleGate(GateConfig())
    g_raw, g = gate.update(0.3)
    assert g_raw == pytest.approx(0.5)
    assert g == pytest.approx(0.3 * 0.5)
    g_raw2, g2 = gate.update(2.0)
    assert g_raw2 == pytest.approx(1 / (1 + math.exp(-(2.0 - 0.3) * 5.0)))
    assert g2 == pytest.approx(0.7 * g + 0.3 * g_raw2)


def test_min_gap_examples_from_design():
    assert min_gap_eff(0.0) == pytest.approx(7.0)
    assert min_gap_eff(10.0) == pytest.approx(13.59, abs=0.01)
    assert min_gap_eff(15.0) == pytest.approx(15.32, abs=0.01)


def test_parked_ego_never_requests_deceleration():
    for d in (3.0, 8.0, 20.0, 50.0):
        assert p5_accel(ego_speed_mps=0.0, gap_m=d, lead_speed_mps=0.0).accel_mps2 == 0.0


def test_p5_is_monotone_in_gap_for_a_stopped_lead():
    v = 30.0 / 3.6
    accels = [p5_accel(ego_speed_mps=v, gap_m=d, lead_speed_mps=0.0).accel_mps2
              for d in (10, 20, 25, 30, 35, 40, 60)]
    assert all(a <= b + 1e-9 for a, b in zip(accels, accels[1:]))
    assert accels[0] < -3.0 and accels[-1] == 0.0
    # 1 s window can at most stop the car: a_P5 >= -v / t_H
    assert min(accels) >= -v / P5Config().horizon_s - 1e-9


def test_controller_stationary_ego_main_zero_what_if_brakes():
    ctl = BrakeController(BrakeConfig())
    out = None
    for i in range(30):
        out = ctl.step(a2_frame(distance=18.0, v_rel=0.0), i / 21.0, ego_speed_mps=0.0)
    assert out["gate"]["value"] > 0.99
    assert out["main"]["acc_cmd_mps2"] == 0.0
    by_kph = {row["kph"]: row for row in out["what_if"]}
    assert by_kph[30.0]["a_p5_mps2"] < 0.0
    assert by_kph[30.0]["acc_cmd_mps2"] == pytest.approx(-3.0)  # clamped to ACC_Cmd range
    assert by_kph[30.0]["a_final_mps2"] < -3.0


def test_gate_scales_brake_and_no_lead_means_no_brake():
    ctl = BrakeController(BrakeConfig())
    weak = ctl.step(a2_frame(logit=-5.0, distance=15.0), 0.0, 0.0, plan_speed_mps=8.0)
    assert weak["gate"]["value"] < 0.01
    assert weak["main"]["a_p5_mps2"] < 0.0
    assert abs(weak["main"]["a_final_mps2"]) < 0.1


def test_vrel_used_takes_the_faster_closing_value():
    ctl = BrakeController(BrakeConfig(vrel_mode="min"))
    out = None
    # A2 says +10 m/s (receding) while distance shrinks at 5 m/s.
    for i in range(25):
        t = i / 21.0
        out = ctl.step(a2_frame(distance=30.0 - 5.0 * t, v_rel=10.0), t, 0.0)
    lead = out["lead"]
    assert lead["ddot_mps"] == pytest.approx(-5.0, abs=0.05)
    assert lead["v_rel_used_mps"] == pytest.approx(-5.0, abs=0.05)
    assert lead["v_rel_source"] == "ddot"
    assert lead["lead_speed_mps"] == 0.0   # max(0, 0 + -5)


def test_a2_mode_ignores_ddot():
    ctl = BrakeController(BrakeConfig(vrel_mode="a2"))
    out = None
    for i in range(25):
        t = i / 21.0
        out = ctl.step(a2_frame(distance=30.0 - 5.0 * t, v_rel=10.0), t, 0.0)
    assert out["lead"]["v_rel_used_mps"] == pytest.approx(10.0, abs=1e-4)


def test_f1_combination_only_adds_missing_brake():
    cfg = BrakeConfig(a_f1_mps2=-6.0)
    ctl = BrakeController(cfg)
    out = None
    for i in range(30):
        out = ctl.step(a2_frame(distance=40.0), i / 21.0, 0.0, plan_speed_mps=30 / 3.6)
    # P5 wants less braking than F1 already gives: δ = 0, F1 unchanged.
    assert out["main"]["a_p5_mps2"] > -6.0
    assert out["main"]["delta_mps2"] == 0.0
    assert out["main"]["a_final_mps2"] == -6.0


def test_distance_rate_needs_span_and_samples():
    rate = DistanceRate(window_s=1.0, min_samples=8)
    assert rate.update(0.0, 20.0) is None
    for i in range(1, 25):
        value = rate.update(i * 0.05, 20.0 + 2.0 * i * 0.05)
    assert value == pytest.approx(2.0)
    rate.update(10.0, None)   # long gap drops old samples
    assert rate.rate() is None


def test_config_round_trip_from_yaml_dict():
    cfg = config_from_dict({"p5": {"horizon_s": 0.5}, "gate": {"ema": 0.5},
                            "what_if_kph": [20, 40]})
    assert cfg.p5.horizon_s == 0.5 and cfg.gate.ema == 0.5
    assert cfg.what_if_kph == (20.0, 40.0)


def test_wheel_info_decode():
    right_raw, left_raw = 123, 456   # 12.3, 45.6 km/h
    word = right_raw | (left_raw << 24)
    right, left = decode_wheel_info(word.to_bytes(8, "little"))
    assert right == pytest.approx(12.3) and left == pytest.approx(45.6)


def test_displayed_p5_intermediates_are_consistent():
    """The equation column prints these numbers; they must reproduce a_P5."""
    ctl = BrakeController(BrakeConfig())
    out = None
    for i in range(10):
        out = ctl.step(a2_frame(distance=35.0, v_rel=-2.0), i / 21.0, 0.0,
                       plan_speed_mps=50 / 3.6)
    main, prof, lead = out["main"], out["main"]["profile"], out["lead"]
    v = 50 / 3.6
    assert main["min_gap_eff_m"] == pytest.approx(7 + 6 * math.log1p(v / 5))
    assert prof["x_h_m"] == pytest.approx(v * prof["t_h_s"])
    assert prof["lead_h_m"] == pytest.approx(35.0 + lead["lead_speed_mps"] * prof["t_h_s"], abs=1e-4)
    assert prof["v_safe_h_mps"] == pytest.approx(
        max(0.0, (prof["gap_h_m"] - main["min_gap_eff_m"]) / 2.0))
    assert prof["u0_mps"] == pytest.approx(v)
    assert main["a_p5_mps2"] == pytest.approx((prof["uh_mps"] - prof["u0_mps"]) / prof["t_h_s"])
    g = out["gate"]
    assert g["value"] == pytest.approx(0.7 * g["prev"] + 0.3 * g["raw"])
    assert main["a_final_mps2"] == pytest.approx(g["value"] * main["delta_mps2"])


def test_relative_speed_at_planning_speed():
    ctl = BrakeController(BrakeConfig(vrel_mode="a2"))
    out = ctl.step(a2_frame(distance=30.0, v_rel=3.0), 0.0, 0.0, plan_speed_mps=50 / 3.6)
    # Parked camera: A2 v_rel is the lead's own speed; at 50 km/h ego the gap closes.
    assert out["lead"]["lead_speed_mps"] == pytest.approx(3.0, abs=1e-4)
    assert out["main"]["lead_rel_speed_at_plan_mps"] == pytest.approx(3.0 - 50 / 3.6, abs=1e-4)
    none = ctl.step(a2_frame(distance=-1.0), 0.05, 0.0, plan_speed_mps=50 / 3.6)
    assert none["main"]["lead_rel_speed_at_plan_mps"] is None


def test_track_mode_uses_tracked_lead_speed_and_falls_back():
    ctl = BrakeController(BrakeConfig())          # default vrel_mode = track
    assert ctl.config.vrel_mode == "track"
    out = ctl.step(a2_frame(distance=30.0, v_rel=12.0), 0.0, 0.0, 50 / 3.6, v_rel_track_mps=-3.0)
    assert out["lead"]["v_rel_used_mps"] == pytest.approx(-3.0)
    assert out["lead"]["v_rel_source"] == "track"
    # No track yet and no ḋ history: fall back to A2.
    fresh = BrakeController(BrakeConfig()).step(a2_frame(distance=30.0, v_rel=12.0), 0.0, 0.0)
    assert fresh["lead"]["v_rel_source"] == "a2"

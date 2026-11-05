"""식 2 · P5 감속식: the senior p5 SCC target generator, read out as m/s².

``replan_profile`` is the same function as
``A1_FSDS_F2_Validation/src/fsds_f2_validation/p5_expert.py`` and is checked
against the bundled senior script ``senior_code/scripts/gen_profile_pedal_labels_recalc.py``
by ``tests/test_p5_matches_senior.py`` (atol 1e-12).  Do not edit it without
re-running that test.

Real-car changes (F2_RuleBased_Design.html 5장):
  - ego path x_i comes from a target-speed plan v_ref (F1 is not decided yet,
    so v_ref,i = v_ego), not from raceline waypoints;
  - lead future position is constant velocity, L_i = d + v_lead * t_i;
  - the plan is read as a 1 s mean acceleration (u_H - u_0) / t_H instead of
    the first step, which is clipped at -D_max = -20 m/s² by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class P5Config:
    time_headway_s: float = 2.0       # T_h, senior p5
    standstill_gap_m: float = 7.0     # s_min, senior p5
    log_alpha_m: float = 6.0          # α, SCC_LOG_ALPHA
    log_v0_mps: float = 5.0           # v_0, SCC_LOG_V0
    max_accel_mps2: float = 5.0       # A_max, senior A_MAX
    max_decel_mps2: float = 20.0      # D_max, senior MAX_DECEL / -A_MIN
    dt_s: float = 0.1                 # Δt, senior label grid
    steps: int = 40                   # N, 4 s plan
    horizon_s: float = 1.0            # t_H, real-car read-out window


@dataclass(frozen=True)
class P5Output:
    accel_mps2: float          # a_P5, negative = decelerate
    min_gap_eff_m: float       # s_min,eff at the planning ego speed
    safe_speed_now_mps: float  # v_safe,0 = max(0, (d - s_min,eff) / T_h)
    planned_speed_mps: np.ndarray   # u_i after both senior passes
    # Intermediate values of 식 2-2 .. 2-4, for display/logging only.
    time_s: np.ndarray              # t_i
    ref_speed_mps: np.ndarray       # v_ref,i
    ref_position_m: np.ndarray      # x_i
    lead_position_m: np.ndarray     # L_i
    safe_speed_mps: np.ndarray      # v_safe,i = max(0, (L_i - x_i - s_min,eff) / T_h)
    horizon_index: int              # H


def _validate_series(*values: np.ndarray) -> None:
    if not values:
        raise ValueError("at least one series is required")
    n = len(values[0])
    if n < 2 or any(np.asarray(value).shape != (n,) for value in values):
        raise ValueError("all series must be one-dimensional and have at least two samples")
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError("trajectory contains non-finite values")


def replan_profile(
    cumulative_s_m: np.ndarray,
    time_s: np.ndarray,
    original_speed_mps: np.ndarray,
    lead_s_m: np.ndarray | None,
    ego_speed_mps: float,
    config: P5Config = P5Config(),
) -> np.ndarray:
    """Run the senior target-generation passes, including the ego-speed anchor."""
    s = np.asarray(cumulative_s_m, dtype=np.float64)
    t = np.asarray(time_s, dtype=np.float64)
    original = np.asarray(original_speed_mps, dtype=np.float64)
    _validate_series(s, t, original)
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("trajectory time must be strictly increasing")
    if ego_speed_mps < 0.0 or not math.isfinite(ego_speed_mps):
        raise ValueError("ego speed must be finite and nonnegative")

    planned = original.copy()
    if lead_s_m is not None:
        lead = np.asarray(lead_s_m, dtype=np.float64)
        _validate_series(s, t, original, lead)
        s_min_eff = config.standstill_gap_m + config.log_alpha_m * math.log1p(
            ego_speed_mps / config.log_v0_mps
        )
        safe = np.maximum(0.0, (lead - s - s_min_eff) / config.time_headway_s)
        planned = np.minimum(planned, safe)

    # Senior backward feasibility pass: start braking before the constrained point.
    for index in range(len(planned) - 2, -1, -1):
        dt = t[index + 1] - t[index]
        planned[index] = min(
            planned[index], planned[index + 1] + config.max_decel_mps2 * dt
        )

    # Dataset correction: start from measured ego speed and respect acceleration bounds.
    planned[0] = ego_speed_mps
    for index in range(1, len(planned)):
        dt = t[index] - t[index - 1]
        low = planned[index - 1] - config.max_decel_mps2 * dt
        high = planned[index - 1] + config.max_accel_mps2 * dt
        planned[index] = min(max(planned[index], low), high)
    return np.maximum(planned, 0.0)


def min_gap_eff(ego_speed_mps: float, config: P5Config = P5Config()) -> float:
    """s_min,eff = s_min + α·ln(1 + v_ego / v_0)."""
    return config.standstill_gap_m + config.log_alpha_m * math.log1p(
        max(ego_speed_mps, 0.0) / config.log_v0_mps)


def p5_accel(
    *,
    ego_speed_mps: float,
    gap_m: float,
    lead_speed_mps: float,
    config: P5Config = P5Config(),
    ref_speed_mps: np.ndarray | None = None,
) -> P5Output:
    """a_P5 for one frame: plan 4 s with a constant-velocity lead, read 1 s mean.

    ``ref_speed_mps`` is the F1 target-speed plan v_ref,i (N samples).  Until F1
    exists it defaults to holding the current speed, v_ref,i = v_ego.
    """
    if not (math.isfinite(ego_speed_mps) and ego_speed_mps >= 0.0):
        raise ValueError("ego speed must be finite and nonnegative")
    if not (math.isfinite(gap_m) and math.isfinite(lead_speed_mps)):
        raise ValueError("gap and lead speed must be finite")
    t = np.arange(config.steps, dtype=np.float64) * config.dt_s
    if ref_speed_mps is None:
        ref = np.full(config.steps, ego_speed_mps, dtype=np.float64)
    else:
        ref = np.asarray(ref_speed_mps, dtype=np.float64)
    # x_0 = 0, x_i = x_{i-1} + v_ref,{i-1}·Δt
    x = np.concatenate(([0.0], np.cumsum(ref[:-1] * config.dt_s)))
    lead = gap_m + max(0.0, lead_speed_mps) * t
    planned = replan_profile(x, t, ref, lead, ego_speed_mps, config)
    h = max(1, min(config.steps - 1, int(round(config.horizon_s / config.dt_s))))
    accel = float((planned[h] - planned[0]) / (t[h] - t[0]))
    if abs(accel) < 1e-6:
        accel = 0.0  # float noise from the passes, not a request
    s_eff = min_gap_eff(ego_speed_mps, config)
    return P5Output(
        accel_mps2=accel,
        min_gap_eff_m=s_eff,
        safe_speed_now_mps=max(0.0, (gap_m - s_eff) / config.time_headway_s),
        planned_speed_mps=planned,
        time_s=t,
        ref_speed_mps=ref,
        ref_position_m=x,
        lead_position_m=lead,
        safe_speed_mps=np.maximum(0.0, (lead - x - s_eff) / config.time_headway_s),
        horizon_index=h,
    )

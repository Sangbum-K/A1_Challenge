"""식 0 → 식 1 → 식 2 → 식 3: one A₂ frame in, one brake request (m/s²) out.

    lead      = extract_lead(A)                       식 0
    ḋ         = LS slope of d over the last 1 s       식 0 교차 확인
    v_rel,used= tracked lead vy (vrel_mode="track", default) or min(v_rel, ḋ) ("min")
    g         = rule gate(max logit, bins 16..25)     식 1
    v_lead    = max(0, v_ego,actual + v_rel,used)
    a_P5      = P5 plan read over 1 s                 식 2
    δ         = min(0, a_P5 − a_F1)
    a_final   = a_F1 + g·δ                             식 3
    ACC_Cmd   = clip(a_final, −3, +1)                 실차 DBC 0x157 range

The output is computed and logged only.  Nothing here talks to the vehicle.

Two ego speeds are kept apart on purpose:
  - actual ego speed: what the camera is really doing.  A₂'s relative
    velocity is relative to *this*, so v_lead always uses it.
  - planning ego speed: the speed P5 plans for.  Normally equal to the actual
    speed.  With a parked ego (v=0) P5 can never ask for deceleration
    (the plan starts at 0 and is floored at 0), so a stationary field test
    also evaluates "what-if" planning speeds against the same observed lead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

import numpy as np

from .a2_lead import LeadObservation, extract_lead
from .closing_rate import DistanceRate
from .gate import GateConfig, RuleGate
from .p5 import P5Config, p5_accel


VREL_MODES = ("track", "min", "a2", "ddot")


@dataclass(frozen=True)
class BrakeConfig:
    p5: P5Config = field(default_factory=P5Config)
    gate: GateConfig = field(default_factory=GateConfig)
    # track: Kalman vy of the tracked lead (falls back to ḋ, then A₂).  A₂ ch3 reads
    # parked cars at +10..+15 m/s on the real camera (bias differs per scene), so it
    # is not used unless asked for.  min: min(v_rel, ḋ) | a2 only | ḋ only.
    vrel_mode: str = "track"
    ddot_window_s: float = 1.0
    ddot_min_samples: int = 8
    a_f1_mps2: float = 0.0           # F1 not decided: 0 = hold current speed
    acc_cmd_min_mps2: float = -3.0   # ACC_Cmd (0x157) physical range
    acc_cmd_max_mps2: float = 1.0
    what_if_kph: tuple[float, ...] = (10.0, 20.0, 30.0, 40.0, 50.0)

    def __post_init__(self):
        if self.vrel_mode not in VREL_MODES:
            raise ValueError(f"vrel_mode must be one of {VREL_MODES}")


def _finite_or_none(x):
    return float(x) if x is not None and math.isfinite(x) else None


class BrakeController:
    def __init__(self, config: BrakeConfig = BrakeConfig()):
        self.config = config
        self.gate = RuleGate(config.gate)
        self.rate = DistanceRate(config.ddot_window_s, config.ddot_min_samples)

    def reset(self) -> None:
        self.gate.reset()
        self.rate.reset()

    def _command(self, plan_speed: float, lead: LeadObservation, lead_speed: float,
                 g: float, detail: bool = False) -> dict:
        cfg = self.config
        profile = None
        if lead.distance_valid:
            out = p5_accel(ego_speed_mps=plan_speed, gap_m=lead.distance_m,
                           lead_speed_mps=lead_speed, config=cfg.p5)
            a_p5 = out.accel_mps2
            s_eff, v_safe = out.min_gap_eff_m, out.safe_speed_now_mps
            if detail:
                h = out.horizon_index
                profile = {
                    "horizon_index": h,
                    "t_h_s": float(out.time_s[h]),
                    "x_h_m": float(out.ref_position_m[h]),
                    "lead_h_m": float(out.lead_position_m[h]),
                    "gap_h_m": float(out.lead_position_m[h] - out.ref_position_m[h]),
                    "v_safe_h_mps": float(out.safe_speed_mps[h]),
                    "u0_mps": float(out.planned_speed_mps[0]),
                    "uh_mps": float(out.planned_speed_mps[h]),
                    "planned_speed_mps": [round(float(v), 4) for v in out.planned_speed_mps],
                    "safe_speed_mps": [round(float(v), 4) for v in out.safe_speed_mps],
                    "ref_speed_mps": [round(float(v), 4) for v in out.ref_speed_mps],
                }
        else:
            a_p5, s_eff, v_safe = 0.0, None, None
        delta = min(0.0, a_p5 - cfg.a_f1_mps2)
        a_final = cfg.a_f1_mps2 + g * delta
        acc_cmd = min(cfg.acc_cmd_max_mps2, max(cfg.acc_cmd_min_mps2, a_final))
        return {
            "plan_speed_mps": plan_speed,
            # Lead speed relative to an ego driving at the planning speed (negative = closing).
            "lead_rel_speed_at_plan_mps": (lead_speed - plan_speed
                                           if math.isfinite(lead_speed) else None),
            "a_p5_mps2": a_p5,
            "delta_mps2": delta,
            "a_final_mps2": a_final,
            "acc_cmd_mps2": acc_cmd,
            "min_gap_eff_m": s_eff,
            "safe_speed_now_mps": v_safe,
            **({"profile": profile} if detail else {}),
        }

    def step(self, raw_40x7: np.ndarray, t_s: float, ego_speed_mps: float,
             plan_speed_mps: float | None = None,
             v_rel_track_mps: float | None = None) -> dict:
        """Process one A₂ frame.  ``t_s`` is the camera timestamp in seconds."""
        if not (math.isfinite(ego_speed_mps) and ego_speed_mps >= 0.0):
            raise ValueError("actual ego speed must be finite and nonnegative")
        cfg = self.config
        lead = extract_lead(raw_40x7)
        ddot = self.rate.update(t_s, lead.distance_m if lead.distance_valid else None)
        g_prev = self.gate.value
        g_raw, g = self.gate.update(lead.score_logit)

        v_rel = lead.v_rel_mps
        if not lead.distance_valid:
            v_rel_used, vrel_source = math.nan, "none"
        elif cfg.vrel_mode == "track":
            if v_rel_track_mps is not None and math.isfinite(v_rel_track_mps):
                v_rel_used, vrel_source = v_rel_track_mps, "track"
            elif ddot is not None:
                v_rel_used, vrel_source = ddot, "ddot"
            else:
                v_rel_used, vrel_source = v_rel, "a2"
        elif cfg.vrel_mode == "a2" or ddot is None:
            v_rel_used, vrel_source = v_rel, "a2"
        elif cfg.vrel_mode == "ddot":
            v_rel_used, vrel_source = ddot, "ddot"
        else:
            v_rel_used = min(v_rel, ddot)
            vrel_source = "ddot" if ddot < v_rel else "a2"
        lead_speed = (max(0.0, ego_speed_mps + v_rel_used)
                      if math.isfinite(v_rel_used) else math.nan)

        plan = ego_speed_mps if plan_speed_mps is None else plan_speed_mps
        if not (math.isfinite(plan) and plan >= 0.0):
            raise ValueError("planning ego speed must be finite and nonnegative")
        main = self._command(plan, lead, lead_speed, g, detail=True)
        what_if = [
            {"kph": kph, **self._command(kph / 3.6, lead, lead_speed, g)}
            for kph in cfg.what_if_kph
        ]
        return {
            "lead": {
                "bin": lead.bin,
                "score_logit": _finite_or_none(lead.score_logit),
                "probability": lead.probability,
                "distance_m": _finite_or_none(lead.distance_m),
                "distance_valid": lead.distance_valid,
                "v_rel_a2_mps": _finite_or_none(v_rel),
                "v_lat_rel_mps": _finite_or_none(lead.v_lat_rel_mps),
                "ddot_mps": _finite_or_none(ddot),
                "v_rel_track_mps": _finite_or_none(v_rel_track_mps),
                "v_rel_used_mps": _finite_or_none(v_rel_used),
                "v_rel_source": vrel_source,
                "lead_speed_mps": _finite_or_none(lead_speed),
            },
            "gate": {"raw": g_raw, "value": g, "prev": g_prev},
            "ego_speed_mps": ego_speed_mps,
            "a_f1_mps2": cfg.a_f1_mps2,
            "main": main,
            "what_if": what_if,
        }


def config_to_dict(config: BrakeConfig) -> dict:
    return asdict(config)


def config_from_dict(data: dict | None) -> BrakeConfig:
    data = dict(data or {})
    p5 = P5Config(**data.pop("p5", {}) or {})
    gate = GateConfig(**data.pop("gate", {}) or {})
    if "what_if_kph" in data:
        data["what_if_kph"] = tuple(float(v) for v in data["what_if_kph"])
    return BrakeConfig(p5=p5, gate=gate, **data)

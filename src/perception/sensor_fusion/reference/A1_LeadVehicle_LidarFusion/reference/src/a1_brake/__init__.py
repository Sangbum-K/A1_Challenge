"""A₂ → rule gate + P5 → brake request (m/s²).  See docs/DESIGN_KR.md."""

from .a2_lead import LeadObservation, extract_lead
from .brake import BrakeConfig, BrakeController, config_from_dict, config_to_dict
from .gate import GateConfig, RuleGate
from .p5 import P5Config, p5_accel, replan_profile

__all__ = [
    "BrakeConfig", "BrakeController", "GateConfig", "LeadObservation", "P5Config",
    "RuleGate", "config_from_dict", "config_to_dict", "extract_lead", "p5_accel",
    "replan_profile",
]

"""식 1 · 규칙 gate: how much to trust the in-lane lead.

Senior rule deployed before the learned TemporalMLP gate
(scc_controller_imitator-main/docs/MODEL_AUDIT_2026-05-26.md L33-37):

    obj_amax = a2_logits[0, 16:26, 0].max()
    g_raw    = sigmoid((obj_amax - 0.3) * 5.0)
    gate     = 0.7*prev + 0.3*g_raw

θ, κ, β were set on simulator A₂ scores.  Real-camera score distributions
differ, so replay field logs before trusting them.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GateConfig:
    threshold_logit: float = 0.3  # θ, GATE_THRESH
    scale: float = 5.0            # κ, GATE_SCALE
    ema: float = 0.7              # β, GATE_EMA (per A₂ frame)
    initial: float = 0.0          # g_0, real-car choice: no brake until evidence


class RuleGate:
    def __init__(self, config: GateConfig = GateConfig()):
        self.config = config
        self.value = config.initial

    def reset(self) -> None:
        self.value = self.config.initial

    def raw(self, score_logit: float) -> float:
        if not math.isfinite(score_logit):
            return 0.0
        x = (score_logit - self.config.threshold_logit) * self.config.scale
        x = max(-80.0, min(80.0, x))
        return 1.0 / (1.0 + math.exp(-x))

    def update(self, score_logit: float) -> tuple[float, float]:
        """Return (g_raw, g) for one A₂ frame."""
        g_raw = self.raw(score_logit)
        beta = self.config.ema
        self.value = beta * self.value + (1.0 - beta) * g_raw
        return g_raw, self.value

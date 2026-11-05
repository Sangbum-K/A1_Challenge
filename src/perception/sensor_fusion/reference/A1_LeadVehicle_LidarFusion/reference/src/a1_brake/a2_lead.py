"""식 0 · A₂ (40,7) 출력에서 앞차 거리·상대속도 꺼내기.

Channel order and scales are the senior label contract
(A1_FSDS_F2_Validation/src/fsds_f2_validation/contracts.py):

    0 object logit, 1 v_long_abs, 2 v_lat_abs, 3 v_long_rel, 4 v_lat_rel,
    5 proximity exp(-d/20), 6 planar distance;   velocity ×20, distance ×40

The lead bin is the argmax of the channel-0 logit over the in-lane bins 16..25,
the same bins and channel the senior rule gate reads.  There is no threshold
here: how much to trust the lead is the gate's job (식 1).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


N_BINS = 40
N_CH = 7
A2_SCALE_V = 20.0
A2_SCALE_DIST = 40.0
IN_LANE_BIN0 = 16
IN_LANE_BIN1 = 26  # exclusive: bins 16..25

CH_OBJ = 0
CH_V_LONG_REL = 3
CH_V_LAT_REL = 4
CH_DIST = 6


@dataclass(frozen=True)
class LeadObservation:
    score_logit: float       # s_t = max_{k∈16..25} A[k,0]  (gate input)
    bin: int                 # k*
    probability: float       # σ(A[k*,0]), display only
    distance_m: float        # d = 40·A[k*,6]
    v_rel_mps: float         # v_rel = 20·A[k*,3]; negative = closing
    v_lat_rel_mps: float     # 20·A[k*,4], logged only
    distance_valid: bool     # finite and > 0


def sigmoid(x: float) -> float:
    x = max(-80.0, min(80.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def extract_lead(raw: np.ndarray) -> LeadObservation:
    a = np.asarray(raw, dtype=np.float64)
    if a.shape != (N_BINS, N_CH):
        raise ValueError(f"A2 output must be ({N_BINS},{N_CH}), got {a.shape}")
    logits = a[IN_LANE_BIN0:IN_LANE_BIN1, CH_OBJ]
    if not np.isfinite(logits).all():
        return LeadObservation(math.nan, -1, 0.0, math.nan, math.nan, math.nan, False)
    k = IN_LANE_BIN0 + int(np.argmax(logits))
    score = float(a[k, CH_OBJ])
    distance = float(a[k, CH_DIST] * A2_SCALE_DIST)
    v_rel = float(a[k, CH_V_LONG_REL] * A2_SCALE_V)
    v_lat = float(a[k, CH_V_LAT_REL] * A2_SCALE_V)
    valid = math.isfinite(distance) and distance > 0.0 and math.isfinite(v_rel)
    return LeadObservation(
        score_logit=score,
        bin=k,
        probability=sigmoid(score),
        distance_m=distance,
        v_rel_mps=v_rel,
        v_lat_rel_mps=v_lat,
        distance_valid=valid,
    )

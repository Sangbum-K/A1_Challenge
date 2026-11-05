"""Prove the p5 reconstruction reproduces the senior generator exactly.

`replan_profile` lives in the senior repository's label-generation script, which
cannot be imported as a module: it chdirs the process at import time and pulls in
a pedal map CSV that is not on this machine. So the function and the three
constants it depends on are lifted out of the source with `ast` and executed on
their own, and every scenario is run through both implementations.

If this test fails, `a1_brake.p5` has drifted from the senior
policy and the recorded deceleration numbers are no longer theirs.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from a1_brake.p5 import P5Config, replan_profile


# Byte-identical bundled copy (senior_code/COPY_MANIFEST.sha256).
SENIOR_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "senior_code/scripts/gen_profile_pedal_labels_recalc.py"
)
WANTED_CONSTANTS = {"A_MAX", "A_MIN", "MAX_DECEL"}


def load_senior_replan():
    """Execute only replan_profile and its constants, with no side effects."""
    tree = ast.parse(SENIOR_SCRIPT.read_text(encoding="utf-8"))
    kept: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "replan_profile":
            kept.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in WANTED_CONSTANTS for t in node.targets
        ):
            kept.append(node)
    names = {
        node.name if isinstance(node, ast.FunctionDef) else node.targets[0].id
        for node in kept
    }
    if "replan_profile" not in names or not WANTED_CONSTANTS <= names:
        raise AssertionError(f"senior script no longer defines {WANTED_CONSTANTS}/replan_profile")
    namespace: dict = {"np": np, "os": __import__("os")}
    exec(compile(ast.Module(body=kept, type_ignores=[]), str(SENIOR_SCRIPT), "exec"), namespace)
    return namespace["replan_profile"], namespace


def test_constants_match_senior():
    _, namespace = load_senior_replan()
    config = P5Config()
    assert namespace["A_MAX"] == config.max_accel_mps2
    assert -namespace["A_MIN"] == config.max_decel_mps2
    assert namespace["MAX_DECEL"] == config.max_decel_mps2


@pytest.mark.parametrize("seed", range(12))
def test_replan_profile_matches_senior(seed):
    senior_replan, _ = load_senior_replan()
    rng = np.random.default_rng(seed)
    config = P5Config()

    ego_speed = float(rng.uniform(0.0, 30.0))
    steps = 40
    time_s = np.arange(steps, dtype=np.float64) * 0.1
    original_speed = np.clip(rng.normal(20.0, 4.0, steps), 1.0, 25.0)
    cumulative = np.concatenate(([0.0], np.cumsum(original_speed[:-1] * 0.1)))
    gap = float(rng.uniform(3.0, 120.0))
    lead_speed = float(rng.uniform(0.0, 25.0))
    lead_s = gap + lead_speed * time_s

    mine = replan_profile(cumulative, time_s, original_speed, lead_s, ego_speed, config)
    theirs, _ = senior_replan(
        cumulative, time_s, original_speed, lead_s, ego_speed,
        t_headway=config.time_headway_s, s_min=config.standstill_gap_m,
    )
    np.testing.assert_allclose(mine, theirs, rtol=0.0, atol=1e-12)


def test_matches_senior_without_a_lead():
    senior_replan, _ = load_senior_replan()
    config = P5Config()
    time_s = np.arange(40, dtype=np.float64) * 0.1
    original_speed = np.full(40, 18.0)
    cumulative = np.concatenate(([0.0], np.cumsum(original_speed[:-1] * 0.1)))

    mine = replan_profile(cumulative, time_s, original_speed, None, 12.0, config)
    theirs, _ = senior_replan(
        cumulative, time_s, original_speed, np.full(40, np.nan), 12.0,
        t_headway=config.time_headway_s, s_min=config.standstill_gap_m,
    )
    np.testing.assert_allclose(mine, theirs, rtol=0.0, atol=1e-12)

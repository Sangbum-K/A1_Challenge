"""Compare the new runtime core to the unchanged project's Python P5."""
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p5_reference import p5_accel

def test_cpp_matches_reference():
    binary = os.environ.get('A1_P5_CLI', '/tmp/a1-fusion-core/p5_cli')
    rng = np.random.default_rng(42)
    cases = [(0, 10, 0), (50/3.6, 20, 0), (10, 60, 10)]
    cases += list(zip(rng.uniform(0,40,200), rng.uniform(.1,160,200), rng.uniform(-5,45,200)))
    for ego, gap, lead in cases:
        actual = float(subprocess.check_output([binary, str(ego), str(gap), str(lead)]))
        expected = p5_accel(ego_speed_mps=ego, gap_m=gap, lead_speed_mps=lead).accel_mps2
        assert abs(actual-expected)<1e-11

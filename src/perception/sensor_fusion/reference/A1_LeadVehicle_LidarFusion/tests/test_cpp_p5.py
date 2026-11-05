from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/src'))
from a1_brake.p5 import p5_accel

BIN = ROOT / '.build/core/p5_numeric'

def test_against_python_reference():
    rng = np.random.default_rng(20261012)
    cases = [(0., 0., 0.), (10., 13., 4.), (10., 200., 10.)]
    cases += list(zip(rng.uniform(0, 40, 200), rng.uniform(-5, 150, 200), rng.uniform(-5, 40, 200)))
    for ego, gap, lead in cases:
        raw, clamped = map(float, subprocess.check_output([str(BIN), str(ego), str(gap), str(lead)], text=True).split(','))
        expected = p5_accel(ego_speed_mps=ego, gap_m=gap, lead_speed_mps=lead).accel_mps2
        assert raw == pytest.approx(expected, abs=1e-10)
        assert clamped == pytest.approx(np.clip(expected, -3, 0), abs=1e-10)

@pytest.mark.parametrize('args', [['nan','10','1'], ['1','inf','1'], ['-1','10','1'], ['1x','10','1']])
def test_invalid_numeric_input(args):
    result = subprocess.run([str(BIN), *args], capture_output=True)
    assert result.returncode != 0
    assert not result.stdout

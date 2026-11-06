"""Rate arithmetic of lidar_rate_check.py on hand-made timestamps; says nothing about a real LiDAR."""
import importlib.util,math
from pathlib import Path

spec=importlib.util.spec_from_file_location('lidar_rate_check',Path(__file__).resolve().parents[1]/'tools'/'lidar_rate_check.py')
lidar_rate_check=importlib.util.module_from_spec(spec);spec.loader.exec_module(lidar_rate_check)

def test_rates():
    arrivals=[10+i*.1 for i in range(51)]
    received,header=lidar_rate_check.rates(arrivals,[500+i*.1 for i in range(51)])
    assert math.isclose(received,10.) and math.isclose(header,10.)
    # Messages that arrive in bursts still average out; header time shows the sensor's own rate.
    received,header=lidar_rate_check.rates([0,.01,.02,1.,1.01],[0,.5,1.,1.5,2.])
    assert math.isclose(received,4./1.01) and math.isclose(header,2.)

def test_too_few_messages():
    assert lidar_rate_check.rates([],[])==(None,None)
    assert lidar_rate_check.rates([1.],[1.])==(None,None)
    assert lidar_rate_check.rates([1.,1.],[1.,2.])==(None,None)

if __name__=='__main__':
    test_rates();test_too_few_messages();print('lidar rate arithmetic: PASS')

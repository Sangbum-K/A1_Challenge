"""Evaluation arithmetic on hand-made CSV rows; says nothing about real tracking quality."""
import csv,importlib.util,math,sys,tempfile
from pathlib import Path

spec=importlib.util.spec_from_file_location('track_eval',Path(__file__).resolve().parents[1]/'tools'/'track_eval.py')
track_eval=importlib.util.module_from_spec(spec);spec.loader.exec_module(track_eval)

def write(path,rows):
    with open(path,'w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def association_rows():
    rows=[]
    for i in range(10):
        primary=-1 if i<2 else 7 if i<6 else 9
        rows.append(dict(stamp=100+i*.1,primary_id=primary,primary_status='measured' if i%2 else 'camera_held',processing_ms=50+i,
                         map_filter_status='active',ego_status='compensated',raceline_status='active',dropped_scans=i//5))
    return rows

def track_rows():
    return [dict(stamp=100+i*.1,id=7,measured=int(i!=3),points=20,range_m=25+i*20,camera_state='visible' if i%2 else 'outside',
                 far_vehicle=int(i>=2),selected=1) for i in range(5)]

def test_summary():
    with tempfile.TemporaryDirectory() as folder:
        folder=Path(folder);write(folder/'association.csv',association_rows());write(folder/'association.tracks.csv',track_rows())
        labels=folder/'labels.csv';labels.write_text('start,end,lead\n100.0,100.45,0\n100.45,101.0,1\n')
        out=track_eval.summarize(folder/'association.csv',labels)
    assert out['scans']==10
    assert out['primary']['id_switches']==1
    assert out['primary']['segments']==2
    assert math.isclose(out['primary']['longest_s'],.3,abs_tol=1e-9)  # First to last scan of a hold.
    assert out['status_share']['camera_held']==.5
    assert out['map_status_share']['active']==1.
    assert out['dropped_scans']==1
    assert out['processing_ms']['p50']==54.5
    assert out['labels']['kept_rate']==1.
    assert math.isclose(out['labels']['wrong_rate'],.6)
    shells=out['range_shells']
    assert shells['20-40']['measured']==1 and shells['80-100']['measured']==0 and shells['80-100']['tracks']==1
    assert out['far_vehicle_scans']==3
    assert out['camera_state_share']['visible']==.4

def test_missing_labels_are_optional():
    with tempfile.TemporaryDirectory() as folder:
        folder=Path(folder);write(folder/'association.csv',association_rows());write(folder/'association.tracks.csv',track_rows())
        assert 'labels' not in track_eval.summarize(folder/'association.csv')

if __name__=='__main__':
    test_summary();test_missing_labels_are_optional();print('track_eval arithmetic: PASS')

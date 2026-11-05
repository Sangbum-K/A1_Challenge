"""Create/verify an explicit source-to-copy SHA256 inventory (including raw bags)."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--create',action='store_true');args=p.parse_args()
    path=ROOT/'COPY_MANIFEST.json'
    if args.create:
        sources={
            'reference/A1_RealCar_A2_P5Brake':ROOT.parent/'A1_RealCar_A2_P5Brake',
            'reference/A1_LeadVehicle_LidarFusion':ROOT.parent/'A1_LeadVehicle_LidarFusion',
            'reference/lidar_visualization_20261008_111019':ROOT.parents[1]/'lidar_visualization_20261008_111019',
            'datasets/a1_map_20261008_111019':Path('/home/ailab/.ros/a1_localization/bags/a1_map_20261008_111019'),
        }
        for name in ('trimble_driver_ros','HesaiLidar_ROS_2.0','a1_gnss_bringup'):
            sources['drivers/'+name]=Path('/home/ailab/git/A1_localization/src')/name
        records=[]
        for dest,source in sources.items():
            for original in sorted(source.rglob('*')):
                if not original.is_file():continue
                relative=str(Path(dest)/original.relative_to(source));copy=ROOT/relative
                digest=sha(original)
                if not copy.is_file() or sha(copy)!=digest:raise RuntimeError('copy mismatch: '+relative)
                records.append(dict(path=relative,source=str(original),size=copy.stat().st_size,sha256=digest))
            print('verified:',dest,flush=True)
        path.write_text(json.dumps(dict(files=records),indent=2)+'\n')
    else:
        records=json.loads(path.read_text())['files']
        for entry in records:
            if sha(ROOT/entry['path'])!=entry['sha256']:raise RuntimeError('modified: '+entry['path'])
    print(f'{len(records)} files verified; {sum(x["size"] for x in records):,} bytes',flush=True)

if __name__=='__main__':main()

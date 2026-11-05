"""One-off, user-authorized duplicate cleanup. Never deletes surviving originals."""
from pathlib import Path
import hashlib,json,os,shutil,time
r=Path(__file__).resolve().parents[2]
out=Path(__file__).with_name('20261104.json')
report={'actions':[],'removed_bytes':0}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
pairs=[('reference/A1_RealCar_A2_P5Brake/outputs',r.parent/'A1_RealCar_A2_P5Brake/outputs'),('datasets/a1_map_20261008_111019',Path('/home/ailab/.ros/a1_localization/bags/a1_map_20261008_111019')),('reference/lidar_visualization_20261008_111019',r.parents[1]/'lidar_visualization_20261008_111019')]
for rel,src in pairs:
 dest=r/rel
 if dest.is_symlink():continue
 assert src.is_dir() and dest.is_dir()
 files=[p for p in dest.rglob('*') if p.is_file()];size=0
 for p in files:
  q=src/p.relative_to(dest)
  assert not p.is_symlink() and q.is_file() and p.stat().st_size==q.stat().st_size,(p,q)
  assert sha(p)==sha(q),p
  size+=p.stat().st_size
 print('Verified',rel,size,flush=True)
 shutil.rmtree(dest);dest.symlink_to(os.path.relpath(src,dest.parent),target_is_directory=True)
 report['actions'].append(dict(path=rel,original=str(src),bytes=size,files=len(files),action='SHA256 verified; duplicate replaced by relative symlink'))
 report['removed_bytes']+=size;out.write_text(json.dumps(report,indent=2)+'\n')
p=r/'vendor/libtorch-cu118.zip'
if p.is_file():
 assert (r/'vendor/libtorch/lib/libtorch_cuda.so').is_file()
 size=p.stat().st_size;p.unlink();report['removed_bytes']+=size
 report['actions'].append(dict(path=str(p),bytes=size,action='removed extracted installer ZIP'))
report['finished']=time.strftime('%Y-%m-%d %H:%M:%S');out.write_text(json.dumps(report,indent=2)+'\n')
print(report,flush=True)

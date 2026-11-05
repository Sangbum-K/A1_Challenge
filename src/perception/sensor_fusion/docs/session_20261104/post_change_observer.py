import time,json,collections,math
from pathlib import Path
import rclpy,numpy as np
from rclpy.qos import qos_profile_sensor_data as qos
from sensor_msgs.msg import Image,PointCloud2
from a1_fusion_cpp.msg import Association,A2Features,BrakeEstimate
from cv_bridge import CvBridge
import cv2
out=Path('/home/ailab/git/sangbum/A1_Challenge-main/A1_SensorFusion_CPP/outputs/far_range_live_20261104');out.mkdir(exist_ok=True)
rclpy.init();n=rclpy.create_node('a1_readonly_moving_monitor');start=time.monotonic();rows=[];counts=collections.Counter();bridge=CvBridge();last_photo=-10;last_cloud=-10
f=(out/'events.jsonl').open('w')
def record(k,v):
 counts[k]+=1;row={'elapsed_s':time.monotonic()-start,'kind':k,**v};f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
def safe(v):return float(v) if math.isfinite(v) else None
def assoc(m):
 record('association',dict(primary=m.primary_track_id,status=m.primary_status,camera_status=m.status,processing_ms=m.processing_ms,tracks=[dict(id=t.id,measured=t.measured,age=t.observation_age_s,range=t.range_m,xy=list(t.center_m[:2]),vx=t.apparent_velocity_mps[0],camera=t.camera_matched,selected=t.lidar_candidate) for t in m.tracks]))
def a2(m):
 a=np.array(m.raw).reshape(40,7)[:,0];p=1/(1+np.exp(-np.clip(a,-80,80)));record('a2',dict(ready=m.history_ready,peak_bin=int(np.nanargmax(p)),peak_probability=safe(float(np.nanmax(p)))))
def cloud(m):
 global last_cloud
 counts['cloud_received']+=1
 if time.monotonic()-last_cloud<1:return
 last_cloud=time.monotonic();xyz=np.stack([np.ndarray((m.height,m.width),dtype='<f4',buffer=m.data,offset=next(f.offset for f in m.fields if f.name==k),strides=(m.row_step,m.point_step)).ravel() for k in ('x','y','z')],axis=1);d=np.linalg.norm(xyz,axis=1);finite=d[np.isfinite(d)]
 record('cloud_stats',dict(points=len(d),valid1to80=int(((d>=1)&(d<=80)).sum()),min=safe(float(np.min(finite))) if len(finite) else None,max=safe(float(np.max(finite))) if len(finite) else None))
def img(m):
 global last_photo
 counts['image_received']+=1
 elapsed=time.monotonic()-start
 if elapsed-last_photo>=5:
  last_photo=elapsed;cv2.imwrite(str(out/f'camera_{int(elapsed):03d}.jpg'),bridge.imgmsg_to_cv2(m,'bgr8'))
def brake(m):record('p5',dict(valid=m.valid,reason=m.reason,id=m.track_id,gap=safe(m.gap_m),vrel=safe(m.relative_speed_mps),request=safe(m.request_mps2)))
subs=[n.create_subscription(c,t,cb,qos) for c,t,cb in [(Image,'/camera_driver_gv/vis/image_raw',img),(PointCloud2,'/lidar_points',cloud),(A2Features,'/lead_fusion/a2',a2),(Association,'/lead_fusion/association',assoc),(BrakeEstimate,'/lead_fusion/estimate',brake)]]
end=start+60;last_report=start
while time.monotonic()<end:
 rclpy.spin_once(n,timeout_sec=.01)
 if time.monotonic()-last_report>=15:print(dict(elapsed=round(time.monotonic()-start),counts=dict(counts)),flush=True);last_report=time.monotonic()
f.close();(out/'counts.json').write_text(json.dumps(dict(counts),indent=2));print(out,dict(counts),flush=True);n.destroy_node();rclpy.shutdown()

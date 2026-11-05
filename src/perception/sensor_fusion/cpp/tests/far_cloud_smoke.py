import argparse,subprocess,time,json
from pathlib import Path
from array import array
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data as qos
from sensor_msgs.msg import PointCloud2,PointField
from a1_fusion_cpp.msg import Association
p=argparse.ArgumentParser();p.add_argument('--binary',required=True);p.add_argument('--output',required=True);p.add_argument('--config',required=True);args=p.parse_args()
rclpy.init();n=rclpy.create_node('far_static_snapshot_replay');rows=[];pub=n.create_publisher(PointCloud2,'/replay/cloud',qos);sub=n.create_subscription(Association,'/lead_fusion/association',rows.append,qos)
proc=subprocess.Popen([args.binary,'--ros-args','--params-file',args.config,'-p','lidar_topic:=/replay/cloud','-p','csv_path:='+args.output+'.csv'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
def spin(sec):
 end=time.monotonic()+sec
 while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.01)
try:
 spin(2);results=[]
 for distance in [24.,49.,99.]:
  xx,yy=np.meshgrid(np.arange(2,100,.4),np.arange(-10,10,.4));ground=np.stack([xx.ravel(),yy.ravel(),np.full(xx.size,-1.8)],axis=1)
  surfaces=[]
  for lateral in [-3.,3.]:
   for dy in np.arange(-.9,1.,.3):
    for z in [-1.1,-.68]:surfaces.append([distance,lateral+dy,z])
  body=np.concatenate([ground,np.array(surfaces)])
  xyz=np.stack([body[:,1],-body[:,0],body[:,2]],axis=1).astype(np.float32);start=len(rows);path=f"synthetic_{distance}m"

  for i in range(15):
   m=PointCloud2();m.header.stamp=n.get_clock().now().to_msg();m.header.frame_id='hesai_lidar';m.height=1;m.width=len(xyz);m.point_step=12;m.row_step=len(xyz)*12;m.fields=[PointField(name=k,offset=j*4,datatype=7,count=1) for j,k in enumerate(['x','y','z'])];m.data=array('B',xyz.tobytes());pub.publish(m);spin(.1)
  batch=rows[start:]
  assert batch, 'no frames'
  tracks=[t for t in batch[-1].tracks if abs(t.range_m-distance)<1 and t.confirmed]
  assert len(tracks)==2, (distance,[(t.id,t.range_m,list(t.raw_size_m)) for t in batch[-1].tracks])
  assert not any(t.lidar_candidate for t in tracks), 'camera-free clouds acquired a range target'
  assert batch[-1].primary_track_id==-1
  results.append(dict(file=str(path),frames=len(batch),processing_ms=[r.processing_ms for r in batch],last_tracks=[dict(id=t.id,range=t.range_m,size=list(t.raw_size_m),points=t.points,candidate=t.lidar_candidate) for t in batch[-1].tracks] if batch else []))
 Path(args.output).write_text(json.dumps(results,indent=2));print("24/49/99m sparse tracks retained; camera-free target acquisition blocked: PASS")
finally:
 proc.terminate();proc.wait(timeout=10);n.destroy_node();rclpy.shutdown()

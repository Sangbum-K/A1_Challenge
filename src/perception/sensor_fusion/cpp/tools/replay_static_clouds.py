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
 for path in sorted(Path('A1_Challenge-main/A1_SensorFusion_CPP/outputs/moving_monitor_20261104_1005').glob('far_raw_*.npz')):
  xyz=np.asarray(np.load(path)['xyz'],dtype=np.float32);start=len(rows)
  for i in range(15):
   m=PointCloud2();m.header.stamp=n.get_clock().now().to_msg();m.header.frame_id='hesai_lidar';m.height=1;m.width=len(xyz);m.point_step=12;m.row_step=len(xyz)*12;m.fields=[PointField(name=k,offset=j*4,datatype=7,count=1) for j,k in enumerate(['x','y','z'])];m.data=array('B',xyz.tobytes());pub.publish(m);spin(.1)
  batch=rows[start:];results.append(dict(file=str(path),frames=len(batch),processing_ms=[r.processing_ms for r in batch],last_tracks=[dict(id=t.id,range=t.range_m,size=list(t.raw_size_m),points=t.points,candidate=t.lidar_candidate) for t in batch[-1].tracks] if batch else []))
 Path(args.output).write_text(json.dumps(results,indent=2))
finally:
 proc.terminate();proc.wait(timeout=10);n.destroy_node();rclpy.shutdown()

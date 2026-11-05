from pathlib import Path
import time,numpy as np,rclpy
from sensor_msgs.msg import PointCloud2
from rclpy.qos import qos_profile_sensor_data as qos
rclpy.init();n=rclpy.create_node('a1_far_cloud_capture');out=Path('/home/ailab/git/sangbum/A1_Challenge-main/A1_SensorFusion_CPP/outputs/far_range_live_20261104');done=0;last=0
f=(out/'far_cloud_capture.txt').open('w')
def cb(m):
 global done,last
 if time.monotonic()-last<1 or done>=5:return
 last=time.monotonic();xyz=np.stack([np.ndarray((m.height,m.width),dtype='<f4',buffer=m.data,offset=next(f.offset for f in m.fields if f.name==k),strides=(m.row_step,m.point_step)).ravel() for k in ('x','y','z')],axis=1)
 np.savez_compressed(out/f'far_raw_{done}.npz',xyz=xyz,stamp=m.header.stamp.sec+m.header.stamp.nanosec/1e9);done+=1
s=n.create_subscription(PointCloud2,'/lidar_points',cb,qos);end=time.monotonic()+8
while done<5 and time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.05)
f.write(str(done));f.close();n.destroy_node();rclpy.shutdown()

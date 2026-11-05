import time,json
from pathlib import Path
import rclpy
from rclpy.qos import qos_profile_sensor_data as qos
from a1_fusion_cpp.msg import Association
rclpy.init();n=rclpy.create_node('a1_track_detail_readonly');out=Path('/home/ailab/git/sangbum/A1_Challenge-main/A1_SensorFusion_CPP/outputs/moving_monitor_20261104_second/details.jsonl');f=out.open('w');start=time.monotonic()
def cb(m):
 r=dict(elapsed=time.monotonic()-start,stamp=m.header.stamp.sec+m.header.stamp.nanosec/1e9,primary=m.primary_track_id,tracks=[dict(id=t.id,measured=t.measured,confirmed=t.confirmed,age=t.observation_age_s,valid=t.prediction_valid,raw_size=list(t.raw_size_m),size=list(t.size_m),range=t.range_m,xy=list(t.center_m[:2]),cov=list(t.position_covariance_xy),points=t.points,vx=t.apparent_velocity_mps[0],selected=t.lidar_candidate) for t in m.tracks]);f.write(json.dumps(r)+'\n');f.flush()
s=n.create_subscription(Association,'/lead_fusion/association',cb,qos)
while time.monotonic()-start<60:rclpy.spin_once(n,timeout_sec=.02)
f.close();n.destroy_node();rclpy.shutdown()

"""Independent-process locality test, 128k-point messages; no hardware drivers."""
import argparse,os,subprocess,time,tempfile,json,hashlib,sys
import rclpy
from sensor_msgs.msg import PointCloud2,PointField
from rclpy.qos import qos_profile_sensor_data as qos
from array import array
p=argparse.ArgumentParser();p.add_argument('--binary',required=True);a=p.parse_args()
os.environ['ROS_LOCALHOST_ONLY']='0';rclpy.init(domain_id=115);s=rclpy.create_node('shared_localization_mock');pub=s.create_publisher(PointCloud2,'/lidar_points',qos)
peer_code='''import rclpy,json,sys,hashlib
from sensor_msgs.msg import PointCloud2
from rclpy.qos import qos_profile_sensor_data as q
rclpy.init(domain_id=116);n=rclpy.create_node('isolated_mock');f=open(sys.argv[1],'w')
def cb(m):
 f.write(json.dumps([m.header.stamp.sec,m.header.stamp.nanosec,m.header.frame_id,m.point_step,hashlib.sha256(bytes(m.data)).hexdigest()])+'\\n');f.flush()
sub=n.create_subscription(PointCloud2,'/lead_fusion/input/lidar_points',cb,q)
rclpy.spin(n)
'''
tmp=tempfile.TemporaryDirectory();file=tmp.name+'/received.jsonl';env=dict(os.environ,ROS_LOCALHOST_ONLY='1',ROS_DOMAIN_ID='116',FASTRTPS_DEFAULT_PROFILES_FILE=os.environ['A1_BRIDGE_TARGET_PROFILE']);peer=subprocess.Popen([sys.executable,'-c',peer_code,file],env=env,stderr=open("/tmp/lidar_bridge_peer_error.log","w"))
proc=None;counter=0;sent={}
def rows():
 try:
  with open(file) as f:return [json.loads(x) for x in f if x.endswith('\n')]
 except FileNotFoundError:return []
def pump(seconds):
 global counter
 end=time.monotonic()+seconds
 while time.monotonic()<end:
  m=PointCloud2();m.header.stamp=s.get_clock().now().to_msg();m.header.frame_id='hesai_lidar';m.height=1;m.width=128000;m.point_step=12;m.row_step=1536000;m.fields=[PointField(name=k,offset=i*4,datatype=7,count=1) for i,k in enumerate(['x','y','z'])];m.data=array('B',bytes([counter%255])*1536000);counter+=1
  sent[(m.header.stamp.sec,m.header.stamp.nanosec)]=hashlib.sha256(bytes(m.data)).hexdigest();pub.publish(m);rclpy.spin_once(s,timeout_sec=.01);time.sleep(.07)
try:
 proc=subprocess.Popen([a.binary,'115','/lidar_points','116','--check']);pump(3);assert proc.wait(timeout=5)==0;assert not rows()
 proc=subprocess.Popen([a.binary,'115','/lidar_points','116']);pump(10);seen=rows();assert peer.poll() is None,("peer died",peer.returncode);assert len(seen)>=10,(len(seen),open(file).read()[:200])
 for sec,ns,frame,step,digest in seen:assert frame=='hesai_lidar' and step==12 and digest==sent[(sec,ns)]
 assert s.count_publishers('/lidar_points')==1
 proc.terminate();assert proc.wait(timeout=5)==0;proc=None;assert s.count_publishers('/lidar_points')==1
 second=s.create_publisher(PointCloud2,'/lidar_points',qos);proc=subprocess.Popen([a.binary,'115','/lidar_points','116','--check']);pump(3);assert proc.wait(timeout=5)==3;proc=None
 s.destroy_publisher(second);s.destroy_publisher(pub);proc=subprocess.Popen([a.binary,'115','/lidar_points','116','--check']);assert proc.wait(timeout=10)==2;proc=None
 assert subprocess.run([a.binary,'107','/lidar_points','107','--check']).returncode!=0
 print('PASS: separate processes local0->local1, 128k-point byte/stamp preservation, one-way, check-only, external-owner survival, duplicate/missing/loop rejection')
finally:
 if proc and proc.poll() is None:proc.terminate();proc.wait(timeout=5)
 peer.terminate();peer.wait(timeout=5);s.destroy_node();rclpy.shutdown();tmp.cleanup()

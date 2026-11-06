"""Bounded sensor reception check using existing installed drivers; stop only our children."""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image,PointCloud2,Imu
from gsof_msgs.msg import NavigationSolution49
from a1_fusion_cpp.msg import A2Features,BrakeEstimate

ROOT=Path(__file__).resolve().parents[2]
WS_ROOT=ROOT.parents[2] if ROOT.parts[-3:]==('src','perception','sensor_fusion') else ROOT.parent  # A1_Total: <ws>/src/perception/sensor_fusion; A1_Challenge: <repo>/sensor_fusion
def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=25);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--with-cpp',action='store_true');p.add_argument('--host-lidar-time',action='store_true')
    p.add_argument('--device', default='cpu')
    p.add_argument('--cpp-install', type=Path, default=WS_ROOT/'install')
    p.add_argument('--models', type=Path, default=ROOT/'cpp/models')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    rclpy.init();n=rclpy.create_node('a1_live_sensor_audit');data=defaultdict(list);latest={};subs=[];children=[];files=[]
    def callback(kind,m):
        now=time.time();stamp=m.header.stamp.sec+m.header.stamp.nanosec/1e9;data[kind].append((now,stamp))
        v=dict(frame=m.header.frame_id,header_stamp=stamp,receive_minus_header_s=now-stamp)
        if kind=='camera':v.update(width=m.width,height=m.height,encoding=m.encoding)
        elif kind=='lidar':v.update(width=m.width,height=m.height,point_step=m.point_step,fields=[f.name for f in m.fields])
        elif kind=='ins':v.update(velocity_ned=[m.velocity.north,m.velocity.east,m.velocity.down],total_speed=m.total_speed,heading_deg=m.heading,roll_deg=m.roll,pitch_deg=m.pitch,imu_alignment=m.status.imu_alignment,gnss_status=m.status.gnss)
        elif kind=='imu':v.update(acceleration=[m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z],angular_rate=[m.angular_velocity.x,m.angular_velocity.y,m.angular_velocity.z])
        elif kind=='a2':v.update(history_ready=m.history_ready,raw=[float(x) for x in m.raw])
        else:v.update(valid=m.valid,reason=m.reason,request_mps2=m.request_mps2 if math.isfinite(m.request_mps2) else None)
        latest[kind]=v
    topics={'camera':('/camera_driver_gv/vis/image_raw',Image),'lidar':('/lidar_points',PointCloud2),'ins':('/gsof_client/gsof/ins_solution_49',NavigationSolution49)}
    if args.with_cpp:topics.update(a2=('/lead_fusion/a2',A2Features),estimate=('/lead_fusion/estimate',BrakeEstimate),imu=('/gnss/imu',Imu))
    for kind,(topic,typ) in topics.items():subs.append(n.create_subscription(typ,topic,lambda m,k=kind:callback(k,m),qos_profile_sensor_data))
    commands={
      'camera':['ros2','launch',str(ROOT/'cpp/launch/hikrobot_gpu.launch.py'),'guid:=169.254.222.118','exposure_us:=3000','fps:=21'],
      'lidar':[str(WS_ROOT/'install/lib/hesai_ros_driver/hesai_ros_driver_node'),'--ros-args','-p','config_path:='+str(WS_ROOT/'src/external/HesaiLidar_ROS_2.0/config/config.yaml')],
      'ins':[str(WS_ROOT/'install/lib/trimble_driver_ros/gsof_client_node'),'--ros-args','-r','__node:=gsof_client','--params-file',str(WS_ROOT/'src/localization/a1_gnss_bringup/config/gsof_client.yaml')],
    }
    if args.host_lidar_time:commands['lidar'][-1]='config_path:='+str(ROOT/'cpp/config/lidar_host_time.yaml')
    if args.with_cpp:
        commands['a2']=[str(args.cpp_install/'lib/a1_fusion_cpp/a2_camera_node'),'--ros-args','-p','model_dir:='+str(args.models),'-p','device:='+args.device]
        commands['estimate']=[str(args.cpp_install/'lib/a1_fusion_cpp/fusion_node'),'--ros-args','--params-file',str(ROOT/'cpp/config/shadow.yaml')]
    try:
        for _ in range(10):rclpy.spin_once(n,timeout_sec=.05)
        for kind,command in commands.items():
            if n.count_publishers(topics[kind][0]):continue
            f=(args.out/(kind+'.log')).open('w');files.append(f)
            children.append((kind,subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=args.out)))
        end=time.monotonic()+args.seconds
        while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.02)
    finally:
        for _,proc in children:
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGINT)
        for _,proc in children:
            try:proc.wait(timeout=4)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        for f in files:f.close()
        result={}
        for kind in topics:
            rows=data[kind];result[kind]=dict(count=len(rows),latest=latest.get(kind),
                receive_hz=(len(rows)-1)/(rows[-1][0]-rows[0][0]) if len(rows)>1 else None,
                median_header_age_s=statistics.median(x-y for x,y in rows) if rows else None)
        result['children']={kind:p.returncode for kind,p in children}
        (args.out/'sensors.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
        n.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()

"""Synthetic ROS system test. Isolated domain; no real sensor/vehicle writes.

Run after sourcing ROS, gsof_msgs and the installed a1_fusion_cpp package.
"""
import argparse
import math
from pathlib import Path
import struct
import subprocess
import time
import rclpy
from rclpy.qos import qos_profile_sensor_data
from gsof_msgs.msg import NavigationSolution49
from sensor_msgs.msg import PointCloud2, PointField
from a1_fusion_cpp.msg import A2Features, BrakeEstimate

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--binary',required=True)
    args=p.parse_args()
    rclpy.init()
    n=rclpy.create_node('a1_cpp_synthetic_test')
    ego=n.create_publisher(NavigationSolution49,'/test/ins',10)
    cam=n.create_publisher(A2Features,'/lead_fusion/a2',10)
    lidar=n.create_publisher(PointCloud2,'/test/cloud',10)
    received=[]
    sub=n.create_subscription(BrakeEstimate,'/lead_fusion/estimate',received.append,10)
    params=['mode:=fusion','ego_source:=ins','clock_verified:=true','calibration_verified:=true',
            'ins_mount_verified:=true','ins_topic:=/test/ins','lidar_topic:=/test/cloud',
            'camera_from_base:=[0.0,0.0,0.0,0.0,-1.5707963267948966,1.5707963267948966]']
    cmd=[args.binary,'--ros-args']
    for param in params:cmd+=['-p',param]
    process=subprocess.Popen(cmd,stdout=subprocess.DEVNULL)
    def spin(seconds):
        stop=time.monotonic()+seconds
        while time.monotonic()<stop:rclpy.spin_once(n,timeout_sec=.005)
    try:
        spin(1.5)
        start=time.time()
        for frame in range(18):
            stamp=n.get_clock().now().to_msg()
            t=stamp.sec+stamp.nanosec/1e9
            nav=NavigationSolution49();nav.header.stamp=stamp;nav.heading=90.
            nav.status.imu_alignment=4;nav.status.gnss=4;nav.velocity.east=10.
            ego.publish(nav);spin(.008)
            a=A2Features();a.header.stamp=stamp;a.header.frame_id='hikrobot_camera_vis';a.history_ready=True
            raw=[float('nan')]*280
            for k in range(40):raw[k*7]=-8.
            raw[20*7]=8.;a.raw=raw;cam.publish(a);spin(.008)
            m=PointCloud2();m.header.stamp=stamp;m.header.frame_id='hesai_lidar';m.height=1
            m.fields=[PointField(name=name,offset=offset,datatype=dtype,count=1)
                      for name,offset,dtype in [('x',0,7),('y',4,7),('z',8,7),('timestamp',12,8)]]
            # Ground + one vehicle-like return cluster. No synthetic camera metrics.
            points=[(x*.4,y*.4,-1.) for x in range(4,62) for y in range(-15,16)]
            distance=25-5*(t-start)
            points += [(distance+x*.2,-.7+y*.2,-.5+z*.2)
                       for x in range(8) for y in range(5) for z in range(6)]
            m.point_step=20;m.width=len(points);m.row_step=20*m.width;m.is_dense=True
            m.data=b''.join(struct.pack('<fffd',x,y,z,t) for x,y,z in points)
            lidar.publish(m);spin(.085)
        good=[m for m in received if m.valid]
        assert len(good)>=8, [(m.valid,m.reason) for m in received]
        last=good[-1]
        assert abs(last.relative_speed_mps+5)<1.0,last
        assert abs(last.lead_speed_mps-5)<1.0,last
        assert last.gap_m>0 and last.request_mps2<0,last
        assert not last.what_if
        # A stopped input must expire; invalid is explicitly NaN, not zero brake.
        spin(.6)
        assert not received[-1].valid and received[-1].reason=='input_timeout'
        assert math.isnan(received[-1].request_mps2)
        print(f'ROS fusion: {len(good)} valid outputs; vrel={last.relative_speed_mps:.3f}, '
              f'lead={last.lead_speed_mps:.3f}, brake={last.request_mps2:.3f}; timeout PASS')
    finally:
        process.terminate()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        n.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()

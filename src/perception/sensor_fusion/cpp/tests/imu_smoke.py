"""IMU-only ROS path: zero-start calibration, integrated velocity, and stale rejection."""
import argparse
import subprocess
import time
import rclpy
from builtin_interfaces.msg import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu
from a1_fusion_cpp.msg import A2Features,BrakeEstimate,EgoMotion

def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',required=True);args=p.parse_args()
    rclpy.init();n=rclpy.create_node('imu_only_test');received=[];motions=[]
    clock=n.create_publisher(Clock,'/clock',10);imu=n.create_publisher(Imu,'/gnss/imu',10)
    camera=n.create_publisher(A2Features,'/lead_fusion/a2',10)
    sub=n.create_subscription(BrakeEstimate,'/lead_fusion/estimate',received.append,10)
    motion_sub=n.create_subscription(EgoMotion,'/lead_fusion/ego_motion',motions.append,10)
    params=['mode:=camera','ego_source:=imu','use_sim_time:=true','clock_verified:=true',
            'camera_metric_validated:=true','imu_stationary_start_confirmed:=true',
            'imu_mount_verified:=true','imu_calibration_s:=0.5','imu_max_integration_s:=3.0']
    command=[args.binary,'--ros-args']
    for param in params:command+=['-p',param]
    process=subprocess.Popen(command,stdout=subprocess.DEVNULL)
    def spin(duration):
        end=time.monotonic()+duration
        while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.002)
    try:
        spin(1.0)
        for i in range(161):
            ns=10_000_000_000+i*10_000_000;stamp=Time(sec=ns//10**9,nanosec=ns%10**9)
            clock.publish(Clock(clock=stamp))
            msg=Imu();msg.header.stamp=stamp;msg.header.frame_id='ins';msg.linear_acceleration.z=9.80665
            msg.linear_acceleration.x=1.0 if i>60 else 0.0;imu.publish(msg);spin(.004)
            if i%5==0:
                a=A2Features();a.header.stamp=stamp;a.header.frame_id='hikrobot_camera_vis';a.history_ready=True
                a.raw=[-8. if k%7==0 else 0. for k in range(280)]
                a.raw[140]=8.;a.raw[143]=-.1;a.raw[146]=.25;camera.publish(a)
            spin(.004)
        spin(.1);good=[m for m in received if m.valid]
        assert good,[(m.reason,m.ego_source) for m in received[-10:]]
        last=good[-1]
        assert last.ego_source=='imu_integrated_experimental',last
        assert abs(last.ego_speed_mps-.995)<.03,last
        assert last.request_mps2<0,last
        motion=[m for m in motions if m.valid][-1]
        assert abs(motion.speed_mps-.995)<.03,motion
        assert abs(motion.longitudinal_acceleration_mps2-1.0)<1e-6,motion
        # A fresh camera observation cannot make an old IMU velocity fresh.
        stamp=Time(sec=12,nanosec=0);clock.publish(Clock(clock=stamp));spin(.01)
        a.header.stamp=stamp;camera.publish(a);spin(.1)
        assert not received[-1].valid,received[-1]
        print(f'IMU ROS: ego={last.ego_speed_mps:.4f} m/s, source={last.ego_source}; stale rejection PASS')
    finally:
        process.terminate()
        try:process.wait(timeout=4)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        n.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()

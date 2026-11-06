"""Unverified GNSS motion is logged but never valid. Synthetic, no sensors, no vehicle commands."""
import argparse,math,subprocess,time
import rclpy
from rclpy.qos import qos_profile_sensor_data
from a1_fusion_cpp.msg import EgoMotion
from gsof_msgs.msg import NavigationSolution49
from nav_msgs.msg import Odometry

def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',required=True);args=p.parse_args()
    rclpy.init();node=rclpy.create_node('unverified_ego_test');egos=[]
    status_pub=node.create_publisher(NavigationSolution49,'/test/status',qos_profile_sensor_data)
    odom_pub=node.create_publisher(Odometry,'/test/odom',qos_profile_sensor_data)
    node.create_subscription(EgoMotion,'/lead_fusion/ego_motion',egos.append,10)
    proc=subprocess.Popen([args.binary,'--ros-args','-p','mode:=association','-p','ego_source:=gnss_odom',
        '-p','gnss_odom_topic:=/test/odom','-p','gnss_status_topic:=/test/status'],stdout=subprocess.DEVNULL)
    def spin(seconds):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.005)
    try:
        deadline=time.monotonic()+8
        while time.monotonic()<deadline and not (odom_pub.get_subscription_count() and status_pub.get_subscription_count()):spin(.05)
        assert odom_pub.get_subscription_count(),'no odometry subscriber'
        spin(.3)
        for _ in range(5):
            ts=node.get_clock().now().to_msg()
            status=NavigationSolution49();status.header.stamp=ts;status.status.imu_alignment=4;status.status.gnss=4
            status_pub.publish(status);spin(.015)
            odom=Odometry();odom.header.stamp=ts;odom.header.frame_id='odom';odom.child_frame_id='base_link'
            odom.pose.pose.orientation.w=1.;odom.twist.twist.linear.x=10.;odom.twist.covariance[0]=.1
            odom_pub.publish(odom);spin(.05)
        logged=[e for e in egos if e.reason=='gnss_odom_unverified']
        assert logged,[e.reason for e in egos]
        last=logged[-1]
        assert not last.valid and math.isnan(last.speed_mps) and last.speed_unverified_mps==10.,last
        print('Unverified GNSS: logged speed, valid stays false: PASS')
    finally:
        proc.terminate()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
        node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()

"""GNSS status + odometry -> existing primary P5, synthetic isolated-domain test."""
import argparse
import math
import os
import subprocess
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from a1_fusion_cpp.msg import Association, BrakeEstimate, ClusterTrack, EgoMotion
from gsof_msgs.msg import NavigationSolution49
from nav_msgs.msg import Odometry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True)
    args = parser.parse_args()
    assert os.environ.get('ROS_LOCALHOST_ONLY') == '1' and int(os.environ.get('ROS_DOMAIN_ID', '0')) >= 100, 'Use an isolated test domain'
    rclpy.init()
    node = rclpy.create_node('gnss_ego_contract_test')
    estimates, egos = [], []
    status_pub = node.create_publisher(NavigationSolution49, '/test/status', qos_profile_sensor_data)
    odom_pub = node.create_publisher(Odometry, '/test/odom', qos_profile_sensor_data)
    assoc_pub = node.create_publisher(Association, '/lead_fusion/association', qos_profile_sensor_data)
    node.create_subscription(BrakeEstimate, '/lead_fusion/estimate', estimates.append, 10)
    node.create_subscription(EgoMotion, '/lead_fusion/ego_motion', egos.append, 10)
    proc = subprocess.Popen([args.binary, '--ros-args', '-p', 'mode:=association',
        '-p', 'ego_source:=gnss_odom', '-p', 'gnss_odom_topic:=/test/odom',
        '-p', 'gnss_status_topic:=/test/status', '-p', 'gnss_odom_verified:=true',
        '-p', 'clock_verified:=true', '-p', 'calibration_verified:=true',
        '-p', 'primary_motion_verified:=true', '-p', 'base_from_lidar:=[1.16, 0.2, 1.2, 0.0, 0.0, 1.5707963267948966]',
        '-p', 'bumper_x_m:=2.0'], stdout=subprocess.DEVNULL)
    def spin(seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.005)
    def frame(alignment=3, child='base_link', age=0., orientation=1., duplicate=False, covariance=.1):
        ts = node.get_clock().now().to_msg()
        ns = ts.sec * 10**9 + ts.nanosec - round(age * 1e9)
        ts.sec, ts.nanosec = divmod(ns, 10**9)
        status = NavigationSolution49(); status.header.stamp = ts
        # 3 (ALIGNED) is what the vehicle's Applanix reports in A1_localization's recordings.
        status.status.imu_alignment = alignment; status.status.gnss = 4
        status_pub.publish(status); spin(.015)
        msg = Odometry(); msg.header.stamp = ts; msg.header.frame_id = 'odom'; msg.child_frame_id = child
        msg.pose.pose.orientation.w = orientation
        msg.pose.pose.position.x = 100000.  # Map/position jumps must not become speed.
        msg.twist.twist.linear.x = 10.; msg.twist.twist.angular.z = .5
        msg.twist.covariance[0] = covariance
        odom_pub.publish(msg); spin(.02)
        if duplicate:
            odom_pub.publish(msg); spin(.02)
        assoc = Association(); assoc.header.stamp = ts; assoc.header.frame_id = 'lidar_vehicle'
        assoc.clock_verified = assoc.calibration_verified = True
        assoc.primary_track_id = 42; assoc.primary_status = 'measured'; assoc.primary_camera_age_s = 0.
        tr = ClusterTrack(); tr.id = 42; tr.confirmed = tr.prediction_valid = tr.measured = tr.camera_matched = True
        tr.center_m = [20., 1., 0.]; tr.size_m = [2., 1.8, 1.5]
        tr.near_surface_x_m = 19.; tr.relative_velocity_mps = [-2.6, 0.]; tr.position_covariance_xy = [.1, 0., 0., .1]
        assoc.tracks = [tr]
        start = len(estimates); assoc_pub.publish(assoc); spin(.06)
        rows = [v for v in estimates[start:] if abs((v.header.stamp.sec-ts.sec)+(v.header.stamp.nanosec-ts.nanosec)/1e9)<1e-5]
        assert rows, 'No primary response'
        return rows[-1]
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and not (odom_pub.get_subscription_count() and status_pub.get_subscription_count() and assoc_pub.get_subscription_count()):
            spin(.05)
        assert odom_pub.get_subscription_count()
        spin(.3)
        good = frame()
        assert good.valid and good.ego_source == 'gnss_odometry', good
        assert math.isclose(good.gap_m, 18.16) and math.isclose(good.relative_speed_mps, -2.6), good
        assert math.isclose(good.x_m, 21.16) and good.ego_speed_mps == 10., good
        assert egos[-1].valid and egos[-1].source == 'gnss_odometry' and math.isnan(egos[-1].longitudinal_acceleration_mps2)
        assert frame(alignment=4).valid, 'FULL_NAV rejected'
        # GPS_ONLY, DEGRADED and IMU_UNKNOWN are not usable alignments.
        for kwargs in ({'alignment':0}, {'alignment':2}, {'alignment':5}, {'child':'ins'}, {'age':1.}, {'age':-.1}, {'orientation':float('nan')}, {'duplicate':True}, {'covariance':-1.}):
            bad = frame(**kwargs)
            assert not bad.valid, (kwargs, bad)
        duplicate_pub = node.create_publisher(Odometry, '/test/odom', qos_profile_sensor_data)
        spin(.2)
        assert not frame().valid, 'duplicate odometry sources accepted'
        node.destroy_publisher(duplicate_pub); spin(.3)
        assert frame().valid, 'recovery failed'
        spin(.4)
        assert not estimates[-1].valid and not egos[-1].valid, 'loss must invalidate P5 and UI'
        # Status loss while odometry continues must not keep ego valid.
        for _ in range(10):
            msg = Odometry(); msg.header.stamp = node.get_clock().now().to_msg()
            msg.header.frame_id = 'odom'; msg.child_frame_id = 'base_link'; msg.pose.pose.orientation.w = 1.
            odom_pub.publish(msg); spin(.02)
        assert not egos[-1].valid
        print('GNSS ego: speed, lever arm, yaw correction, status/odom loss, bad frames/clocks/covariance, duplicate inputs, recovery: PASS')
    finally:
        proc.terminate()
        try: proc.wait(timeout=5)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()

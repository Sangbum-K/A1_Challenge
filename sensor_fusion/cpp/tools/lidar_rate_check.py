#!/usr/bin/env python3
"""Measure the LiDAR message rate with a best_effort, depth-1 subscriber, the way A1_localization's
verify script reads it. `ros2 topic hz` can under-report large PointCloud2 messages, so do not trust it alone.
Subscribes only; publishes nothing and starts no driver. Exit 1 below --min-hz or with under 3 messages."""
import argparse
import sys
import time


def rates(arrivals, stamps):
    """Hz from receive times and from header stamps; None when there are fewer than 2 messages."""
    if len(arrivals) < 2 or arrivals[-1] <= arrivals[0]:
        return None, None
    received = (len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
    span = stamps[-1] - stamps[0]
    return received, ((len(stamps) - 1) / span if span > 0 else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', default='/lidar_points')
    parser.add_argument('--seconds', type=float, default=5.)
    parser.add_argument('--min-hz', type=float, default=4.)
    parser.add_argument('--warn-hz', type=float, default=7.)
    args = parser.parse_args()
    import rclpy  # Imported here so rates() can be tested without ROS.
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2
    rclpy.init()
    node = rclpy.create_node('a1_lidar_rate_check')
    arrivals, stamps, widths = [], [], set()

    def on_cloud(msg):
        arrivals.append(time.monotonic())
        stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9)
        widths.add((msg.width, msg.height, msg.point_step))

    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
    node.create_subscription(PointCloud2, args.topic, on_cloud, qos)
    try:
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.05)
        publishers = node.count_publishers(args.topic)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    received, header = rates(arrivals, stamps)
    print(f'{args.topic}: {len(arrivals)} messages in {args.seconds:.0f} s, publishers={publishers}, '
          f'receive {received if received is None else round(received, 2)} Hz, '
          f'header {header if header is None else round(header, 2)} Hz, width/height/point_step={sorted(widths)}')
    if len(arrivals) < 3 or received is None:
        print('FAIL: fewer than 3 messages; is the LiDAR driver (localization) running in this domain?')
        sys.exit(1)
    if publishers != 1:
        print(f'WARN: expected exactly 1 publisher, found {publishers}; the fusion node refuses mixed sources')
    if received < args.min_hz:
        print(f'FAIL: {received:.2f} Hz is below {args.min_hz:g} Hz; track confirmation cannot work')
        sys.exit(1)
    if received < args.warn_hz:
        print(f'WARN: {received:.2f} Hz is below {args.warn_hz:g} Hz; the pipeline is tuned for 10 Hz')


if __name__ == '__main__':
    main()

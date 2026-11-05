"""Publish a synthetic BayerRG8 stream for an integration smoke test only."""

import argparse
import time

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/a1_a2_synthetic/image_raw")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=21.0)
    parser.add_argument("--duration", type=float, default=15.0)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0 or args.duration <= 0:
        raise ValueError("width, height, fps, and duration must be positive")

    bayer = np.zeros((args.height, args.width), dtype=np.uint8)
    x_gradient = np.linspace(30, 220, args.width, dtype=np.uint8)
    bayer[0::2, 0::2] = x_gradient[0::2]
    bayer[0::2, 1::2] = 110
    bayer[1::2, 0::2] = 110
    bayer[1::2, 1::2] = x_gradient[1::2]

    rclpy.init(args=None)
    node = rclpy.create_node("a1_a2_synthetic_bayer_pub")
    publisher = node.create_publisher(Image, args.topic, qos_profile_sensor_data)
    msg = Image()
    msg.header.frame_id = "synthetic_bayer_not_a_camera"
    msg.height = args.height
    msg.width = args.width
    msg.encoding = "bayer_rggb8"
    msg.is_bigendian = 0
    msg.step = args.width
    msg.data = bayer.tobytes()
    period = 1.0 / args.fps
    started = time.perf_counter()
    next_publish = started
    published = 0
    try:
        while time.perf_counter() - started < args.duration:
            msg.header.stamp = node.get_clock().now().to_msg()
            publisher.publish(msg)
            published += 1
            next_publish += period
            time.sleep(max(0.0, next_publish - time.perf_counter()))
    finally:
        node.destroy_node()
        rclpy.shutdown()
    elapsed = time.perf_counter() - started
    print(f"[synthetic] published={published} observed_rate={published / elapsed:.3f} FPS")


if __name__ == "__main__":
    main()

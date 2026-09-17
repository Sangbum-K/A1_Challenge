"""Loopback test for the latest-frame ROS2 BayerRG8 adapter."""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "live_camera_a2_21fps.py"
TOPIC = "/a1_a2_adapter_test/image_raw"


def load_runner():
    spec = importlib.util.spec_from_file_location("live_camera_a2_21fps", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    runner = load_runner()
    camera = runner.LatestROSFrame(TOPIC, "a1_a2_adapter_test_sub", expected_fps=21.0)
    camera.start()
    publisher_node = rclpy.create_node("a1_a2_adapter_test_pub")
    publisher = publisher_node.create_publisher(Image, TOPIC, qos_profile_sensor_data)
    try:
        deadline = time.perf_counter() + 3.0
        while publisher.get_subscription_count() == 0 and time.perf_counter() < deadline:
            time.sleep(0.01)
        if publisher.get_subscription_count() == 0:
            raise RuntimeError("loopback publisher did not discover the subscriber")

        bayer = np.zeros((8, 8), dtype=np.uint8)
        bayer[0::2, 0::2] = 220  # R sites in BayerRGGB
        bayer[0::2, 1::2] = 100  # G
        bayer[1::2, 0::2] = 100  # G
        bayer[1::2, 1::2] = 30   # B
        msg = Image()
        msg.header.stamp = publisher_node.get_clock().now().to_msg()
        msg.header.frame_id = "synthetic_bayer"
        msg.height = bayer.shape[0]
        msg.width = bayer.shape[1]
        msg.encoding = "bayer_rggb8"
        msg.is_bigendian = 0
        msg.step = bayer.shape[1]
        msg.data = bayer.tobytes()
        publisher.publish(msg)

        sample = camera.get_after(0, timeout=3.0)
        if sample is None:
            raise RuntimeError(camera.error or "no loopback image received")
        seq, source_stamp, host_received_at, frame = sample
        if frame.shape != (8, 8, 3) or frame.dtype != np.uint8:
            raise RuntimeError(f"unexpected debayered image {frame.shape} {frame.dtype}")
        print(
            f"[OK] seq={seq} source_stamp={source_stamp:.9f} "
            f"host_received_at={host_received_at:.9f} BGR_shape={frame.shape} "
            f"encoding={camera.negotiated['encoding']}",
            flush=True,
        )
    finally:
        publisher_node.destroy_node()
        camera.stop()


if __name__ == "__main__":
    main()

"""Fast camera aiming / exposure check. No model, no checkpoints, no CUDA.

The a2 path only ever sees the band y=roi_y0:roi_y1 of the transported frame.
Once the camera is mounted on a vehicle, whether the road and the lead car
land inside that band is decided by the lens, the mount pitch and the sensor
OffsetY -- none of which the model can compensate for.  This tool shows the
band live together with the three things that silently ruin an outdoor run:

  clipped   percentage of band pixels at/above 250 -- blown-out sky/road
  crushed   percentage of band pixels at/below 5  -- underexposed shadow
  focus     Laplacian variance of the band -- a lens focused on an indoor
            monitor is not focused at 30-80 m

Keys: w/s move the band, W/S move it fast, [ / ] change exposure, r reset,
space save a snapshot, p print the current numbers, q/Esc quit.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_params import DynamicCameraParams, step_exposure  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROI_Y0 = 235
ROI_Y1 = 485
WINDOW = "aim: full frame + model band"


def parse_args():
    ap = argparse.ArgumentParser(description="실차 장착용 카메라 조준/노출/초점 확인")
    ap.add_argument("--ros-topic", default="/camera_driver_gv/vis/image_raw")
    ap.add_argument("--ros-node-name", default="aim_camera")
    ap.add_argument("--roi-y0", type=int, default=ROI_Y0)
    ap.add_argument("--roi-y1", type=int, default=ROI_Y1)
    ap.add_argument("--sensor-offset-y", type=int, default=240,
                    help="launch의 offset_y. 센서 절대 행 번호를 같이 표시하기 위한 값")
    ap.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "aim"))
    ap.add_argument("--duration", type=float, default=0.0,
                    help="0이면 q를 누를 때까지")
    ap.add_argument("--driver-node", default="/camera_driver_gv",
                    help="노출을 실시간 변경할 camera driver node 이름")
    return ap.parse_args()


class LatestImage:
    def __init__(self, topic: str, node_name: str):
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image

        self._rclpy = rclpy
        self._bridge = CvBridge()
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node(node_name)
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        self.node.create_subscription(Image, topic, self._callback, qos)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.lock = threading.Lock()
        self.frame = None
        self.stamps = []
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def _spin(self):
        while self.running and self._rclpy.ok():
            self.executor.spin_once(timeout_sec=0.1)

    def _callback(self, msg):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.lock:
            self.frame = frame
            self.stamps.append(time.perf_counter())
            if len(self.stamps) > 60:
                del self.stamps[:-60]

    def read(self):
        with self.lock:
            fps = 0.0
            if len(self.stamps) >= 2:
                span = self.stamps[-1] - self.stamps[0]
                fps = (len(self.stamps) - 1) / span if span > 0 else 0.0
            return (None, fps) if self.frame is None else (self.frame.copy(), fps)

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()


def band_metrics(frame_bgr, y0, y1):
    band = frame_bgr[y0:y1]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    total = gray.size
    return {
        "mean": float(gray.mean()),
        "clipped_pct": 100.0 * float((gray >= 250).sum()) / total,
        "crushed_pct": 100.0 * float((gray <= 5).sum()) / total,
        "focus": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def verdict(metrics):
    if metrics["clipped_pct"] > 5.0:
        return "OVEREXPOSED - lower exposure or stop the lens down", (0, 0, 255)
    if metrics["crushed_pct"] > 20.0:
        return "UNDEREXPOSED - raise exposure or gain", (0, 140, 255)
    if metrics["focus"] < 60.0:
        return "SOFT - refocus the lens on a distant target", (0, 140, 255)
    return "OK", (0, 200, 0)


def draw(frame_bgr, y0, y1, metrics, fps, offset_y, exposure_us, exposure_hint):
    canvas = frame_bgr.copy()
    shade = canvas.copy()
    cv2.rectangle(shade, (0, 0), (canvas.shape[1], y0), (0, 0, 0), -1)
    cv2.rectangle(shade, (0, y1), (canvas.shape[1], canvas.shape[0]), (0, 0, 0), -1)
    canvas = cv2.addWeighted(shade, 0.55, canvas, 0.45, 0.0)
    cv2.rectangle(canvas, (0, y0), (canvas.shape[1] - 1, y1 - 1), (0, 255, 0), 2)
    # the 40 lateral bins the model splits this band into
    for index in range(1, 40):
        x = int(round(index * canvas.shape[1] / 40))
        colour = (0, 255, 255) if index in (16, 26) else (0, 110, 0)
        cv2.line(canvas, (x, y0), (x, y1), colour, 1)
    message, colour = verdict(metrics)
    exposure_text = (f"exposure {exposure_us:.0f} us" if exposure_us is not None
                     else f"exposure fixed ({exposure_hint})")
    lines = [
        f"band y={y0}:{y1} (h={y1 - y0})  sensor rows {offset_y + y0}:{offset_y + y1}",
        f"mean={metrics['mean']:.1f}  clipped={metrics['clipped_pct']:.2f}%  "
        f"crushed={metrics['crushed_pct']:.2f}%  focus={metrics['focus']:.0f}",
        exposure_text,
        f"camera {fps:.2f} FPS   [w/s] band  [ [ / ] ] exposure  [space] snapshot  [q] quit",
    ]
    for row, text in enumerate(lines):
        origin = (12, 26 + row * 26)
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    origin = (12, 26 + len(lines) * 26)
    cv2.putText(canvas, message, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(canvas, message, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.72, colour, 2, cv2.LINE_AA)
    return canvas


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    height = args.roi_y1 - args.roi_y0
    y0, y1 = args.roi_y0, args.roi_y1
    camera = LatestImage(args.ros_topic, args.ros_node_name)
    params = DynamicCameraParams(args.driver_node)
    print(f"[aim] {params.summary}", flush=True)
    exposure_us = params.get("ExposureTime") if params.supports("ExposureTime") else None
    started = time.perf_counter()
    last = None
    print(f"[aim] topic={args.ros_topic}", flush=True)
    try:
        while True:
            if args.duration > 0.0 and time.perf_counter() - started > args.duration:
                break
            frame, fps = camera.read()
            if frame is None:
                time.sleep(0.05)
                continue
            y1 = min(frame.shape[0], max(height, y1))
            y0 = y1 - height
            metrics = band_metrics(frame, y0, y1)
            last = (frame, metrics, fps)
            cv2.imshow(WINDOW, draw(frame, y0, y1, metrics, fps, args.sensor_offset_y,
                                    exposure_us, params.summary))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("w"):
                y1 = max(height, y1 - 5)
            elif key == ord("s"):
                y1 = min(frame.shape[0], y1 + 5)
            elif key == ord("W"):
                y1 = max(height, y1 - 40)
            elif key == ord("S"):
                y1 = min(frame.shape[0], y1 + 40)
            elif key == ord("r"):
                y0, y1 = args.roi_y0, args.roi_y1
            elif key == ord("p"):
                print(f"[aim] --roi-y0 {y0} --roi-y1 {y1}  {metrics}", flush=True)
            elif key in (ord("["), ord("]")) and exposure_us is not None:
                exposure_us = step_exposure(exposure_us, 1 if key == ord("]") else -1)
                params.set_async("ExposureTime", exposure_us)
            elif key == ord(" "):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                path = output_dir / f"aim_{stamp}.png"
                cv2.imwrite(str(path), frame)
                print(f"[aim] saved {path}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        cv2.destroyAllWindows()
    if last is not None:
        frame, metrics, fps = last
        result = {
            "roi_y": [y0, y1],
            "exposure_us": exposure_us,
            "sensor_rows": [args.sensor_offset_y + y0, args.sensor_offset_y + y1],
            "camera_fps": fps,
            "band": metrics,
            "verdict": verdict(metrics)[0],
        }
        path = output_dir / "aim_last.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        print(f"[aim] saved {path}", flush=True)
        keep = f"--roi-y0 {y0} --roi-y1 {y1}"
        if exposure_us is not None:
            keep += f" ; ./run.sh live --exposure {exposure_us:.0f}"
        print(f"[aim] 이 값을 유지하려면: {keep}", flush=True)


if __name__ == "__main__":
    main()

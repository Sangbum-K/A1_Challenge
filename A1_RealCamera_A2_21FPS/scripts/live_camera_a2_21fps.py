"""Physical-camera throughput test for the production 21-frame a2 path.

This script intentionally stops at a2. It does not load f1/f2, MORAI, a
raceline, or a pedal controller.

Pipeline:
  camera_aravis2/ROS2 BayerRG8 (or UVC prototype) -> debayer/BGR/RGB
  -> 250x1280 ROI -> ImageNet normalization
  -> segmentation-pretrained EfficientNetV2-S last feature
  -> raw i1 token (40,C) -> timestamp history (current + 20 target ages)
  -> A2ModelMultiFrame.forward_tokens -> a2 logits (40,7)

With --display, the exact a2 numbers are decoded into a diagnostic overlay:
40 horizontal object-probability strips, distance and both relative
velocity annotations, proximity bars, and the fixed center-interest bins
16..25.  These graphics are post-processing, not additional model outputs.

The capture thread keeps only the latest frame, so a slow model produces an
explicit drop count instead of silently processing an old camera queue.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import signal
import sys
import threading
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SENIOR_REPO = (PROJECT_ROOT / "senior_code").resolve()
SENIOR_A2_FILE = SENIOR_REPO / "src" / "model" / "image_models" / "a2_model.py"
if not SENIOR_A2_FILE.exists():
    raise RuntimeError(
        f"bundled senior source not found at {SENIOR_REPO}. "
        "Copy the complete A1_RealCamera_A2_21FPS folder again.")
_a2_spec = importlib.util.spec_from_file_location("senior_a2_model", SENIOR_A2_FILE)
if _a2_spec is None or _a2_spec.loader is None:
    raise RuntimeError(f"cannot load bundled senior module: {SENIOR_A2_FILE}")
_a2_module = importlib.util.module_from_spec(_a2_spec)
sys.modules[_a2_spec.name] = _a2_module
_a2_spec.loader.exec_module(_a2_module)
A2ModelMultiFrame = _a2_module.A2ModelMultiFrame

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_params import DynamicCameraParams, step_exposure  # noqa: E402


EXPECTED_H = 720
EXPECTED_W = 1280
ROI_Y0 = 235
ROI_Y1 = 485
N_BINS = 40
PREV_AGES_S = tuple(i * 0.05 for i in range(1, 21))
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
A2_SCALE_V = 20.0
A2_SCALE_DIST = 40.0
CENTER_BIN0 = 16
CENTER_BIN1 = 26  # exclusive; bins 16..25
DISPLAY_PANEL_H = 190
DISPLAY_WINDOW = "raw camera 1280x720 + a2 diagnostics"


def parse_args():
    ap = argparse.ArgumentParser(
        description="실물 camera_aravis2/ROS2 카메라 -> production mf20 a2 21 FPS 측정")
    ap.add_argument("--input", choices=["ros2", "uvc"], default="ros2",
                    help="최종 실험은 ros2; uvc는 기존 prototype 확인용")
    ap.add_argument("--ros-topic", default="/camera_driver_gv/vis/image_raw",
                    help="camera_aravis2 sensor_msgs/Image topic")
    ap.add_argument("--ros-node-name", default="live_camera_a2_21fps",
                    help="ROS2 subscriber node name")
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    ap.add_argument("--camera-backend", choices=["auto", "avfoundation", "v4l2"], default="auto")
    ap.add_argument("--width", type=int, default=EXPECTED_W)
    ap.add_argument("--height", type=int, default=EXPECTED_H)
    ap.add_argument("--camera-fps", type=float, default=21.0,
                    help="camera에 요청할 acquisition FPS")
    ap.add_argument("--target-fps", type=float, default=21.0,
                    help="pass/fail 기준 output FPS; 21 FPS budget=47.62 ms")
    ap.add_argument("--duration", type=float, default=60.0,
                    help="model warm-up 뒤 측정 시간(초). 0이면 q/Esc 또는 Ctrl-C까지 무제한; "
                         "실차 주행 run처럼 길이를 미리 모를 때 사용")
    ap.add_argument("--seg-ckpt", required=True,
                    help="seg-pretrained backbone checkpoint")
    ap.add_argument("--a2-ckpt", required=True,
                    help="A2ModelMultiFrame 21-frame checkpoint")
    ap.add_argument("--backbone", default="tf_efficientnetv2_s")
    ap.add_argument("--device", default="auto", help="auto|cuda|mps|cpu")
    ap.add_argument("--amp", action="store_true", help="CUDA fp16 autocast")
    ap.add_argument("--warmup-iters", type=int, default=10)
    ap.add_argument("--first-frame-timeout", type=float, default=5.0,
                    help="첫 frame을 기다리는 시간(초). bag replay처럼 publisher가 "
                         "model load보다 늦게 시작할 수 있으면 늘린다")
    ap.add_argument("--roi-y0", type=int, default=ROI_Y0)
    ap.add_argument("--roi-y1", type=int, default=ROI_Y1)
    ap.add_argument("--allow-resize", action="store_true",
                    help="camera가 요청 해상도를 주지 않을 때 resize. geometry가 달라져 정확도 평가는 무효")
    ap.add_argument("--geometry-valid-for-accuracy", action="store_true",
                    help="calibration/FOV/mount까지 검증된 경우만 명시. 단순 shape 일치는 불충분")
    ap.add_argument("--geometry-note", default="uncalibrated physical-camera geometry; throughput only",
                    help="summary.json에 기록할 geometry 근거/제약")
    ap.add_argument("--ego-vx", type=float, default=0.0,
                    help="A2ModelMultiFrame ego_motion[0] 종방향 속도 m/s. 카메라가 실제로 "
                         "움직이는 주행 run에서 0을 주면 학습 조건과 어긋난다")
    ap.add_argument("--ego-wz", type=float, default=0.0,
                    help="A2ModelMultiFrame ego_motion[1] yaw rate rad/s")
    ap.add_argument("--obj-thr", type=float, default=0.5)
    ap.add_argument("--display-min-prob", type=float, default=0.10,
                    help="overlay에 희미하게 표시할 최소 object probability")
    ap.add_argument("--display", action="store_true",
                    help="실시간 RGB + a2 40-bin 진단 overlay")
    ap.add_argument("--display-every", type=int, default=1,
                    help="화면만 켠 경우 N번째 A2 출력마다 화면 갱신; 모델/로그는 매 frame 처리")
    ap.add_argument("--tune-exposure", action="store_true",
                    help="display 중 [ / ] 키로 카메라 노출을 실시간 조절하고 "
                         "모델 band의 포화/암부 비율을 표시한다")
    ap.add_argument("--driver-node", default="/camera_driver_gv",
                    help="--tune-exposure가 파라미터를 설정할 camera driver node 이름")
    ap.add_argument("--save-overlay-video", action="store_true",
                    help="overlay를 output-dir/a2_overlay.mp4에 저장; throughput baseline에서는 끄기")
    ap.add_argument("--video-fps", type=float, default=0.0,
                    help="저장 영상 FPS. 0이면 --camera-fps 사용")
    ap.add_argument("--log-full-a2", action="store_true",
                    help="각 frame의 raw (40,7) a2와 decoded 40-bin 값을 frames.jsonl에 저장")
    ap.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "real_camera_a2_21fps"))
    ap.add_argument("--allow-partial-load", action="store_true",
                    help="필수 checkpoint key 누락을 경고만 하고 실행; 정식 실험에서는 사용 금지")
    return ap.parse_args()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def autocast_context(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def unwrap_state_dict(obj):
    if not isinstance(obj, dict):
        raise RuntimeError(f"checkpoint must be a dict, got {type(obj).__name__}")
    for key in ("model", "state_dict", "state"):
        if key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    return {str(k): v for k, v in obj.items() if torch.is_tensor(v)}


def key_candidates(key: str):
    base = key[len("module."):] if key.startswith("module.") else key
    out = [key, base]
    for prefix in ("backbone.", "encoder.backbone.", "model.backbone."):
        if base.startswith(prefix):
            out.append(base[len(prefix):])
    return list(dict.fromkeys(out))


def load_backbone(name: str, ckpt_path: str, device: torch.device, allow_partial: bool):
    backbone = timm.create_model(
        name, pretrained=False, features_only=True, out_indices=[-1])
    own = backbone.state_dict()
    incoming = unwrap_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    matched = {}
    for key, value in incoming.items():
        for candidate in key_candidates(key):
            if candidate in own and own[candidate].shape == value.shape:
                matched[candidate] = value
                break
    coverage = sum(own[k].numel() for k in matched) / max(1, sum(v.numel() for v in own.values()))
    if coverage < 0.95 and not allow_partial:
        raise RuntimeError(
            f"seg backbone checkpoint coverage={coverage:.1%}; "
            "expected >=95%. Wrong architecture/checkpoint?")
    missing, unexpected = backbone.load_state_dict(matched, strict=False)
    info = {
        "matched_keys": len(matched),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "parameter_coverage": coverage,
    }
    return backbone.to(device).eval(), info


def load_a2(name: str, ckpt_path: str, device: torch.device, allow_partial: bool):
    # The untouched senior constructor requests timm pretrained weights even
    # though forward_tokens() never uses its internal image backbone. Avoid a
    # network/cache dependency on a fresh Ubuntu PC; exact bundled checkpoints
    # are loaded immediately below and the live tokenizer has its own seg model.
    original_create_model = timm.create_model

    def create_without_download(*create_args, **create_kwargs):
        create_kwargs["pretrained"] = False
        return original_create_model(*create_args, **create_kwargs)

    timm.create_model = create_without_download
    try:
        model = A2ModelMultiFrame(
            img_backbone=name,
            d_token=256,
            n_heads=4,
            n_ch=7,
            d_out=128,
            n_frames=21,
        )
    finally:
        timm.create_model = original_create_model
    own = model.state_dict()
    incoming = unwrap_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    matched = {}
    for key, value in incoming.items():
        base = key[len("module."):] if key.startswith("module.") else key
        candidates = [base]
        if base.startswith("a2."):
            candidates.append(base[len("a2."):])
        for candidate in candidates:
            if candidate in own and own[candidate].shape == value.shape:
                matched[candidate] = value
                break
    required_prefixes = ("encoder.proj.", "fusion.", "fc_proj.", "head.")
    missing_required = [
        k for k in own
        if k.startswith(required_prefixes) and k not in matched
    ]
    if missing_required and not allow_partial:
        raise RuntimeError(
            "a2 checkpoint is missing production keys: " + ", ".join(missing_required[:10]))
    missing, unexpected = model.load_state_dict(matched, strict=False)
    info = {
        "matched_keys": len(matched),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "missing_required": missing_required,
    }
    return model.to(device).eval(), info


def preprocess(frame_bgr: np.ndarray, args, device: torch.device):
    h, w = frame_bgr.shape[:2]
    resized = False
    if (h, w) != (args.height, args.width):
        if not args.allow_resize:
            raise RuntimeError(
                f"camera returned {w}x{h}, expected {args.width}x{args.height}. "
                "Fix camera mode or pass --allow-resize for throughput-only testing.")
        frame_bgr = cv2.resize(frame_bgr, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
        resized = True
    if not (0 <= args.roi_y0 < args.roi_y1 <= frame_bgr.shape[0]):
        raise RuntimeError(f"invalid ROI y={args.roi_y0}:{args.roi_y1} for {frame_bgr.shape}")
    roi_rgb = cv2.cvtColor(frame_bgr[args.roi_y0:args.roi_y1], cv2.COLOR_BGR2RGB)
    arr = roi_rgb.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    # model_frame_bgr is the exact geometry used for inference and therefore
    # the only correct canvas for a 40-bin overlay.
    return tensor, resized, frame_bgr


def percentile(values, q):
    if not values:
        return math.nan
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LatestFrameCamera:
    """Continuously capture and expose only the newest frame."""

    def __init__(self, index: int, backend: str, width: int, height: int, fps: float):
        backend_id = {
            "auto": cv2.CAP_ANY,
            "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY),
            "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
        }[backend]
        self.cap = cv2.VideoCapture(index, backend_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera index {index} with backend={backend}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._negotiated = {
            "transport": "OpenCV UVC prototype",
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps_reported": float(self.cap.get(cv2.CAP_PROP_FPS)),
            "backend": self.cap.getBackendName() if hasattr(self.cap, "getBackendName") else "unknown",
        }
        self.condition = threading.Condition()
        self.latest = None
        self.seq = 0
        self.running = False
        self.error = None
        self.capture_times = deque(maxlen=max(5000, int(fps * 600)))
        self.thread = None

    @property
    def negotiated(self):
        return dict(self._negotiated)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, name="camera-capture", daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            captured_at = time.perf_counter()
            if not ok:
                self.error = "camera read failed"
                with self.condition:
                    self.condition.notify_all()
                break
            with self.condition:
                self.seq += 1
                # UVC has no sensor/header timestamp. Use the host capture-
                # completion clock for both history selection and latency.
                self.latest = (self.seq, captured_at, captured_at, frame)
                self.capture_times.append(captured_at)
                self.condition.notify_all()

    def get_after(self, seq: int, timeout: float = 2.0):
        deadline = time.perf_counter() + timeout
        with self.condition:
            while self.running and not self.error and (self.latest is None or self.latest[0] <= seq):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
            if self.error or self.latest is None or self.latest[0] <= seq:
                return None
            return self.latest

    def reset_measurement(self):
        with self.condition:
            self.capture_times.clear()
            return self.latest[0] if self.latest is not None else 0

    def stop(self):
        self.running = False
        self.cap.release()
        with self.condition:
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    @property
    def timestamp_note(self):
        return (
            "OpenCV supplies a host capture-completion timestamp, not the sensor exposure "
            "timestamp; true sensor-to-a2 latency requires the ROS2 camera path."
        )


class LatestROSFrame:
    """Subscribe with sensor-data QoS and retain only the latest ROS image."""

    def __init__(self, topic: str, node_name: str, expected_fps: float):
        try:
            import rclpy
            from cv_bridge import CvBridge
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise RuntimeError(
                "ROS2 Python modules are unavailable. Source /opt/ros/humble/setup.bash "
                "before activating the project .venv."
            ) from exc

        self._rclpy = rclpy
        self._executor_type = SingleThreadedExecutor
        self._bridge = CvBridge()
        self._topic = topic
        self._node_name = node_name
        self._image_type = Image
        self._qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.condition = threading.Condition()
        self.latest = None
        self.seq = 0
        self.running = False
        self.error = None
        self.capture_times = deque(maxlen=max(5000, int(expected_fps * 600)))
        self.thread = None
        self.node = None
        self.executor = None
        self.subscription = None
        self._negotiated = {
            "transport": "camera_aravis2/ROS2",
            "topic": topic,
            "qos": "sensor-data keep-last depth=1 best-effort",
        }

    @property
    def negotiated(self):
        return dict(self._negotiated)

    @property
    def timestamp_note(self):
        return (
            "History ages use the ROS header timestamp. capture_to_a2_ms_host starts at "
            "subscriber callback arrival; capture_to_a2_ms_header uses the ROS header clock "
            "and is meaningful only when camera/PTP and host clocks are synchronized."
        )

    def start(self):
        if not self._rclpy.ok():
            self._rclpy.init(args=None)
        self.node = self._rclpy.create_node(self._node_name)
        self.subscription = self.node.create_subscription(
            self._image_type, self._topic, self._callback, self._qos)
        self.executor = self._executor_type()
        self.executor.add_node(self.node)
        self.running = True
        self.thread = threading.Thread(target=self._spin, name="ros2-image-subscription", daemon=True)
        self.thread.start()

    def _spin(self):
        try:
            while self.running and self._rclpy.ok():
                self.executor.spin_once(timeout_sec=0.1)
        except Exception as exc:  # surfaced to the inference thread
            self.error = f"ROS2 executor failed: {exc}"
            with self.condition:
                self.condition.notify_all()

    def _callback(self, msg):
        host_received_at = time.perf_counter()
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise RuntimeError(
                    f"cv_bridge returned dtype={frame.dtype}, shape={frame.shape}; expected BGR8")
            header_stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            source_stamp = header_stamp if header_stamp > 0.0 else host_received_at
        except Exception as exc:
            self.error = f"cannot convert ROS image encoding={msg.encoding!r}: {exc}"
            with self.condition:
                self.condition.notify_all()
            return
        with self.condition:
            self.seq += 1
            self.latest = (self.seq, source_stamp, host_received_at, frame)
            self.capture_times.append(host_received_at)
            self._negotiated.update({
                "width": int(msg.width),
                "height": int(msg.height),
                "encoding": str(msg.encoding),
                "step": int(msg.step),
                "frame_id": str(msg.header.frame_id),
                "header_timestamp_available": header_stamp > 0.0,
            })
            self.condition.notify_all()

    def get_after(self, seq: int, timeout: float = 2.0):
        deadline = time.perf_counter() + timeout
        with self.condition:
            while self.running and not self.error and (self.latest is None or self.latest[0] <= seq):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
            if self.error or self.latest is None or self.latest[0] <= seq:
                return None
            return self.latest

    def reset_measurement(self):
        with self.condition:
            self.capture_times.clear()
            return self.latest[0] if self.latest is not None else 0

    def stop(self):
        self.running = False
        with self.condition:
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.executor is not None and self.node is not None:
            self.executor.remove_node(self.node)
        if self.node is not None:
            self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()


def select_history(history, current_time: float):
    frames = []
    age_errors_ms = []
    source_seqs = []
    for target_age in PREV_AGES_S:
        seq, timestamp, token = min(
            history, key=lambda item: abs((current_time - item[1]) - target_age))
        actual_age = current_time - timestamp
        frames.append(token.unsqueeze(0))
        age_errors_ms.append(abs(actual_age - target_age) * 1000.0)
        source_seqs.append(seq)
    return frames, age_errors_ms, source_seqs


def decode_a2(logits: torch.Tensor):
    """Copy one (1,40,7) output to CPU and decode the senior label scales.

    Channel order is kept explicit so the visualization cannot silently drift
    from the training contract.  No thresholding or display clipping changes
    these decoded values.
    """
    raw = logits[0].detach().float().cpu().numpy()
    if raw.shape != (N_BINS, 7):
        raise RuntimeError(f"unexpected a2 shape {raw.shape}; expected ({N_BINS}, 7)")
    object_probability = 1.0 / (1.0 + np.exp(-np.clip(raw[:, 0], -80.0, 80.0)))
    return {
        "raw": raw,
        "object_probability": object_probability.astype(np.float32),
        "v_long_abs_mps": raw[:, 1] * A2_SCALE_V,
        "v_lat_abs_mps": raw[:, 2] * A2_SCALE_V,
        "v_long_rel_mps": raw[:, 3] * A2_SCALE_V,
        "v_lat_rel_mps": raw[:, 4] * A2_SCALE_V,
        "proximity": raw[:, 5],
        "distance_m": raw[:, 6] * A2_SCALE_DIST,
    }


def empty_decoded_a2():
    return {
        "raw": np.full((N_BINS, 7), np.nan, dtype=np.float32),
        "object_probability": np.zeros(N_BINS, dtype=np.float32),
        "v_long_abs_mps": np.full(N_BINS, np.nan, dtype=np.float32),
        "v_lat_abs_mps": np.full(N_BINS, np.nan, dtype=np.float32),
        "v_long_rel_mps": np.full(N_BINS, np.nan, dtype=np.float32),
        "v_lat_rel_mps": np.full(N_BINS, np.nan, dtype=np.float32),
        "proximity": np.zeros(N_BINS, dtype=np.float32),
        "distance_m": np.full(N_BINS, np.nan, dtype=np.float32),
    }


def validate_runtime_contract(image, feature, token, logits, args):
    expected_image = (1, 3, args.roi_y1 - args.roi_y0, args.width)
    if tuple(image.shape) != expected_image:
        raise RuntimeError(f"image tensor shape {tuple(image.shape)} != {expected_image}")
    if feature.ndim != 4 or feature.shape[0] != 1 or feature.shape[1] != 256:
        raise RuntimeError(
            f"backbone feature shape {tuple(feature.shape)}; expected (1,256,H,40-like)")
    if tuple(token.shape) != (1, N_BINS, 256):
        raise RuntimeError(f"raw i1 token shape {tuple(token.shape)} != (1,{N_BINS},256)")
    if tuple(logits.shape) != (1, N_BINS, 7):
        raise RuntimeError(f"a2 output shape {tuple(logits.shape)} != (1,{N_BINS},7)")


def contiguous_groups(mask: np.ndarray, start: int = 0, stop: int = N_BINS):
    """Return [start, stop) groups for adjacent true bins in a slice."""
    groups = []
    group_start = None
    for index in range(start, stop):
        if bool(mask[index]) and group_start is None:
            group_start = index
        if group_start is not None and (not bool(mask[index]) or index == stop - 1):
            group_stop = index if not bool(mask[index]) else index + 1
            groups.append((group_start, group_stop))
            group_start = None
    return groups


def summarize_group(decoded, start: int, stop: int):
    prob = decoded["object_probability"][start:stop]
    peak_bin = start + int(np.argmax(prob))
    weights = np.maximum(prob.astype(np.float64), 1e-6)
    weights /= weights.sum()

    def weighted(key):
        values = decoded[key][start:stop].astype(np.float64)
        finite = np.isfinite(values)
        if not bool(finite.any()):
            return None
        finite_weights = weights[finite]
        finite_weights /= finite_weights.sum()
        return float(np.sum(values[finite] * finite_weights))

    return {
        "start_bin": start,
        "stop_bin": stop,
        "peak_bin": peak_bin,
        "object_probability": float(decoded["object_probability"][peak_bin]),
        "distance_m": weighted("distance_m"),
        "v_long_rel_mps": weighted("v_long_rel_mps"),
        "v_lat_rel_mps": weighted("v_lat_rel_mps"),
        "proximity": weighted("proximity"),
    }


def detected_objects(decoded, obj_thr: float, start: int = 0, stop: int = N_BINS):
    mask = decoded["object_probability"] >= obj_thr
    groups = contiguous_groups(mask, start=start, stop=stop)
    return [summarize_group(decoded, begin, end) for begin, end in groups]


def a2_summary(decoded, obj_thr: float):
    """Choose a central lead candidate without treating every regression as valid.

    Regression channels are reported only inside an object-positive contiguous
    group.  If more than one central group exists, the nearest non-negative
    predicted distance wins; otherwise the highest-probability group wins.
    """
    objects = detected_objects(decoded, obj_thr, CENTER_BIN0, CENTER_BIN1)
    if objects:
        physically_valid = [
            obj for obj in objects
            if obj["distance_m"] is not None and obj["distance_m"] >= 0.0
        ]
        if physically_valid:
            chosen = min(physically_valid, key=lambda obj: obj["distance_m"])
        else:
            chosen = max(objects, key=lambda obj: obj["object_probability"])
        return {
            "lead": True,
            "bin": chosen["peak_bin"],
            "bin_span": [chosen["start_bin"], chosen["stop_bin"] - 1],
            "object_probability": chosen["object_probability"],
            "distance_m": chosen["distance_m"],
            "v_long_rel_mps": chosen["v_long_rel_mps"],
            "v_lat_rel_mps": chosen["v_lat_rel_mps"],
            "proximity": chosen["proximity"],
        }
    center_max = float(np.max(decoded["object_probability"][CENTER_BIN0:CENTER_BIN1]))
    return {
        "lead": False,
        "bin": None,
        "bin_span": None,
        "object_probability": center_max,
        "distance_m": None,
        "v_long_rel_mps": None,
        "v_lat_rel_mps": None,
        "proximity": None,
    }


def _bin_x(index: int, width: int):
    return int(round(index * width / N_BINS))


def _put_text_box(image, text, origin, color=(255, 255, 255), scale=0.52, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    x = max(0, min(x, image.shape[1] - text_w - 8))
    y = max(text_h + 7, min(y, image.shape[0] - baseline - 4))
    cv2.rectangle(image, (x - 4, y - text_h - 5),
                  (x + text_w + 4, y + baseline + 3), (20, 20, 20), -1)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _draw_vrel_arrow(image, center, value):
    """Draw the display convention: approaching=red left, receding=blue right."""
    if value is None or not math.isfinite(value) or abs(value) < 0.15:
        cv2.circle(image, center, 4, (180, 180, 180), -1, cv2.LINE_AA)
        return
    length = 28
    if value < 0.0:
        start, end, color = (center[0] + length, center[1]), center, (0, 0, 255)
    else:
        start, end, color = (center[0] - length, center[1]), center, (255, 100, 0)
    cv2.arrowedLine(image, start, end, color, 3, cv2.LINE_AA, tipLength=0.35)


def render_a2_overlay(frame_bgr: np.ndarray, decoded, record, args, fps_recent: float,
                      status_text: str = None):
    """Render diagnostics over the full, unresized camera frame.

    The green outline is the actual model crop.  Red tint represents only the
    lateral A2 bins and must not be interpreted as a 2-D detection box.
    """
    shown = frame_bgr.copy()
    height, width = shown.shape[:2]
    roi_y0, roi_y1 = args.roi_y0, args.roi_y1
    prob = decoded["object_probability"]

    # A2 retained only lateral location, so each probability is drawn as a
    # light vertical strip across the model ROI rather than as a fake 2-D box.
    # Keeping the tint light preserves the appearance of the received frame.
    for index, value in enumerate(prob):
        if value < args.display_min_prob:
            continue
        x0, x1 = _bin_x(index, width), _bin_x(index + 1, width)
        alpha = min(0.32, 0.04 + 0.28 * float(value))
        region = shown[roi_y0:roi_y1, x0:x1]
        red = np.zeros_like(region)
        red[:, :, 2] = 255
        cv2.addWeighted(red, alpha, region, 1.0 - alpha, 0.0, dst=region)

    # The whole received 1280x720 frame stays visible.  Only this full-width
    # 1280x250 region is cropped and normalized for the backbone.
    cv2.rectangle(shown, (0, roi_y0), (width - 1, roi_y1 - 1), (0, 255, 0), 2)
    _put_text_box(
        shown,
        f"MODEL INPUT CROP x=0:{width}, y={roi_y0}:{roi_y1} "
        f"({width}x{roi_y1 - roi_y0})",
        (12, roi_y0 - 8),
        color=(80, 255, 80),
        scale=0.52,
    )

    # One label per contiguous object-positive group avoids printing 5-10
    # duplicate distance values on a single wide vehicle.
    objects = sorted(
        detected_objects(decoded, args.obj_thr),
        key=lambda obj: obj["object_probability"], reverse=True)[:6]
    for row, obj in enumerate(objects):
        peak = obj["peak_bin"]
        x = (_bin_x(obj["start_bin"], width) + _bin_x(obj["stop_bin"], width)) // 2
        label_y = roi_y0 + 25 + (row % 3) * 30
        distance = obj["distance_m"]
        vrel = obj["v_long_rel_mps"]
        vlat = obj["v_lat_rel_mps"]
        distance_text = "d=?" if distance is None else f"d={distance:.1f}m"
        vrel_text = "vL=?" if vrel is None else f"vL={vrel:+.1f}"
        vlat_text = "vT=?" if vlat is None else f"vT={vlat:+.1f}"
        _put_text_box(shown, f"b{peak} p={obj['object_probability']:.2f} {distance_text} "
                      f"{vrel_text} {vlat_text}m/s",
                      (x - 110, label_y), color=(255, 255, 255), scale=0.46)
        cv2.line(shown, (x, label_y + 7), (x, roi_y1 - 8), (255, 255, 255), 1, cv2.LINE_AA)
        _draw_vrel_arrow(shown, (x, min(roi_y1 - 15, label_y + 20)), vrel)

    # Top status is intentionally based on pre-render timestamps. Rendering
    # cost is reported separately and does not get mislabeled as model compute.
    steady = record["history_span_s"] >= PREV_AGES_S[-1]
    state = "READY 21/21" if steady else f"HISTORY WARMUP {record['history_span_s']:.2f}/1.00s"
    top_lines = [
        f"RAW CAMERA INPUT {width}x{height} | full frame shown | software resize: "
        f"{'YES' if record['resized'] else 'NO'}",
        f"A2 output {fps_recent:5.1f} FPS | model compute {record['compute_ms']:5.1f} ms | "
        f"capture->A2 {record['capture_to_a2_ms_host']:5.1f} ms",
        f"{state} | unique past {record['history_unique_sources']:02d}/20 | "
        f"skipped total {record['skipped_total']} | obj threshold {args.obj_thr:.2f}",
    ]
    if status_text:
        top_lines.append(status_text)
    # Show the selected peak bin's seven model channels without group averaging.
    # All 40 bins are saved separately when --log-full-a2 is enabled.
    peak = record["a2"].get("bin")
    if peak is not None:
        raw = decoded["raw"][peak]
        top_lines.append(
            f"PEAK b{peak:02d} RAW a2 [obj,Labs,Tabs,Lrel,Trel,prox,dist] = "
            + ",".join(f"{float(value):+.2f}" for value in raw))
        top_lines.append(
            f"DECODED b{peak:02d} p={decoded['object_probability'][peak]:.2f} "
            f"abs(L,T)=({decoded['v_long_abs_mps'][peak]:+.1f},"
            f"{decoded['v_lat_abs_mps'][peak]:+.1f}) "
            f"rel(L,T)=({decoded['v_long_rel_mps'][peak]:+.1f},"
            f"{decoded['v_lat_rel_mps'][peak]:+.1f})m/s "
            f"prox={decoded['proximity'][peak]:.2f} "
            f"d={decoded['distance_m'][peak]:.1f}m")
    for line_index, line in enumerate(top_lines):
        _put_text_box(shown, line, (15, 28 + line_index * 28),
                      color=(80, 255, 80) if line_index <= 1 else (255, 255, 255), scale=0.56)

    panel = np.full((DISPLAY_PANEL_H, width, 3), 20, dtype=np.uint8)
    cell_y0, cell_y1 = 34, 70
    for index, value in enumerate(prob):
        x0, x1 = _bin_x(index, width), _bin_x(index + 1, width)
        red = int(round(40 + 215 * min(1.0, max(0.0, float(value)))))
        cv2.rectangle(panel, (x0, cell_y0), (max(x0, x1 - 1), cell_y1), (20, 20, red), -1)
        if index % 5 == 0:
            cv2.putText(panel, str(index), (x0 + 2, cell_y1 + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.rectangle(panel, (_bin_x(CENTER_BIN0, width), cell_y0),
                  (_bin_x(CENTER_BIN1, width) - 1, cell_y1), (0, 255, 255), 2)
    cv2.putText(panel, "OBJECT PROBABILITY (sigmoid a2[:,0])", (12, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)

    prox = np.clip(np.nan_to_num(decoded["proximity"], nan=0.0), 0.0, 1.0)
    bar_base, bar_top = 168, 105
    for index, value in enumerate(prox):
        x0, x1 = _bin_x(index, width), _bin_x(index + 1, width)
        y = int(round(bar_base - float(value) * (bar_base - bar_top)))
        cv2.rectangle(panel, (x0 + 2, y), (max(x0 + 2, x1 - 2), bar_base), (40, 210, 210), -1)
    cv2.putText(panel, "PROXIMITY (a2[:,5], display-clamped 0..1)", (12, 99),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(
        panel,
        "yellow outline = lead-search bins 16..25 (not crop/box/lane) | red left: approaching | blue right: receding",
        (12, 186), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190, 190, 190), 1, cv2.LINE_AA)
    return np.vstack((shown, panel))


def band_clipping(frame_bgr: np.ndarray, y0: int, y1: int):
    """How much of the band the model sees is blown out or crushed.

    Outdoors this is the difference between a working run and 40 bins of
    white, and it is not obvious on a laptop screen in sunlight.
    """
    gray = cv2.cvtColor(frame_bgr[y0:y1], cv2.COLOR_BGR2GRAY)
    total = max(1, gray.size)
    return (100.0 * float((gray >= 250).sum()) / total,
            100.0 * float((gray <= 5).sum()) / total)


def ego_motion_tensor(args, device, dtype=torch.float32):
    """A2ModelMultiFrame ego_motion (1,2) = [vx, wz].

    The fusion block feeds this through ego_proj into the cross-attention KV
    set, so it is a real model input, not a placeholder.  A static camera is
    correctly described by zeros; a camera on a moving vehicle is not.
    """
    return torch.tensor([[args.ego_vx, args.ego_wz]], dtype=dtype, device=device)


def warmup(frame, args, device, backbone, a2):
    image, _, _ = preprocess(frame, args, device)
    amp_enabled = args.amp and device.type == "cuda"
    for _ in range(args.warmup_iters):
        with torch.inference_mode(), autocast_context(device, amp_enabled):
            feature = backbone(image)[-1]
            token = F.adaptive_avg_pool2d(feature, (1, N_BINS)).squeeze(2).transpose(1, 2)
            a2.forward_tokens([token] * 21, ego_motion_tensor(args, device, token.dtype))
    sync_device(device)


def summarize(records, camera, args, model_info, resized_any, nan_frames):
    steady = [r for r in records if r["history_span_s"] >= PREV_AGES_S[-1]]
    basis = steady if steady else records
    completion = [r["completed_at"] for r in basis]
    elapsed = completion[-1] - completion[0] if len(completion) >= 2 else 0.0
    output_fps = (len(completion) - 1) / elapsed if elapsed > 0 else 0.0
    cap_times = list(camera.capture_times)
    cap_elapsed = cap_times[-1] - cap_times[0] if len(cap_times) >= 2 else 0.0
    capture_fps = (len(cap_times) - 1) / cap_elapsed if cap_elapsed > 0 else 0.0
    compute = [r["compute_ms"] for r in basis]
    capture_to_a2 = [r["capture_to_a2_ms_host"] for r in basis]
    capture_to_a2_header = [
        r["capture_to_a2_ms_header"] for r in basis
        if r["capture_to_a2_ms_header"] is not None
    ]
    capture_to_output = [r["capture_to_output_ms_host"] for r in basis]
    preprocess = [r["preprocess_ms"] for r in basis]
    backbone = [r["backbone_ms"] for r in basis]
    a2_latency = [r["a2_ms"] for r in basis]
    postprocess = [r["postprocess_ms"] for r in basis]
    visualization = [r["visualization_ms"] for r in basis if r["visualization_ms"] is not None]
    intervals = [
        (completion[i] - completion[i - 1]) * 1000.0
        for i in range(1, len(completion))
    ]
    capture_intervals = [
        (cap_times[i] - cap_times[i - 1]) * 1000.0
        for i in range(1, len(cap_times))
    ]
    age_errors = [x for r in basis for x in r["history_age_error_ms"]]
    duplicate_slots = sum(20 - r["history_unique_sources"] for r in basis)
    duplicate_ratio = duplicate_slots / max(1, 20 * len(basis))
    skipped = sum(r["skipped_since_previous"] for r in records)
    seen = len(records) + skipped
    drop_ratio = skipped / max(1, seen)
    budget_ms = 1000.0 / args.target_fps
    tolerance_fps = args.target_fps * 0.98
    compute_pass = percentile(compute, 95) <= budget_ms
    capture_pass = capture_fps >= tolerance_fps
    output_pass = output_fps >= tolerance_fps
    end_to_end_pass = capture_pass and output_pass and compute_pass and nan_frames == 0
    if not capture_pass:
        bottleneck = "camera acquisition below target"
    elif not compute_pass:
        bottleneck = "backbone+a2 compute exceeds frame budget"
    elif not output_pass:
        bottleneck = "pipeline output throughput below target"
    else:
        bottleneck = "none detected"
    def latency_stats(values):
        return {
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
            "p99": percentile(values, 99),
        }
    return {
        "scope": "physical camera -> i1 -> 21-frame a2 only",
        "senior_repo": str(SENIOR_REPO),
        "senior_a2_source": str(SENIOR_A2_FILE),
        "senior_a2_source_sha256": model_info["senior_a2_source_sha256"],
        "seg_checkpoint_sha256": model_info["seg_checkpoint_sha256"],
        "a2_checkpoint_sha256": model_info["a2_checkpoint_sha256"],
        "target_fps": args.target_fps,
        "object_threshold": args.obj_thr,
        "frame_budget_ms": budget_ms,
        "roi_y": [args.roi_y0, args.roi_y1],
        "ego_motion_input": {
            "vx_mps": args.ego_vx,
            "wz_radps": args.ego_wz,
            "note": (
                "ego_motion is a real A2ModelMultiFrame input. Zeros are only correct "
                "for a stationary camera; a moving-vehicle run with zeros is a "
                "train/deploy mismatch on the velocity channels."
            ),
        },
        "camera": camera.negotiated,
        "device": str(model_info["device"]),
        "amp": bool(args.amp and model_info["device"].type == "cuda"),
        "seg_load": model_info["seg_load"],
        "a2_load": model_info["a2_load"],
        "measured_frames_total": len(records),
        "measured_frames_steady": len(steady),
        "capture_fps_host_observed": capture_fps,
        "a2_output_fps_steady": output_fps,
        "capture_to_a2_ms_host": latency_stats(capture_to_a2),
        "capture_to_a2_ms_header": latency_stats(capture_to_a2_header),
        "capture_to_output_ms_host": latency_stats(capture_to_output),
        "compute_ms": latency_stats(compute),
        "stage_ms": {
            "preprocess": latency_stats(preprocess),
            "backbone": latency_stats(backbone),
            "a2": latency_stats(a2_latency),
            "a2_decode_and_summary": latency_stats(postprocess),
            "visualization": latency_stats(visualization),
        },
        "camera_interval_ms": latency_stats(capture_intervals),
        "output_interval_ms": latency_stats(intervals),
        "history_age_error_ms": {
            "p50": percentile(age_errors, 50),
            "p95": percentile(age_errors, 95),
            "max": max(age_errors) if age_errors else math.nan,
        },
        "skipped_camera_frames": skipped,
        "drop_ratio": drop_ratio,
        "history_duplicate_slot_ratio": duplicate_ratio,
        "nan_or_inf_frames": nan_frames,
        "resized_any_frame": resized_any,
        "geometry_valid_for_accuracy": bool(
            args.geometry_valid_for_accuracy and not resized_any),
        "geometry_note": args.geometry_note,
        "visualization": {
            "display_enabled": bool(args.display),
            "display_every": args.display_every,
            "video_enabled": bool(args.save_overlay_video),
            "full_a2_log_enabled": bool(args.log_full_a2),
            "timing_note": (
                "model compute excludes visualization; steady output FPS still includes the prior "
                "frame's rendering/window/video-write overhead when visualization is enabled."
            ),
        },
        "pass": {
            "camera_acquisition": capture_pass,
            "model_p95_under_budget": compute_pass,
            "a2_output_rate": output_pass,
            "sustained_a2_21fps": end_to_end_pass,
        },
        "likely_bottleneck": bottleneck,
        "timestamp_note": camera.timestamp_note,
    }


def main():
    args = parse_args()
    if not 0.0 <= args.obj_thr <= 1.0:
        raise ValueError("--obj-thr must be in [0,1]")
    if not 0.0 <= args.display_min_prob <= 1.0:
        raise ValueError("--display-min-prob must be in [0,1]")
    if args.target_fps <= 0.0 or args.camera_fps <= 0.0:
        raise ValueError("FPS arguments must be positive")
    if args.duration < 0.0:
        raise ValueError("--duration must be >= 0 (0 means run until q/Esc or Ctrl-C)")
    if args.display_every < 1:
        raise ValueError("--display-every must be >= 1")
    duration = args.duration if args.duration > 0.0 else math.inf
    if args.roi_y1 - args.roi_y0 != ROI_Y1 - ROI_Y0:
        print(f"[warn] ROI height {args.roi_y1 - args.roi_y0} != trained {ROI_Y1 - ROI_Y0}; "
              "object scale and lateral-bin geometry no longer match training",
              flush=True)
    if args.display and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("[warn] no DISPLAY/WAYLAND_DISPLAY; disabling --display. "
              "Add --save-overlay-video to keep a visual record.", flush=True)
        args.display = False
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    if args.amp and device.type != "cuda":
        print("[warn] --amp is enabled only on CUDA; running fp32", flush=True)

    backbone, seg_load = load_backbone(
        args.backbone, args.seg_ckpt, device, args.allow_partial_load)
    a2, a2_load = load_a2(
        args.backbone, args.a2_ckpt, device, args.allow_partial_load)
    for module in (backbone, a2):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    if args.input == "ros2":
        camera = LatestROSFrame(args.ros_topic, args.ros_node_name, args.camera_fps)
    else:
        camera = LatestFrameCamera(
            args.camera, args.camera_backend, args.width, args.height, args.camera_fps)
    print(f"[camera] negotiated={camera.negotiated}", flush=True)
    print(f"[model] device={device} seg={seg_load} a2={a2_load}", flush=True)
    camera.start()
    records = []
    history = deque(maxlen=128)  # (seq, host capture time, token(40,C) on device)
    last_seq = 0
    resized_any = False
    nan_frames = 0
    first_processed_at = None
    frames_path = output_dir / "frames.jsonl"
    video_path = output_dir / "a2_overlay.mp4"
    video_writer = None
    skipped_total = 0
    contract_validated = False
    stop_requested = threading.Event()

    def request_stop(signum, frame):
        stop_requested.set()

    signal.signal(signal.SIGTERM, request_stop)
    try:
        if args.first_frame_timeout > 5.0:
            print(f"[wait] first frame, up to {args.first_frame_timeout:.0f}s", flush=True)
        first = camera.get_after(last_seq, timeout=args.first_frame_timeout)
        if first is None:
            raise RuntimeError(
                camera.error
                or f"no camera frame received within {args.first_frame_timeout:.0f} seconds")
        last_seq, _, _, first_frame = first
        print(f"[warmup] iterations={args.warmup_iters}", flush=True)
        warmup(first_frame, args, device, backbone, a2)
        last_seq = camera.reset_measurement()
        first_processed_at = time.perf_counter()
        amp_enabled = args.amp and device.type == "cuda"
        # ego_motion is constant for a run, so build it once instead of
        # re-uploading a (1,2) tensor on every frame.
        ego_input = ego_motion_tensor(args, device)
        camera_params = None
        exposure_us = None
        exposure_msg = ""
        if args.display and args.tune_exposure:
            camera_params = DynamicCameraParams(args.driver_node)
            print(f"[params] {camera_params.summary}", flush=True)
            if camera_params.supports("ExposureTime"):
                exposure_us = camera_params.get("ExposureTime")
                exposure_msg = "[ / ] exposure"
            else:
                exposure_msg = "exposure not tunable; set --exposure at launch"

        with frames_path.open("w", encoding="utf-8") as frame_log:
            while not stop_requested.is_set() and time.perf_counter() - first_processed_at < duration:
                item = camera.get_after(last_seq, timeout=2.0)
                if item is None:
                    if camera.error:
                        raise RuntimeError(camera.error)
                    continue
                seq, source_stamp, host_received_at, frame = item
                skipped = max(0, seq - last_seq - 1)
                last_seq = seq
                started = time.perf_counter()

                image, resized, model_frame = preprocess(frame, args, device)
                resized_any = resized_any or resized
                sync_device(device)
                preprocessed = time.perf_counter()

                with torch.inference_mode(), autocast_context(device, amp_enabled):
                    feature = backbone(image)[-1]
                    token = F.adaptive_avg_pool2d(
                        feature, (1, N_BINS)).squeeze(2).transpose(1, 2)
                sync_device(device)
                backbone_done = time.perf_counter()

                history.append((seq, source_stamp, token[0].detach()))
                past, age_errors_ms, source_seqs = select_history(history, source_stamp)
                token_list = [token] + past
                with torch.inference_mode(), autocast_context(device, amp_enabled):
                    logits = a2.forward_tokens(token_list, ego_input)
                sync_device(device)
                a2_done = time.perf_counter()
                a2_done_wall = time.time()

                finite = bool(torch.isfinite(logits).all())
                if not finite:
                    nan_frames += 1
                if not contract_validated:
                    validate_runtime_contract(image, feature, token, logits, args)
                    contract_validated = True
                    print(
                        f"[contract] image={tuple(image.shape)} feature={tuple(feature.shape)} "
                        f"i1={tuple(token.shape)} a2={tuple(logits.shape)}",
                        flush=True,
                    )
                decoded = decode_a2(logits) if finite else empty_decoded_a2()
                perception = a2_summary(decoded, args.obj_thr) if finite else {
                    "lead": False,
                    "error": "a2 output contains NaN or Inf",
                }
                completed = time.perf_counter()
                history_span = source_stamp - history[0][1]
                header_available = bool(camera.negotiated.get("header_timestamp_available", False))
                skipped_total += skipped
                record = {
                    "seq": seq,
                    "source_timestamp": source_stamp,
                    "host_received_at": host_received_at,
                    "completed_at": completed,
                    "capture_to_a2_ms_host": (a2_done - host_received_at) * 1000.0,
                    "capture_to_a2_ms_header": (
                        (a2_done_wall - source_stamp) * 1000.0 if header_available else None),
                    "capture_to_output_ms_host": (completed - host_received_at) * 1000.0,
                    "preprocess_ms": (preprocessed - started) * 1000.0,
                    "backbone_ms": (backbone_done - preprocessed) * 1000.0,
                    "a2_ms": (a2_done - backbone_done) * 1000.0,
                    "compute_ms": (a2_done - started) * 1000.0,
                    "postprocess_ms": (completed - a2_done) * 1000.0,
                    "visualization_ms": None,
                    "skipped_since_previous": skipped,
                    "history_span_s": history_span,
                    "history_age_error_ms": age_errors_ms,
                    "history_unique_sources": len(set(source_seqs)),
                    "history_source_seq": source_seqs,
                    "finite": finite,
                    "resized": resized,
                    "a2": perception,
                }
                record["skipped_total"] = skipped_total
                if args.log_full_a2 and finite:
                    record["a2_bins"] = {
                        "raw_40x7": decoded["raw"].tolist(),
                        "object_probability": decoded["object_probability"].tolist(),
                        "distance_m": decoded["distance_m"].tolist(),
                        "v_long_abs_mps": decoded["v_long_abs_mps"].tolist(),
                        "v_lat_abs_mps": decoded["v_lat_abs_mps"].tolist(),
                        "v_long_rel_mps": decoded["v_long_rel_mps"].tolist(),
                        "v_lat_rel_mps": decoded["v_lat_rel_mps"].tolist(),
                        "proximity": decoded["proximity"].tolist(),
                    }

                quit_requested = False
                # A slower diagnostic display can preserve model throughput on
                # the RT boot. Video still renders every frame when enabled.
                render_this_frame = args.save_overlay_video or (
                    args.display and len(records) % args.display_every == 0)
                if render_this_frame:
                    visual_started = time.perf_counter()
                    recent_completion = [r["completed_at"] for r in records[-20:]] + [completed]
                    if len(recent_completion) >= 2:
                        recent_elapsed = recent_completion[-1] - recent_completion[0]
                        fps_recent = ((len(recent_completion) - 1) / recent_elapsed
                                      if recent_elapsed > 0.0 else 0.0)
                    else:
                        fps_recent = 0.0
                    status_text = None
                    if args.display:
                        clipped, crushed = band_clipping(
                            model_frame, args.roi_y0, args.roi_y1)
                        exposure_part = (f"exposure {exposure_us:.0f} us"
                                         if exposure_us is not None else "exposure fixed")
                        warning = ""
                        if clipped > 5.0:
                            warning = "  << OVEREXPOSED"
                        elif crushed > 20.0:
                            warning = "  << UNDEREXPOSED"
                        status_text = (
                            f"{exposure_part} | band clipped {clipped:4.1f}% "
                            f"crushed {crushed:4.1f}% | {exposure_msg}{warning}")
                    overlay = render_a2_overlay(
                        model_frame, decoded, record, args, fps_recent, status_text)
                    if args.save_overlay_video:
                        if video_writer is None:
                            video_fps = args.video_fps if args.video_fps > 0.0 else args.camera_fps
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            video_writer = cv2.VideoWriter(
                                str(video_path), fourcc, video_fps,
                                (overlay.shape[1], overlay.shape[0]))
                            if not video_writer.isOpened():
                                raise RuntimeError(f"cannot create overlay video: {video_path}")
                            print(f"[video] {video_path} @ {video_fps:.3f} FPS", flush=True)
                        video_writer.write(overlay)
                    if args.display:
                        cv2.imshow(DISPLAY_WINDOW, overlay)
                        key = cv2.waitKey(1) & 0xFF
                        quit_requested = key in (27, ord("q"))
                        if key in (ord("["), ord("]")) and exposure_us is not None:
                            exposure_us = step_exposure(
                                exposure_us, 1 if key == ord("]") else -1)
                            camera_params.set_async("ExposureTime", exposure_us)
                    record["visualization_ms"] = (time.perf_counter() - visual_started) * 1000.0

                records.append(record)
                frame_log.write(json.dumps(record, ensure_ascii=False) + "\n")
                if len(records) % max(1, int(round(args.target_fps))) == 0:
                    frame_log.flush()
                if quit_requested:
                    break
    except KeyboardInterrupt:
        print("[stop] interrupted", flush=True)
    finally:
        if video_writer is not None:
            video_writer.release()
        camera.stop()
        if args.display:
            cv2.destroyAllWindows()

    if not records:
        raise RuntimeError("no frame was processed")
    model_info = {
        "device": device,
        "seg_load": seg_load,
        "a2_load": a2_load,
        "senior_a2_source_sha256": sha256_file(SENIOR_A2_FILE),
        "seg_checkpoint_sha256": sha256_file(Path(args.seg_ckpt)),
        "a2_checkpoint_sha256": sha256_file(Path(args.a2_ckpt)),
    }
    summary = summarize(records, camera, args, model_info, resized_any, nan_frames)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[saved] {summary_path}", flush=True)
    if not summary["pass"]["sustained_a2_21fps"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""Verify the installed Nuvo CUDA runtime and the bundled A2 model contract."""

import importlib.util
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "live_camera_a2_21fps.py"
SEG_CKPT = ROOT / "senior_code" / "data" / "ckpts" / "seg_pretrain_fx640_v3_backbone.pt"
A2_CKPT = ROOT / "senior_code" / "data" / "ckpts" / "a2_mf20_v2s_n20.pt"


def load_runner():
    spec = importlib.util.spec_from_file_location("live_camera_a2_21fps", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    device = torch.device("cuda")
    runner = load_runner()
    print(
        f"[cuda] torch={torch.__version__} runtime={torch.version.cuda} "
        f"device={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    backbone, seg_info = runner.load_backbone(
        "tf_efficientnetv2_s", str(SEG_CKPT), device, allow_partial=False)
    a2, a2_info = runner.load_a2(
        "tf_efficientnetv2_s", str(A2_CKPT), device, allow_partial=False)
    args = SimpleNamespace(width=1280, height=720, roi_y0=235, roi_y1=485, allow_resize=False)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    image, _, _ = runner.preprocess(frame, args, device)

    started = time.perf_counter()
    with torch.inference_mode():
        feature = backbone(image)[-1]
        token = F.adaptive_avg_pool2d(feature, (1, runner.N_BINS)).squeeze(2).transpose(1, 2)
        logits = a2.forward_tokens([token] * 21, torch.zeros(1, 2, device=device))
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    runner.validate_runtime_contract(image, feature, token, logits, args)
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("a2 output contains NaN or Inf")
    print(f"[checkpoint] seg={seg_info}", flush=True)
    print(f"[checkpoint] a2={a2_info}", flush=True)
    print(
        f"[contract] image={tuple(image.shape)} feature={tuple(feature.shape)} "
        f"i1={tuple(token.shape)} a2={tuple(logits.shape)}",
        flush=True,
    )
    print(f"[OK] one cold synthetic forward completed in {elapsed_ms:.2f} ms", flush=True)

    def forward_once():
        current_image, _, _ = runner.preprocess(frame, args, device)
        with torch.inference_mode():
            current_feature = backbone(current_image)[-1]
            current_token = F.adaptive_avg_pool2d(
                current_feature, (1, runner.N_BINS)).squeeze(2).transpose(1, 2)
            a2.forward_tokens(
                [current_token] * 21, torch.zeros(1, 2, device=device))
        torch.cuda.synchronize()

    for _ in range(10):
        forward_once()
    samples_ms = []
    for _ in range(30):
        sample_started = time.perf_counter()
        forward_once()
        samples_ms.append((time.perf_counter() - sample_started) * 1000.0)
    p95 = float(np.percentile(np.asarray(samples_ms), 95))
    print(
        f"[warm] n=30 p50={statistics.median(samples_ms):.2f} ms "
        f"p95={p95:.2f} ms budget={1000.0 / 21.0:.2f} ms",
        flush=True,
    )


if __name__ == "__main__":
    main()

"""환경 구축 후 sanity check 스크립트.

사용
----
    python scripts/sanity_check_env.py

torch / torchvision / 모델 팩토리 / 더미 forward 가 모두 동작하는지만 확인한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.models import get_model  # noqa: E402


def _make_dummy_args() -> argparse.Namespace:
    return argparse.Namespace(
        model_name="ImageEgoSCCPredictor",
        target="accel_brake",
        output_shape="control",
        include_speed_profile=True,
        input_time_to_traj="rel_time",
        lstm_input_ego_size=6,
        hidden_dim=128,
        img_feature_dim=128,
        img_pretrained=False,  # 네트워크 다운로드 피하기 위해 False
        tfr_seq_length=40,
        pred_first_only=False,
    )


def main() -> int:
    args = _make_dummy_args()

    print(f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    model = get_model(args)
    model.eval()

    B = 2
    image = torch.randn(B, 3, 224, 224)
    ego = torch.randn(B, args.lstm_input_ego_size)
    with torch.no_grad():
        out = model(image, ego)
    print(f"[ok] {args.model_name} output shape: {tuple(out.shape)}  (expected ({B}, 40, 1))")

    args.model_name = "ImageTrajEgoSCCPredictor"
    model = get_model(args)
    model.eval()
    traj_dim = 3 + (1 if args.input_time_to_traj != "none" else 0)
    traj = torch.randn(B, args.tfr_seq_length, traj_dim)
    with torch.no_grad():
        out = model(traj, ego, image=image)
    print(f"[ok] {args.model_name} output shape: {tuple(out.shape)}  (expected ({B}, 40, 1))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

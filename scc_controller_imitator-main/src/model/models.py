"""SCC Controller Imitator 모델 팩토리.

controller_imitator/src/model/models.py 와 동일한 계약을 따른다.
새 모델은 반드시 ``args.model_name`` 기준 분기 + ``forward(traj_or_img, ego)`` 시그니처를 만족해야 한다.
"""

from .image_models.image_scc_models import (
    ImageEgoSCCPredictor,
    ImageTrajEgoSCCPredictor,
)


def get_model(args):
    """모델 팩토리.

    지원 모델
        - ImageEgoSCCPredictor:       (image, ego) -> accel/brake
        - ImageTrajEgoSCCPredictor:   (image + planned traj, ego) -> accel/brake

    Args:
        args: 학습 스크립트에서 파싱된 namespace. 최소 ``model_name`` 을 가져야 한다.
    """
    if args.model_name == "ImageEgoSCCPredictor":
        return ImageEgoSCCPredictor(args)
    if args.model_name == "ImageTrajEgoSCCPredictor":
        return ImageTrajEgoSCCPredictor(args)

    raise ValueError(f"Model {args.model_name} not available.")

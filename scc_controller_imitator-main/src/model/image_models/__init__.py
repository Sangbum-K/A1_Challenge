# image_scc_models.py 가 v2 로 rename 됨 — 호환성 위해 v2 의 클래스 (V2 suffix 도 포함) 를 re-export
from .image_scc_models_v2 import (
    SCCControllerImitatorV2,
    LateralStripEncoder,
    ImageFusionBlock,
)
# 외부에서 SCCControllerImitator 로도 import 가능하도록 alias
SCCControllerImitator = SCCControllerImitatorV2

__all__ = ["SCCControllerImitator", "SCCControllerImitatorV2", "LateralStripEncoder", "ImageFusionBlock"]

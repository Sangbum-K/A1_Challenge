# 신규 모델은 src.model.image_models.iy_model.IyModel 을 직접 import.
# 구버전 SCCControllerImitator (image_scc_models.py) 는 legacy/ 로 이동.
# SCCControllerImitatorV2 (image_scc_models_v2.py) 는 closed_loop_node 호환을 위해
# 일시적으로 src/ 에 유지 (IyModel 전환 예정).
__all__ = []

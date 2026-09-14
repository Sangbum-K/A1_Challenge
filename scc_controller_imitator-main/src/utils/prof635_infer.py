"""#635-format deployment 추론 코어 (ROS 무관, 테스트 가능).

per-tick:
  raceline + ego pose → traj6 (raceline_search, mode='speed' — #654/v5 는 채널2=speed 로 학습)
  ego_state → ego6 [vx,vy,ax,ay,tire_angle,yaw_rate] (tire = EGO steering 있으면 그것, 아니면 kinematic)
  camera → a_2 v7b → i2_mem (closed_loop 기존 경로; 여기선 외부 주입. seg=fx640_v3 + ego_mot=0 으로 계산할 것)
  Prof635SideModel(traj6, ego6, i2_mem) → steer(deg), pedal(-1~1)
  pedal → (accel_pedal, brake_pedal)
"""
from __future__ import annotations
import numpy as np
import torch
from .raceline_search import sample_traj, resolve_cutoff

WB = 2.35
STEER_NORM = 40.0


def build_ego6(vx, vy, ax, ay, yaw_rate, tire_angle=None):
    """tire_angle 주어지면(EGO steering) 사용, 아니면 kinematic. |vx|<5 → 0."""
    if tire_angle is None:
        tire_angle = float(np.arctan(WB * yaw_rate / vx)) if abs(vx) >= 5.0 else 0.0
    return np.array([vx, vy, ax, ay, tire_angle, yaw_rate], dtype=np.float32)


def pedal_to_pedals(pedal: float):
    """pedal(-1~1) → (accel_pedal, brake_pedal) ∈ [0,1]. 양수=accel, 음수=brake."""
    p = float(pedal)
    return (max(p, 0.0), max(-p, 0.0))


@torch.no_grad()
def infer_step(model, raceline, ex, ey, e_yaw, ego6, i2_mem,
               cutoff, polygon=None, device="cpu"):
    """한 tick 추론. 반환 dict(steer_deg, pedal, accel_pedal, brake_pedal, traj6).
    i2_mem: (40,128) np or tensor (a_2 v7b 출력). polygon: (6,2) or None.
    """
    traj6 = sample_traj(raceline, ex, ey, e_yaw, cutoff, polygon, mode="speed")  # (6,3) ch2=speed (#654/v5)
    b = {
        "traj6":  torch.from_numpy(traj6).unsqueeze(0).to(device),
        "ego6":   torch.from_numpy(np.asarray(ego6, np.float32)).unsqueeze(0).to(device),
        "i2_mem": (i2_mem if torch.is_tensor(i2_mem) else torch.from_numpy(np.asarray(i2_mem, np.float32))
                   ).unsqueeze(0).to(device),
    }
    steer_n, pedal = model(b)
    steer_deg = float(steer_n.item()) * STEER_NORM
    pedal = float(pedal.item())
    acc, brk = pedal_to_pedals(pedal)
    return {"steer_deg": steer_deg, "pedal": pedal, "accel_pedal": acc,
            "brake_pedal": brk, "traj6": traj6}

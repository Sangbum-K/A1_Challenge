"""MPC.bag 에서 time-aligned (vehicle_state, trajectory) 튜플 K 개 추출.

ROS msg → numpy 변환 (npize_* 함수 차용) → npz 저장.
저장된 데이터로 우리 MPC sanity test 가능.

각 sample npz 키:
  vs (1, 4) [x, y, yaw, speed]
  ms (1, 8) [x, y, yaw, vx, vy, ax, ay, tire_angle]
  traj_point (M, 4) [x, y, yaw, speed]
  traj_refmap (M, 5) [x, y, ?, ?, ?]
  vs_stamp / traj_stamp / objects (zeros) / ...
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, "/home/ailab-12/git/A1_Rule/devel/lib/python3/dist-packages")
sys.path.insert(0, "/home/ailab-12/git/A1_Rule/src/app/control/acados_scc_control/scripts")

import rosbag
import numpy as np

BAG_PATH = "/home/ailab-12/git/A1_Rule/MPC.bag"
OUT_NPZ = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/mpc_bag_baseline.npz"

VEHICLE_TOPIC = "/app/loc/vehicle_state"
TRAJECTORY_TOPIC = "/app/pla/trajectory"
CONTROL_IO_TOPIC = "/app/con/ml_control_io"  # 정답 reference (MPC 출력 확인용)

K_SAMPLES = 5
TIME_TOLERANCE_NS = int(0.1 * 1e9)  # 100ms


def vs_to_numpy(vs_msg):
    """VehicleState → (vs_4, ms_8). npize_vehicle_state 와 동일."""
    x0 = np.array([[vs_msg.x, vs_msg.y, vs_msg.yaw, vs_msg.vx]], dtype=np.float64)
    ms = np.zeros((1, 8), dtype=np.float64)
    ms[0, 0] = vs_msg.x
    ms[0, 1] = vs_msg.y
    ms[0, 2] = vs_msg.yaw
    ms[0, 3] = vs_msg.vx
    ms[0, 4] = vs_msg.vy
    ms[0, 5] = vs_msg.ax
    ms[0, 6] = vs_msg.ay
    ms[0, 7] = vs_msg.vehicle_can.steering_tire_angle
    return x0, ms


def traj_to_numpy(traj_msg):
    """Trajectory → point (N, 4) [x, y, yaw, speed] in global frame.

    NOTE: ROS Trajectory msg 의 reference_map 필드는 npize_trajectory_local 에서
    trajectory.point 의 (x, y) 를 그대로 재사용한다 (utils/trajectory_process.py 참고).
    그래서 별도 추출 불필요 — sanity test 측에서 동일 점들을 reference_map 으로 둔다.
    """
    n_pt = len(traj_msg.point)
    pt = np.zeros((n_pt, 4), dtype=np.float64)
    for i, p in enumerate(traj_msg.point):
        pt[i, 0] = p.x
        pt[i, 1] = p.y
        pt[i, 2] = p.yaw
        pt[i, 3] = p.speed
    return pt


def control_io_to_numpy(c_msg):
    """ControlInputOutput → (steer_seq, accel_seq). 정답 비교용."""
    steer = np.asarray(c_msg.command_steer, dtype=np.float64)
    accel = np.asarray(c_msg.command_acceleration, dtype=np.float64)
    return steer, accel


def main():
    print(f"opening bag: {BAG_PATH}")
    bag = rosbag.Bag(BAG_PATH)

    # Index trajectory msgs by stamp
    print("indexing trajectory msgs ...")
    traj_idx = []
    for topic, msg, t in bag.read_messages(topics=[TRAJECTORY_TOPIC]):
        traj_idx.append((t.to_nsec(), msg))
    print(f"  {len(traj_idx)} trajectory msgs")

    # Index control_io msgs by stamp
    print("indexing control_io msgs ...")
    cio_idx = []
    for topic, msg, t in bag.read_messages(topics=[CONTROL_IO_TOPIC]):
        cio_idx.append((t.to_nsec(), msg))
    print(f"  {len(cio_idx)} control_io msgs")

    # For each vehicle_state, find nearest trajectory + control_io
    print("scanning vehicle_state msgs for aligned tuples ...")
    samples = []
    for topic, msg, t in bag.read_messages(topics=[VEHICLE_TOPIC]):
        vs_t = t.to_nsec()
        # only take samples with nonzero speed (avoid startup transients)
        if msg.vx < 1.0:
            continue
        # find nearest trajectory
        traj_dt, traj_msg = min(((abs(t_ns - vs_t), m) for t_ns, m in traj_idx), key=lambda x: x[0])
        if traj_dt > TIME_TOLERANCE_NS:
            continue
        # find nearest control_io (ground truth output of MPC)
        cio_dt, cio_msg = min(((abs(t_ns - vs_t), m) for t_ns, m in cio_idx), key=lambda x: x[0])
        if cio_dt > TIME_TOLERANCE_NS:
            continue
        samples.append((vs_t, msg, traj_msg, cio_msg))
        if len(samples) >= K_SAMPLES * 50:  # plenty
            break
    print(f"  candidate aligned tuples: {len(samples)}")

    # pick K spread evenly
    if len(samples) < K_SAMPLES:
        raise RuntimeError("not enough aligned samples")
    step = max(1, len(samples) // K_SAMPLES)
    picks = samples[::step][:K_SAMPLES]
    print(f"  picking {len(picks)} samples (step={step})")

    out = {}
    for i, (vs_t, vs_msg, traj_msg, cio_msg) in enumerate(picks):
        x0, ms = vs_to_numpy(vs_msg)
        pt = traj_to_numpy(traj_msg)
        steer_seq, accel_seq = control_io_to_numpy(cio_msg)
        out[f"s{i}__vs_global"] = x0          # global xy / yaw
        out[f"s{i}__ms_global"] = ms
        out[f"s{i}__traj_global"] = pt
        out[f"s{i}__gt_steer"] = steer_seq    # MPC 정답
        out[f"s{i}__gt_accel"] = accel_seq
        out[f"s{i}__stamp_ns"] = np.array(vs_t, dtype=np.int64)
        print(f"  [{i}] stamp={vs_t}  vs(x,y,yaw,v)={x0.tolist()[0]}  "
              f"traj_pts={pt.shape}  gt_steer.shape={steer_seq.shape}")

    bag.close()
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez(OUT_NPZ, **out)
    print(f"\nwrote {OUT_NPZ} ({os.path.getsize(OUT_NPZ)/1024:.1f} KB)")


if __name__ == "__main__":
    main()

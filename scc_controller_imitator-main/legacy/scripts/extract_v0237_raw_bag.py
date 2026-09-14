"""v0237 source raw bag (mpc_trajectory_generator_no_random_rotation.bag) 에서 직접 추출.

각 ControlInputOutput msg 가 vehicle_state + trajectory + GT control 시퀀스를 모두 보유.
이건 v0237 pkl 의 진짜 source. baseline 처럼 처리해서 MPC 정상 동작 확인.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, "/home/ailab-12/git/A1_Rule/devel/lib/python3/dist-packages")

import rosbag
import numpy as np

BAG = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/controller_imitator/data/Raw_bag_multi/v0237_latlon_separate_with_without_random_rotation_combined/mpc_trajectory_generator_no_random_rotation.bag"
OUT = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/v0237_raw_bag_sample10.npz"
K = 10


def vs_to_np(vs):
    x0 = np.array([[vs.x, vs.y, vs.yaw, vs.vx]], dtype=np.float64)
    ms = np.zeros((1, 8), dtype=np.float64)
    ms[0, 0] = vs.x; ms[0, 1] = vs.y; ms[0, 2] = vs.yaw
    ms[0, 3] = vs.vx; ms[0, 4] = vs.vy
    ms[0, 5] = vs.ax; ms[0, 6] = vs.ay
    ms[0, 7] = vs.vehicle_can.steering_tire_angle
    return x0, ms


def traj_to_np(traj):
    n = len(traj.point)
    arr = np.zeros((n, 4), dtype=np.float64)
    for i, p in enumerate(traj.point):
        arr[i, 0] = p.x; arr[i, 1] = p.y; arr[i, 2] = p.yaw; arr[i, 3] = p.speed
    return arr


def main():
    print(f"opening {BAG} (~1.5 GB, ~3 min open time) ...")
    import time
    t0 = time.time()
    bag = rosbag.Bag(BAG)
    info = bag.get_type_and_topic_info()
    total = info.topics["/app/con/control_io"].message_count
    print(f"  open took {time.time()-t0:.0f}s, total msgs = {total}")

    step = max(1, total // 200)
    print(f"  sampling every {step}th msg, target K={K} ...")
    out = {}
    idx = 0
    saved = 0
    for topic, msg, t in bag.read_messages(topics=["/app/con/control_io"]):
        idx += 1
        if idx % step != 0:
            continue
        if msg.vehicle_state.vx < 1.0:
            continue
        vs_np, ms_np = vs_to_np(msg.vehicle_state)
        traj_np = traj_to_np(msg.trajectory)
        solved_np = traj_to_np(msg.solved_trajectory)
        if len(traj_np) < 5:
            continue
        cmd_steer = np.asarray(msg.command_steer, dtype=np.float64)
        cmd_accel_pedal = np.asarray(msg.command_accel_pedal, dtype=np.float64)
        cmd_brake_pedal = np.asarray(msg.command_brake_pedal, dtype=np.float64)
        cmd_speed = np.asarray(msg.command_speed, dtype=np.float64)
        cmd_accel = np.asarray(msg.command_acceleration, dtype=np.float64)
        i = saved
        out[f"s{i}__vs_global"] = vs_np
        out[f"s{i}__ms_global"] = ms_np
        out[f"s{i}__traj_global"] = traj_np
        out[f"s{i}__solved_global"] = solved_np
        out[f"s{i}__gt_steer"] = cmd_steer
        out[f"s{i}__gt_accel_pedal"] = cmd_accel_pedal
        out[f"s{i}__gt_brake_pedal"] = cmd_brake_pedal
        out[f"s{i}__gt_speed"] = cmd_speed
        out[f"s{i}__gt_acceleration"] = cmd_accel
        print(f"  [{i}] vx={vs_np[0,3]:.2f}, traj pts={len(traj_np)}, "
              f"traj_first_xy=({traj_np[0,0]:.1f},{traj_np[0,1]:.1f}) "
              f"traj_last_xy=({traj_np[-1,0]:.1f},{traj_np[-1,1]:.1f}) "
              f"cmd_steer len={len(cmd_steer)} cmd_speed[0]={cmd_speed[0]:.2f}")
        saved += 1
        if saved >= K:
            break
    bag.close()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, **out)
    print(f"\nwrote {OUT} ({os.path.getsize(OUT)/1024:.1f} KB), {saved} samples")


if __name__ == "__main__":
    main()

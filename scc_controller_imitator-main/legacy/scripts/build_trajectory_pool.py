"""Phase 2 prerequisite: v0237 raw bag 에서 trajectory pool 빌드.

각 ControlInputOutput msg 의 vehicle_state + trajectory 추출:
  - vehicle_state.vx (= ego initial speed at that frame)
  - trajectory.point[i].(x, y, yaw, speed)  → (40, 4) array (ego-local)

전체 345K msg 중 일부 sampling (예: 매 10번째 = 34.5K trajectories).

Output: v0237_traj_pool.npz
  trajs (P, 40, 4) [x, y, yaw, speed]
  v0s   (P,)        initial speed
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, "/home/ailab-12/git/A1_Rule/devel/lib/python3/dist-packages")
import rosbag
import numpy as np

BAGS = [
    "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/controller_imitator/data/Raw_bag_multi/v0237_latlon_separate_with_without_random_rotation_combined/mpc_trajectory_generator_no_random_rotation.bag",
    "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/controller_imitator/data/Raw_bag_multi/v0237_latlon_separate_with_without_random_rotation_combined/mpc_trajectory_generator_random_rotation.bag",
]
OUT_NPZ = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/v0237_traj_pool.npz"
SAMPLE_EVERY = 10   # 매 N 번째 msg


def main():
    trajs, v0s = [], []
    for bag_path in BAGS:
        print(f"opening {os.path.basename(bag_path)}")
        t0 = time.time()
        bag = rosbag.Bag(bag_path)
        total = bag.get_type_and_topic_info().topics["/app/con/control_io"].message_count
        print(f"  total msgs: {total}")
        idx = 0
        kept = 0
        for topic, msg, t in bag.read_messages(topics=["/app/con/control_io"]):
            idx += 1
            if idx % SAMPLE_EVERY != 0: continue
            if len(msg.trajectory.point) != 40: continue
            v0 = float(msg.vehicle_state.vx)
            if v0 < 0.5: continue
            pts = np.zeros((40, 4), dtype=np.float32)
            for i, p in enumerate(msg.trajectory.point):
                pts[i, 0] = p.x; pts[i, 1] = p.y; pts[i, 2] = p.yaw; pts[i, 3] = p.speed
            trajs.append(pts)
            v0s.append(v0)
            kept += 1
            if kept % 5000 == 0:
                print(f"  kept {kept} ({kept/(time.time()-t0):.0f}/s)")
        bag.close()
        print(f"  {kept} trajectories from this bag\n")

    trajs = np.stack(trajs, axis=0)
    v0s = np.asarray(v0s, dtype=np.float32)
    print(f"\ntotal pool: {len(trajs)} trajectories")
    print(f"v0 range: [{v0s.min():.2f}, {v0s.max():.2f}], median={np.median(v0s):.2f}")

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez_compressed(OUT_NPZ, trajs=trajs, v0s=v0s)
    print(f"\nwrote {OUT_NPZ} ({os.path.getsize(OUT_NPZ)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

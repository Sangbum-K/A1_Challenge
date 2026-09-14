"""controller_imitator all_normal_data.pkl → all_normal_data.npz 변환.

pkl 의 pickle.load 가 매 process startup 마다 5-10분 소요. npz 는 mmap 가능 + 즉시 lazy access.

input pkl 형식: {"data": [dict, dict, ...]} 또는 [dict, ...]
각 dict: {ego_info: array, trajectory_points_local: array, target_steer: array, target_acceleration: array}

output npz:
  ego_5d       (N, 5)   float32
  traj_40      (N, 40, 3) float32
  target       (N, 40, 2) float32  [steer, ax]

사용:
  python scripts/convert_ci_aug_pkl_to_npz.py <ci_aug_dir>...
"""
import sys, time, pickle
from pathlib import Path
import numpy as np
import torch

def convert(pkl_path: Path):
    out = pkl_path.parent / "all_normal_data.npz"
    if out.exists():
        print(f"  [skip] {out.name} already exists")
        return
    t0 = time.time()
    print(f"  loading {pkl_path.name} ({pkl_path.stat().st_size/1e9:.2f} GB)...")
    p = pickle.load(open(pkl_path, "rb"))
    samples = p["data"] if isinstance(p, dict) and "data" in p else p
    N = len(samples)
    print(f"  parsed: {N:,} samples in {time.time()-t0:.1f}s")
    t0 = time.time()

    ego_5d = np.zeros((N, 5), dtype=np.float32)
    traj_40 = np.zeros((N, 40, 3), dtype=np.float32)
    target = np.zeros((N, 40, 2), dtype=np.float32)  # [steer, ax]

    def _as(x, expected_shape):
        if isinstance(x, torch.Tensor): x = x.cpu().numpy()
        x = np.asarray(x, dtype=np.float32)
        # pad/truncate to expected_shape
        if x.shape != expected_shape:
            target_arr = np.zeros(expected_shape, dtype=np.float32)
            slices = tuple(slice(0, min(x.shape[i] if i < x.ndim else 1, expected_shape[i])) for i in range(len(expected_shape)))
            try:
                target_arr[slices] = x[tuple(s for s in slices[:x.ndim])]
            except Exception: pass
            return target_arr
        return x

    for i, d in enumerate(samples):
        ego = d["ego_info"]
        ego_arr = ego.cpu().numpy() if isinstance(ego, torch.Tensor) else np.asarray(ego, dtype=np.float32)
        ego_5d[i] = ego_arr[:5]

        traj = d["trajectory_points_local"]
        traj_arr = traj.cpu().numpy() if isinstance(traj, torch.Tensor) else np.asarray(traj, dtype=np.float32)
        traj_40[i, :min(40, len(traj_arr))] = traj_arr[:40, :3]

        st = np.asarray(d["target_steer"], dtype=np.float32)[:40]
        ax = np.asarray(d["target_acceleration"], dtype=np.float32)[:40]
        target[i, :len(st), 0] = st
        target[i, :len(ax), 1] = ax

        if (i+1) % 100000 == 0:
            print(f"    {i+1:,}/{N:,} processed ({(time.time()-t0):.1f}s)")

    print(f"  arrays built in {time.time()-t0:.1f}s. saving {out.name}...")
    t0 = time.time()
    np.savez(out, ego_5d=ego_5d, traj_40=traj_40, target=target)  # uncompressed for fast load
    print(f"  saved {out.name} ({out.stat().st_size/1e9:.2f} GB) in {time.time()-t0:.1f}s")


def main():
    dirs = [Path(d) for d in sys.argv[1:]]
    print(f"[convert] {len(dirs)} dirs")
    for d in dirs:
        print(f"\n=== {d.name} ===")
        pkl = d / "all_normal_data.pkl"
        if pkl.exists():
            convert(pkl)
        else:
            print(f"  [skip] no all_normal_data.pkl in {d}")

if __name__ == "__main__":
    main()

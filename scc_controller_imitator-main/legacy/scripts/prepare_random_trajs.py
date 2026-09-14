"""SCC v2 batch 용 random trajectory bank 빌더 (per-frame binary).

각 frame 에 대해:
  - logged 1 개 (frame_list 에 자동 포함, traj override 없음)
  - random K-1 개: v0237_traj_pool 에서 |v0 - ego_vx| < VX_TOL sampling
    → ego-local → world frame 변환

출력:
  data/scc_traj_bank/{run_id}__t{tick:07d}.trajs.bin  -- 1 file/frame
    format: int32 K, int32 N_wp, float32[K * N_wp * 4]  (x, y, yaw, speed)
  /tmp/scc_frame_list_kn.txt   — synth_batch 용 frame list (1 line/frame)
    format: run_id tick dt_ms K bank_file_or_-
"""
from __future__ import annotations
import os, sys, json, argparse, struct
from pathlib import Path
import numpy as np

POOL = "/home/ailab-12/git/scc_controller_imitator/data/v0237_traj_pool.npz"
META = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/multiagnet_compact_meta.json"
INDEX = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/data/processed/index.json"
RAW = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/data/raw/sangam_track/MultiAgnet"
SCC_DONE = Path("/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_v2")
BANK = Path("/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_traj_bank")

DEG2RAD = np.pi / 180.0


def parse_ego_world_pose(ego_file: Path) -> tuple[float, float, float, float] | None:
    if not ego_file.exists(): return None
    x = y = z = None; yaw_deg = None; vx = None
    for line in ego_file.read_text().splitlines():
        if ":" not in line: continue
        k, _, v = line.partition(":")
        k = k.strip()
        if k == "position":
            nums = [float(t) for t in v.split()]
            if len(nums) >= 2: x, y = nums[0], nums[1]
        elif k == "orientation":
            nums = [float(t) for t in v.split()]
            if len(nums) >= 3: yaw_deg = nums[2]
        elif k == "velocity":
            nums = [float(t) for t in v.split()]
            if nums: vx = nums[0]
    if None in (x, y, yaw_deg, vx): return None
    return (x, y, yaw_deg * DEG2RAD, vx)


def ego_to_world_batch(traj_local: np.ndarray, ex: float, ey: float, eyaw: float) -> np.ndarray:
    """(K, N, 4) ego-local → (K, N, 4) world. 마지막 dim: [x, y, yaw, speed]."""
    c, s = np.cos(eyaw), np.sin(eyaw)
    out = np.empty_like(traj_local)
    out[..., 0] = ex + c * traj_local[..., 0] - s * traj_local[..., 1]
    out[..., 1] = ey + s * traj_local[..., 0] + c * traj_local[..., 1]
    out[..., 2] = traj_local[..., 2] + eyaw
    out[..., 3] = traj_local[..., 3]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="frame 당 trajectory 수 (logged 1 + random K-1)")
    ap.add_argument("--vx-tol", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--out-list", default="/tmp/scc_frame_list_kn.txt")
    args = ap.parse_args()

    print("loading pool ...")
    pool = np.load(POOL)
    pool_trajs = pool["trajs"].astype(np.float32)   # (P, 40, 4)
    pool_v0s = pool["v0s"]
    P, N_wp, _ = pool_trajs.shape
    print(f"  pool: {P} trajs, N_wp={N_wp}")

    meta = json.load(open(META))
    idx = json.load(open(INDEX))
    run_dt = {e["run_id"]: float(e["dt_per_tick_ms"]) for e in idx}

    # logged SCC 존재하는 frame 만 random 추가 대상 (FrameLoader 가 fail 한 frame 은 random 도 fail)
    logged_keys = {p.stem.replace(".scc","").rsplit("__", 1)[0]
                    for p in SCC_DONE.glob("*__logged.scc.txt")}
    for p in SCC_DONE.glob("*.scc.txt"):
        if "__rand_" in p.stem or "__logged" in p.stem: continue
        logged_keys.add(p.stem.replace(".scc",""))
    print(f"  frames with logged SCC: {len(logged_keys)}")

    BANK.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    K_rand = args.k - 1
    f_list = open(args.out_list, "w")
    n_frames = 0; n_skip_no_logged = 0; n_skip_no_ego = 0
    for m in meta:
        run = m["run_id"]; tick = int(m["tick"])
        key = f"{run}__t{tick:07d}"
        if key not in logged_keys:
            n_skip_no_logged += 1; continue
        ego = parse_ego_world_pose(Path(RAW) / run / "EGO_INFO" / f"{tick}.txt")
        if ego is None:
            n_skip_no_ego += 1; continue
        ex, ey, eyaw, vx = ego

        # random K-1 sampling (vx-tol 내)
        mask = np.abs(pool_v0s - vx) < args.vx_tol
        valid_idx = np.where(mask)[0]
        if len(valid_idx) < K_rand:
            valid_idx = np.arange(P)
        picks = rng.choice(valid_idx, size=K_rand, replace=False)
        trajs_local = pool_trajs[picks]    # (K_rand, N_wp, 4)
        trajs_world = ego_to_world_batch(trajs_local, ex, ey, eyaw).astype(np.float32)

        bank_path = BANK / f"{key}.trajs.bin"
        with open(bank_path, "wb") as bf:
            bf.write(struct.pack("ii", K_rand, N_wp))
            bf.write(trajs_world.tobytes(order="C"))

        f_list.write(f"{run} {tick} {run_dt.get(run, 10.0)} {args.k} {bank_path}\n")
        n_frames += 1

        if args.limit_frames and n_frames >= args.limit_frames:
            break

    f_list.close()
    print(f"\nwrote {args.out_list}")
    print(f"  {n_frames} frames, K={args.k} → {n_frames * args.k} samples ({n_frames} logged + {n_frames * K_rand} random)")
    print(f"  skip: no_logged={n_skip_no_logged}, no_ego={n_skip_no_ego}")
    print(f"  bank dir: {BANK}  ({n_frames} files, ~{(8 + K_rand * N_wp * 16) // 1024} KB each)")


if __name__ == "__main__":
    main()

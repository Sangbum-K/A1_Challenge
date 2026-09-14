"""신규 raw run 들에 대해 random trajectory bank + synth_batch frame-list 생성.

기존 prepare_random_trajs.py 의 jiwon_e2e index/meta 의존성 제거 → raw EGO_INFO 직접 스캔.
사용:
  python scripts/prepare_new_run_bank.py \
      --raw-base /home/ailab-12/AILabSSD/02_Data/57_A1_E2E/00_raw \
      --track sangam_track \
      --policy 0519_agent1_pertX_pitX \
      --k 100 \
      --bank-out AILabDatasetLocal/scc_controller_imitator/data/scc_traj_bank_agent1 \
      --frame-list /tmp/scc_frame_list_agent1.txt
"""
from __future__ import annotations
import os, sys, argparse, struct, glob, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

POOL = "/home/ailab-12/git/scc_controller_imitator/data/v0237_traj_pool.npz"
DEG2RAD = np.pi / 180.0
CANON_DT_MS = 10.0


def parse_ego_world_pose(ego_file: Path):
    """raw EGO_INFO 1 파일 → (x, y, yaw_rad, vx)"""
    if not ego_file.exists(): return None
    x = y = yaw_deg = vx = None
    try:
        for line in open(ego_file, "rt"):
            k, _, v = line.strip().partition(":")
            if not v: continue
            k = k.strip(); v = v.strip()
            if k == "position":
                nums = [float(t) for t in v.split()]
                if len(nums) >= 2: x, y = nums[0], nums[1]
            elif k == "orientation":
                nums = [float(t) for t in v.split()]
                if len(nums) >= 3: yaw_deg = nums[2]
            elif k == "velocity":
                nums = [float(t) for t in v.split()]
                if nums: vx = nums[0]
    except Exception:
        return None
    if None in (x, y, yaw_deg, vx): return None
    return (x, y, yaw_deg * DEG2RAD, vx)


def ego_to_world_batch(traj_local: np.ndarray, ex: float, ey: float, eyaw: float) -> np.ndarray:
    c, s = np.cos(eyaw), np.sin(eyaw)
    out = np.empty_like(traj_local)
    out[..., 0] = ex + c * traj_local[..., 0] - s * traj_local[..., 1]
    out[..., 1] = ey + s * traj_local[..., 0] + c * traj_local[..., 1]
    out[..., 2] = traj_local[..., 2] + eyaw
    out[..., 3] = traj_local[..., 3]
    return out


def enumerate_frames(raw_base: Path, track: str, policy: str, file_step: int = 5):
    """track/policy 의 모든 run/tick 열거. (run_id, tick) 리스트 반환."""
    policy_dir = raw_base / track / policy
    runs = sorted([p.name for p in policy_dir.iterdir() if p.is_dir()])
    frames = []
    for run in runs:
        ego_dir = policy_dir / run / "EGO_INFO"
        if not ego_dir.exists(): continue
        ticks = []
        for p in ego_dir.iterdir():
            stem = p.stem  # "10000"
            try: ticks.append(int(stem))
            except ValueError: continue
        ticks.sort()
        # downsample to file_step (5)
        if ticks:
            tmin = ticks[0]
            ticks = [t for t in ticks if (t - tmin) % file_step == 0]
        for t in ticks:
            frames.append((run, t))
    return frames


def _process_frame(args_tuple):
    """worker: 1 frame 의 ego parse + bank write. return (run, tick, ok, bank_path or None)"""
    raw_base, track, policy, run, tick, k, vx_tol, seed_offset, bank_dir = args_tuple
    ego_path = Path(raw_base) / track / policy / run / "EGO_INFO" / f"{tick}.txt"
    ego = parse_ego_world_pose(ego_path)
    if ego is None: return (run, tick, False, None)
    ex, ey, eyaw, vx = ego
    K_rand = k - 1
    if K_rand <= 0:
        return (run, tick, True, None)
    # local sampling — pool 은 worker init 시 1회 load
    pool_trajs = _process_frame._pool_trajs
    pool_v0s   = _process_frame._pool_v0s
    rng = np.random.RandomState(seed_offset)
    P = pool_trajs.shape[0]
    N_wp = pool_trajs.shape[1]
    mask = np.abs(pool_v0s - vx) < vx_tol
    valid_idx = np.where(mask)[0]
    if len(valid_idx) < K_rand:
        valid_idx = np.arange(P)
    picks = rng.choice(valid_idx, size=K_rand, replace=False)
    trajs_local = pool_trajs[picks]
    trajs_world = ego_to_world_batch(trajs_local, ex, ey, eyaw).astype(np.float32)

    bank_path = Path(bank_dir) / f"{run}__t{tick:07d}.trajs.bin"
    try:
        with open(bank_path, "wb") as bf:
            bf.write(struct.pack("ii", K_rand, N_wp))
            bf.write(trajs_world.tobytes(order="C"))
    except Exception:
        return (run, tick, False, None)
    return (run, tick, True, str(bank_path))


def _worker_init(pool_path):
    z = np.load(pool_path)
    _process_frame._pool_trajs = z["trajs"].astype(np.float32)
    _process_frame._pool_v0s   = z["v0s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-base", required=True, type=Path)
    ap.add_argument("--track",    required=True)
    ap.add_argument("--policy",   required=True)
    ap.add_argument("--k", type=int, default=100, help="frame 당 trajectory 수 (logged 1 + random K-1)")
    ap.add_argument("--vx-tol", type=float, default=5.0)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--bank-out", type=Path, required=True,
                    help="random bank 출력 디렉토리 (per-frame .trajs.bin)")
    ap.add_argument("--frame-list", type=Path, required=True,
                    help="synth_batch frame-list 출력 경로 (.txt)")
    ap.add_argument("--dt-per-tick-ms", type=float, default=CANON_DT_MS,
                    help="canonical 10ms. 측정 가능하면 --measure-dt 추가")
    ap.add_argument("--file-step", type=int, default=5)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    print(f"[bank] pool loading...")
    pool = np.load(POOL)
    pool_trajs = pool["trajs"].astype(np.float32)   # (P, N_wp, 4)
    pool_v0s   = pool["v0s"]
    P, N_wp, _ = pool_trajs.shape
    print(f"  pool: {P} trajs, N_wp={N_wp}")

    print(f"[scan] raw {args.raw_base} / {args.track} / {args.policy}")
    frames = enumerate_frames(args.raw_base, args.track, args.policy, args.file_step)
    print(f"  {len(frames)} frames")
    if args.limit_frames: frames = frames[:args.limit_frames]

    args.bank_out.mkdir(parents=True, exist_ok=True)
    args.frame_list.parent.mkdir(parents=True, exist_ok=True)

    # parallel ego+bank write (workers, per-frame seed)
    work_args = [(str(args.raw_base), args.track, args.policy, run, tick,
                  args.k, args.vx_tol, args.seed + i, str(args.bank_out))
                 for i, (run, tick) in enumerate(frames)]

    f_list = open(args.frame_list, "w")
    n_ok = n_skip = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers,
                              initializer=_worker_init, initargs=(POOL,)) as ex:
        for i, (run, tick, ok, bank_path) in enumerate(ex.map(_process_frame, work_args, chunksize=64)):
            if not ok:
                n_skip += 1; continue
            if bank_path:
                f_list.write(f"{run} {tick} {args.dt_per_tick_ms:.3f} {args.k} {bank_path}\n")
            else:
                f_list.write(f"{run} {tick} {args.dt_per_tick_ms:.3f} 1 -\n")
            n_ok += 1
            if (i + 1) % 5000 == 0:
                dt = time.time() - t0
                rate = (i + 1) / dt
                eta = (len(frames) - (i + 1)) / max(rate, 1e-3)
                print(f"  [{i+1}/{len(frames)}] ok={n_ok} skip={n_skip}  rate={rate:.0f}/s ETA={eta:.0f}s")
    f_list.close()
    print(f"\n[done] frames={n_ok}  K={args.k}  total samples={n_ok*args.k}  in {time.time()-t0:.1f}s")
    print(f"  skip: {n_skip}")
    print(f"  frame-list: {args.frame_list}")
    print(f"  bank dir: {args.bank_out}  ({n_ok} files)")


if __name__ == "__main__":
    main()

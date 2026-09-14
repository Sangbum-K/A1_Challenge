"""신규 데이터 (speedway 등) 용 logged trajectory bank.

각 frame 의 future N ticks → world-frame (x, y, yaw_rad, speed) per-frame .bin.

사용:
  python scripts/prepare_logged_trajs_v2.py \
      --raw-base /home/ailab-12/AILabSSD/02_Data/57_A1_E2E/00_raw \
      --track speedway_track --policy 0519_agent1_pertX_pitX \
      --frame-list .../scc_frame_list_spd.txt \
      --out .../scc_logged_bank_spd
"""
from __future__ import annotations
import os, sys, argparse, struct, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

DEG2RAD = np.pi / 180.0
FILE_STEP = 5
CANON_DT_MS = 10.0


def parse_ego(p):
    if not Path(p).exists(): return None
    x = y = None; yaw_deg = None; vx = None
    try:
        for line in open(p, "rt"):
            if ":" not in line: continue
            k, _, v = line.partition(":")
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


def _process_frame(args_tuple):
    raw_base, track, policy, run, tick, n_wp, dt_s, dt_per_tick_ms, file_step, out_dir = args_tuple
    raw_step = dt_s * 1000.0 / dt_per_tick_ms
    step_ticks = int(round(raw_step / file_step)) * file_step
    if step_ticks <= 0: step_ticks = file_step
    pts = []
    ego_dir = Path(raw_base) / track / policy / run / "EGO_INFO"
    for i in range(n_wp):
        t = tick + step_ticks * i
        ego = parse_ego(ego_dir / f"{t}.txt")
        if ego is None: break
        x, y, yaw, vx = ego
        pts.append([x, y, yaw, vx])
    if len(pts) < 2: return (run, tick, False)
    pts_arr = np.array(pts, dtype=np.float32)
    out_path = Path(out_dir) / f"{run}__t{tick:07d}.bin"
    try:
        with open(out_path, "wb") as bf:
            bf.write(struct.pack("i", len(pts_arr)))
            bf.write(pts_arr.tobytes(order="C"))
    except Exception:
        return (run, tick, False)
    return (run, tick, True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-base", required=True, type=Path)
    ap.add_argument("--track",    required=True)
    ap.add_argument("--policy",   required=True)
    ap.add_argument("--frame-list", type=Path, required=True,
                    help="scc_frame_list_*.txt (run tick dt_ms K bank_file)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-wp", type=int, default=40)
    ap.add_argument("--dt",   type=float, default=0.1)
    ap.add_argument("--dt-per-tick-ms", type=float, default=CANON_DT_MS)
    ap.add_argument("--file-step", type=int, default=FILE_STEP)
    ap.add_argument("--workers",   type=int, default=16)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    frames = []
    with open(args.frame_list) as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) < 2 or toks[0].startswith("#"): continue
            frames.append((toks[0], int(toks[1])))
    print(f"[logged-bank] {len(frames)} frames  n_wp={args.n_wp}  dt={args.dt}s")

    work = [(str(args.raw_base), args.track, args.policy, run, tick,
             args.n_wp, args.dt, args.dt_per_tick_ms, args.file_step, str(args.out))
            for (run, tick) in frames]
    n_ok = n_fail = 0; t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (_, _, ok) in enumerate(ex.map(_process_frame, work, chunksize=64)):
            if ok: n_ok += 1
            else: n_fail += 1
            if (i + 1) % 5000 == 0:
                dt = time.time() - t0
                rate = (i + 1) / dt
                eta  = (len(frames) - (i + 1)) / max(rate, 1e-3)
                print(f"  [{i+1}/{len(frames)}] ok={n_ok} fail={n_fail}  rate={rate:.0f}/s ETA={eta:.0f}s")
    print(f"\n[done] ok={n_ok} fail={n_fail} in {time.time()-t0:.1f}s → {args.out}")


if __name__ == "__main__":
    main()

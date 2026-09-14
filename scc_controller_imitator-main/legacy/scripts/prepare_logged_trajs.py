"""logged trajectory pre-bake.

각 frame 의 logged input trajectory (raw EGO_INFO 의 향후 N tick) 를 binary 로 저장:
  data/scc_logged_bank/{run}__t{tick:07d}.bin
    format: int32 N_wp, float32[N_wp * 4]   (x, y, yaw_rad, speed)  world frame

random bank (scc_traj_bank/{key}.trajs.bin) 와 동일한 (N_wp, 4) 구조 (다만 K=1 = 단일 traj).

사용:
  python scripts/prepare_logged_trajs.py [--n-wp 40] [--dt 0.1]

frame 선정: scc_synth_v2/*__logged.scc.txt 가 존재하는 frame 만 (= SCC 합성 성공).
file_step = 5 tick (raw EGO_INFO 간격).
"""
from __future__ import annotations
import os, sys, json, argparse, struct
from pathlib import Path
import numpy as np

DEG2RAD = np.pi / 180.0
RAW = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/data/raw"
SCC_DIR = Path("/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_v2")
META = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/multiagnet_compact_meta.json"
INDEX = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/data/processed/index.json"
OUT = Path("/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_logged_bank")

FILE_STEP = 5    # raw EGO_INFO tick interval


def parse_ego(p: Path):
    if not p.exists(): return None
    x = y = z = None; yaw_deg = None; vx = None
    for line in p.read_text().splitlines():
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-wp", type=int, default=40)
    ap.add_argument("--dt", type=float, default=0.1)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(META))
    idx = json.load(open(INDEX))
    run_dt = {e["run_id"]: float(e["dt_per_tick_ms"]) for e in idx}

    # logged SCC 있는 frame 만 → file step 계산 후 future N ticks 저장
    logged_keys = set()
    for p in SCC_DIR.glob("*__logged.scc.txt"):
        logged_keys.add(p.stem.replace(".scc","").rsplit("__", 1)[0])
    # 옛 format (no __logged suffix) 도 포함
    for p in SCC_DIR.glob("*.scc.txt"):
        if "__rand_" in p.stem or "__logged" in p.stem: continue
        logged_keys.add(p.stem.replace(".scc",""))
    print(f"frames with logged SCC: {len(logged_keys)}")

    raw_p = Path(RAW); n_ok = 0; n_fail = 0
    for m in meta:
        run = m["run_id"]; tick = int(m["tick"])
        key = f"{run}__t{tick:07d}"
        if key not in logged_keys: continue
        # step_ticks = round(dt_ms / file_step / dt_per_tick_ms ...) — synth_batch 와 동일 식
        dt_per_tick_ms = run_dt.get(run, 10.0)
        raw_step = args.dt * 1000.0 / dt_per_tick_ms
        step_ticks = int(round(raw_step / FILE_STEP)) * FILE_STEP
        if step_ticks <= 0: step_ticks = FILE_STEP

        ego_dir = raw_p / m["track"] / m["policy"] / run / "EGO_INFO"
        wps = np.zeros((args.n_wp, 4), dtype=np.float32)
        ok = True; valid = 0
        for i in range(args.n_wp):
            t = tick + i * step_ticks
            ego = parse_ego(ego_dir / f"{t}.txt")
            if ego is None:
                # repeat last valid (synth_batch 와 동일 fallback)
                if valid == 0: ok = False; break
                wps[i] = wps[i-1]
            else:
                ex, ey, eyaw, vx = ego
                wps[i] = [ex, ey, eyaw, vx]
                valid += 1
        if not ok or valid < args.n_wp // 2:
            n_fail += 1; continue

        # save
        out_p = OUT / f"{key}.bin"
        with open(out_p, "wb") as bf:
            bf.write(struct.pack("i", args.n_wp))
            bf.write(wps.tobytes(order="C"))
        n_ok += 1
        if n_ok % 500 == 0:
            print(f"  {n_ok} ok, {n_fail} fail")
    print(f"\n[done] {n_ok} frames saved, {n_fail} failed")
    print(f"  out_dir: {OUT}")


if __name__ == "__main__":
    main()

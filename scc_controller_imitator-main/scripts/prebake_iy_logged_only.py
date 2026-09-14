"""prebake-logged-only — MPC 합성 없이 raw EGO log 의 future steer/ax 를 target 으로 사용.

용도: DriverB_new 처럼 합성 시간이 너무 큰 데이터를 빠르게 학습에 추가.
한 frame 당 1 sample (logged only, K=1, random bank 없음).

각 sample:
  sample_keys     "{run}__t{tick:07d}__logged"
  curr_keys       "{run}__t{tick:07d}"
  prev_keys       "{run}__t{tick-5:07d}"
  d_yaws          curr EGO yaw - prev EGO yaw (rad)
  sample_fx       640 (DriverB_new)
  ego_5d          [vx, vy, ax, ay, yaw_rate]  curr EGO
  traj_40 (40,3)  raceline sampling (ego frame, [x, y, speed])
  target (40,2)   future 40 EGO 의 [steer (deg), acceleration (m/s²)]
  mask (40,)      future 40 frame 의 EGO 가 존재하면 True
  obj_target (40,) aux_targets 가 있으면 lookup, 없으면 zeros
  prev_dtick      5

사용:
  python scripts/prebake_iy_logged_only.py \
      --frame-list .../scc_frame_list_DriverB_new.txt \
      --raw-base /home/ailab-12/AILabDataset/02_Custom_Dataset/57_A1_E2E/00_raw \
      --track sangam_track --policy DriverB_new \
      --raceline-csv data/control/raceline_sangam.csv \
      --aux-dir .../aux_targets_DriverB_new \
      --out .../prebake/f1_prebake_iy_DriverB_new.npz
"""
from __future__ import annotations
import os, sys, json, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

DEG2RAD = np.pi / 180.0
FILE_STEP = 5     # raw EGO file 간격 (5 tick)
DT = 0.1          # 100ms
N_WP = 40


from functools import lru_cache


@lru_cache(maxsize=20000)
def _parse_ego_str(p_str: str):
    """str-keyed cache — same file 여러 sample 에서 lookup 시 NFS read 1회."""
    from pathlib import Path as _P
    p = _P(p_str)
    if not p.exists(): return None
    out = {}
    try:
        for l in open(p):
            if ':' not in l: continue
            k, _, v = l.partition(':')
            k = k.strip(); v = v.strip()
            if k in ('position', 'orientation', 'velocity', 'acceleration', 'angularVelocity'):
                nums = [float(t) for t in v.split() if t not in ('Nan', 'nan')]
                if len(nums) >= 2: out[k] = nums
            elif k in ('steer', 'accel', 'brake'):
                try: out[k] = float(v)
                except: pass
    except: return None
    return out if 'position' in out and 'orientation' in out else None


def parse_ego(p):
    """raw EGO_INFO → dict (LRU-cached). 각 worker 별 local cache (ProcessPool fork)."""
    return _parse_ego_str(str(p))


def load_raceline(csv_path):
    """semicolon-delimited: s; x; y; yaw_rad; curv; speed; pit_flag → (N, 4) [x, y, yaw, speed]."""
    rows = []
    with open(csv_path) as f:
        for line in f:
            if not line.strip() or line.startswith('#'): continue
            t = line.strip().split(';')
            if len(t) < 6: continue
            try:
                rows.append([float(t[1]), float(t[2]), float(t[3]), float(t[5])])
            except: pass
    return np.array(rows, dtype=np.float32)


def sample_raceline_traj(raceline, ex, ey, e_yaw_rad, ev, n_wp=40, dt=0.1):
    """ego 위치에서 가장 가까운 raceline point 부터, vx-based 등시간격 sampling → (n_wp, 3) ego frame."""
    dx = raceline[:, 0] - ex; dy = raceline[:, 1] - ey
    dists = dx*dx + dy*dy
    near_idx = int(np.argmin(dists))
    # arc-length 기준 sampling
    n = len(raceline)
    # cumulative dist from near
    rolled = np.roll(raceline, -near_idx, axis=0)
    seg = np.linalg.norm(np.diff(rolled[:, :2], axis=0), axis=1)
    cs = np.concatenate([[0], np.cumsum(seg)])
    # 각 wp 의 target arc-length: ev * dt * i (단, ev 작으면 minimum 1 m)
    speed = max(ev, 1.0)
    targ = np.array([speed * dt * i for i in range(n_wp)])
    # interpolate
    out = np.zeros((n_wp, 3), dtype=np.float32)
    for i, t in enumerate(targ):
        if t > cs[-1]: t = cs[-1]
        idx = np.searchsorted(cs, t)
        idx = min(max(idx, 1), n - 1)
        u = (t - cs[idx-1]) / max(cs[idx] - cs[idx-1], 1e-6)
        xw = rolled[idx-1, 0] + u * (rolled[idx, 0] - rolled[idx-1, 0])
        yw = rolled[idx-1, 1] + u * (rolled[idx, 1] - rolled[idx-1, 1])
        sp = rolled[idx-1, 3] + u * (rolled[idx, 3] - rolled[idx-1, 3])
        # world → ego
        c, s = np.cos(-e_yaw_rad), np.sin(-e_yaw_rad)
        dxw, dyw = xw - ex, yw - ey
        out[i, 0] = c * dxw - s * dyw   # forward
        out[i, 1] = s * dxw + c * dyw   # left
        out[i, 2] = sp
    return out


def process_one(args_tuple):
    """worker — 1 frame 처리."""
    (run, tick, raw_base, track, policy, raceline,
     aux_dir, prev_dtick, n_wp, dt) = args_tuple
    try:
        run_dir = Path(raw_base) / track / policy / run
        ego_dir = run_dir / "EGO_INFO"
        ego_curr = parse_ego(ego_dir / f"{tick}.txt")
        if ego_curr is None: return None
        prev_tick = tick - prev_dtick
        ego_prev = parse_ego(ego_dir / f"{prev_tick}.txt")
        if ego_prev is None: return None

        # ego_5d
        v = ego_curr.get('velocity', [0,0,0])
        a = ego_curr.get('acceleration', [0,0,0])
        av = ego_curr.get('angularVelocity', [0,0,0])
        ego_5d = np.array([v[0], v[1] if len(v)>1 else 0,
                           a[0], a[1] if len(a)>1 else 0,
                           av[2] if len(av)>2 else 0], dtype=np.float32)
        # d_yaw
        d_yaw = (ego_curr['orientation'][2] - ego_prev['orientation'][2]) * DEG2RAD

        # traj_40
        ex, ey = ego_curr['position'][0], ego_curr['position'][1]
        e_yaw = ego_curr['orientation'][2] * DEG2RAD
        ev = v[0]
        traj = sample_raceline_traj(raceline, ex, ey, e_yaw, ev, n_wp, dt)

        # target — future 40 EGO 의 (steer, ax)
        step_ticks = int(round(dt * 1000)) / 1   # 100ms = 100 / file_step (5) tick 단위
        # raw file 은 5 tick step, dt=100ms = 100/10 (ms/tick) = 10 tick? 확인
        # 위 framework 에서 controller_imitator 의 dt_per_tick_ms 가 10ms. dt 100ms = 10 tick.
        # 그러나 raw file 은 5 tick step → 10 tick 의 future 는 file 2 단계 후.
        # 더 일반화: dt=0.1s 마다 file lookup. tick i = curr_tick + 100*i (1 tick = 1 ms 가정 — DriverB_new file 명 0.txt 부터 100000.txt 등)
        # 직접 file 이름 list 에서 next 찾기
        # 단순화: 100 tick = 100ms (10 ms/tick 가정)
        target = np.zeros((n_wp, 2), dtype=np.float32)
        mask = np.zeros(n_wp, dtype=np.bool_)
        for i in range(n_wp):
            t_fut = tick + int(round(dt * 1000 * (i+1)))   # 100ms × (i+1)
            # 가장 가까운 5 의 배수
            t_fut = (t_fut // 5) * 5
            ego_fut = parse_ego(ego_dir / f"{t_fut}.txt")
            if ego_fut is None: continue
            steer_rad = ego_fut.get('steer', 0.0)
            ax_fut = ego_fut.get('acceleration', [0,0,0])[0]
            target[i, 0] = steer_rad / DEG2RAD   # rad → deg
            target[i, 1] = ax_fut
            mask[i] = True

        # obj_target
        obj_t = np.zeros(n_wp, dtype=np.float32)
        if aux_dir is not None:
            fid = f"{track}__{policy}__{run}__t{tick:07d}"
            ap = Path(aux_dir) / f"{fid}.npz"
            if ap.exists():
                try:
                    aux_t = np.load(ap)["target"]
                    obj_t = (aux_t[:, 0] > 0.5).astype(np.float32)
                except: pass

        return dict(
            run=run, tick=tick,
            curr_key=f"{run}__t{tick:07d}",
            prev_key=f"{run}__t{prev_tick:07d}",
            d_yaw=float(d_yaw), ego_5d=ego_5d,
            traj=traj, target=target, mask=mask, obj_target=obj_t,
        )
    except Exception as e:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-list", required=True)
    ap.add_argument("--raw-base", required=True)
    ap.add_argument("--track", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--raceline-csv", required=True)
    ap.add_argument("--aux-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prev-dtick", type=int, default=5)
    ap.add_argument("--fx", type=float, default=640.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"[prebake-logged] raceline {args.raceline_csv}")
    raceline = load_raceline(args.raceline_csv)
    print(f"  raceline points: {len(raceline)}")

    frames = []
    with open(args.frame_list) as f:
        for l in f:
            t = l.strip().split()
            if len(t) >= 2: frames.append((t[0], int(t[1])))
    if args.limit: frames = frames[:args.limit]
    print(f"[prebake-logged] frames: {len(frames)}")

    tasks = [(run, tick, args.raw_base, args.track, args.policy, raceline,
              args.aux_dir, args.prev_dtick, N_WP, DT) for run, tick in frames]

    N = len(tasks)
    sample_keys = []; sample_tags = []
    curr_keys = []; prev_keys = []
    d_yaws = np.zeros(N, dtype=np.float32)
    sample_fx = np.full(N, args.fx, dtype=np.float32)
    ego_5d = np.zeros((N, 5), dtype=np.float32)
    traj = np.zeros((N, N_WP, 3), dtype=np.float32)
    target = np.zeros((N, N_WP, 2), dtype=np.float32)
    mask = np.zeros((N, N_WP), dtype=np.bool_)
    obj_target = np.zeros((N, N_WP), dtype=np.float32)
    n_valid = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_one, tasks, chunksize=64)):
            if r is None:
                sample_keys.append(f"INVALID_{i}"); sample_tags.append("invalid")
                curr_keys.append(""); prev_keys.append("")
                continue
            sample_keys.append(f"{r['run']}__t{r['tick']:07d}__logged")
            sample_tags.append("logged")
            curr_keys.append(r["curr_key"]); prev_keys.append(r["prev_key"])
            d_yaws[i] = r["d_yaw"]; ego_5d[i] = r["ego_5d"]
            traj[i] = r["traj"]; target[i] = r["target"]
            mask[i] = r["mask"]; obj_target[i] = r["obj_target"]
            n_valid += 1
            if (i+1) % 2000 == 0:
                rate = (i+1) / max(time.time()-t0, 1e-3)
                print(f"  [{i+1}/{N}] valid={n_valid}  rate={rate:.0f}/s  ETA={(N-i-1)/max(rate,1):.0f}s", flush=True)
    print(f"\n[done] valid={n_valid}/{N}  time={time.time()-t0:.0f}s")

    out_tmp = args.out + ".partial.npz"
    print(f"[prebake-logged] saving → {args.out}")
    np.savez_compressed(out_tmp,
                        sample_keys=np.array(sample_keys),
                        sample_tags=np.array(sample_tags),
                        curr_keys=np.array(curr_keys),
                        prev_keys=np.array(prev_keys),
                        d_yaws=d_yaws,
                        sample_fx=sample_fx,
                        ego_5d=ego_5d,
                        traj=traj,
                        target=target,
                        mask=mask,
                        obj_target=obj_target,
                        prev_dtick=args.prev_dtick)
    os.replace(out_tmp, args.out)
    print("  done.")


if __name__ == "__main__":
    main()

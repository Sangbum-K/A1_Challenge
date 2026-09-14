"""prebake-logged-only (dt0p1s raceline 호환) — MPC 합성 없이 raw EGO log 의
future steer/ax 를 target 으로, traj 는 dt0p1s raceline (t;x;y;yaw, speed 유도) 에서 sample.

prebake_iy_logged_only.py 와 동일 schema 출력. 차이점은 raceline 만:
  - 옛 형식 (s;x;y;yaw;curv;speed;pit) 대신 dt0p1s (t_s;x;y;psi) + 끝 6행 pit polygon.
  - traj sampling 은 src/utils/raceline_search.sample_traj_ego (검증된 nearest+polygon search,
    speed 는 xy 간격/DT 로 유도). target_times = [0.1*i for i in range(40)] → (40,3) ego frame.
  - control target (steer, ax) 은 logged_only 와 동일하게 raw EGO future 에서.

출력 (prebake_iy_logged_only 와 동일):
  sample_keys "{run}__t{tick:07d}__logged" / sample_tags "logged"
  curr_keys "{run}__t{tick:07d}" / prev_keys "{run}__t{tick-prev_dtick:07d}"
  d_yaws / sample_fx / ego_5d (5) / traj (40,3 ego [x_fwd,y_left,speed])
  target (40,2 [steer_deg, ax]) / mask (40) / obj_target (40) / prev_dtick

사용:
  python scripts/prebake_iy_logged_dt0p1s.py \
      --frame-list .../scc_frame_list_spd_0522_40m.txt \
      --raw-base /home/ailab-12/AILabSSD/02_Data/57_A1_E2E/00_raw \
      --track speedway_track --policy 0522_agent1_40m_pertX_pitX \
      --track-short speedway --map high --data-dir <REPO>/data \
      --aux-dir .../aux_targets_spd_0522_40m \
      --out .../prebake/f1_prebake_iy_spd_0522_40m_logged_prev20.npz \
      --prev-dtick 20
"""
from __future__ import annotations
import os, sys, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.utils import raceline_dt0p1s   # closed_loop 와 동일 dt0p1s waypoint 추출 (정합)

DEG2RAD = np.pi / 180.0
DT = 0.1
N_WP = 40


@lru_cache(maxsize=20000)
def _parse_ego_str(p_str: str):
    p = Path(p_str)
    if not p.exists():
        return None
    out = {}
    try:
        for l in open(p):
            if ':' not in l:
                continue
            k, _, v = l.partition(':')
            k = k.strip(); v = v.strip()
            if k in ('position', 'orientation', 'velocity', 'acceleration', 'angularVelocity'):
                nums = [float(t) for t in v.split() if t not in ('Nan', 'nan')]
                if len(nums) >= 2:
                    out[k] = nums
            elif k in ('steer', 'accel', 'brake'):
                try:
                    out[k] = float(v)
                except Exception:
                    pass
    except Exception:
        return None
    return out if 'position' in out and 'orientation' in out else None


def parse_ego(p):
    return _parse_ego_str(str(p))


# raceline map 은 worker 마다 한 번 로드되도록 module-level 캐시 (ProcessPool fork copy).
_RLMAP = {}


def _get_rlmap(track_short, mp, data_dir):
    key = (track_short, mp, data_dir)
    if key not in _RLMAP:
        _RLMAP[key] = raceline_dt0p1s.load_map(track_short, mp, data_dir)
    return _RLMAP[key]


def process_one(args_tuple):
    (run, tick, raw_base, track, policy, track_short, mp, data_dir,
     aux_dir, prev_dtick) = args_tuple
    try:
        rlmap = _get_rlmap(track_short, mp, data_dir)
        run_dir = Path(raw_base) / track / policy / run
        ego_dir = run_dir / "EGO_INFO"
        ego_curr = parse_ego(ego_dir / f"{tick}.txt")
        if ego_curr is None:
            return None
        prev_tick = tick - prev_dtick
        ego_prev = parse_ego(ego_dir / f"{prev_tick}.txt")
        if ego_prev is None:
            return None

        v = ego_curr.get('velocity', [0, 0, 0])
        a = ego_curr.get('acceleration', [0, 0, 0])
        av = ego_curr.get('angularVelocity', [0, 0, 0])
        ego_5d = np.array([v[0], v[1] if len(v) > 1 else 0,
                           a[0], a[1] if len(a) > 1 else 0,
                           av[2] if len(av) > 2 else 0], dtype=np.float32)
        d_yaw = (ego_curr['orientation'][2] - ego_prev['orientation'][2]) * DEG2RAD

        ex, ey = ego_curr['position'][0], ego_curr['position'][1]
        e_yaw = ego_curr['orientation'][2] * DEG2RAD
        # dt0p1s raceline sampling (ego frame [x_fwd, y_left, speed]) — closed_loop 와 동일 함수
        traj, _meta = raceline_dt0p1s.sample_traj40(rlmap, ex, ey, e_yaw, n=N_WP)

        # control target — future 40 EGO 의 (steer deg, ax)
        target = np.zeros((N_WP, 2), dtype=np.float32)
        mask = np.zeros(N_WP, dtype=np.bool_)
        for i in range(N_WP):
            t_fut = tick + int(round(DT * 1000 * (i + 1)))
            t_fut = (t_fut // 5) * 5
            ego_fut = parse_ego(ego_dir / f"{t_fut}.txt")
            if ego_fut is None:
                continue
            steer_rad = ego_fut.get('steer', 0.0)
            ax_fut = ego_fut.get('acceleration', [0, 0, 0])[0]
            target[i, 0] = steer_rad / DEG2RAD
            target[i, 1] = ax_fut
            mask[i] = True

        obj_t = np.zeros(N_WP, dtype=np.float32)
        if aux_dir is not None:
            fid = f"{track}__{policy}__{run}__t{tick:07d}"
            ap = Path(aux_dir) / f"{fid}.npz"
            if ap.exists():
                try:
                    aux_t = np.load(ap)["target"]
                    obj_t = (aux_t[:, 0] > 0.5).astype(np.float32)
                except Exception:
                    pass

        return dict(
            run=run, tick=tick,
            curr_key=f"{run}__t{tick:07d}",
            prev_key=f"{run}__t{prev_tick:07d}",
            d_yaw=float(d_yaw), ego_5d=ego_5d,
            traj=traj, target=target, mask=mask, obj_target=obj_t,
        )
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-list", required=True)
    ap.add_argument("--raw-base", required=True)
    ap.add_argument("--track", required=True, help="raw 경로용 (예: speedway_track)")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--track-short", required=True, help="raceline_dt0p1s.load_map 용 (sangam|speedway)")
    ap.add_argument("--map", default="high", help="raceline map: high|mild")
    ap.add_argument("--data-dir", required=True, help="data/ (control/{track}_{map}_*.csv 위치)")
    ap.add_argument("--aux-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prev-dtick", type=int, default=20)
    ap.add_argument("--fx", type=float, default=640.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rlmap = raceline_dt0p1s.load_map(args.track_short, args.map, args.data_dir)
    print(f"[prebake-dt0p1s] raceline map={args.track_short}/{args.map} path={rlmap.get('path')}")
    print(f"  raceline points: {rlmap['N']}  polygon: "
          f"{None if rlmap['polygon'] is None else np.asarray(rlmap['polygon']).shape}  "
          f"P={rlmap['P']} M={rlmap['M']}")

    frames = []
    with open(args.frame_list) as f:
        for l in f:
            t = l.strip().split()
            if len(t) >= 2:
                frames.append((t[0], int(t[1])))
    if args.limit:
        frames = frames[:args.limit]
    print(f"[prebake-dt0p1s] frames: {len(frames)}")

    tasks = [(run, tick, args.raw_base, args.track, args.policy,
              args.track_short, args.map, args.data_dir, args.aux_dir, args.prev_dtick)
             for run, tick in frames]

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
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-3)
                print(f"  [{i+1}/{N}] valid={n_valid}  rate={rate:.0f}/s  "
                      f"ETA={(N-i-1)/max(rate,1):.0f}s", flush=True)
    print(f"\n[done] valid={n_valid}/{N}  time={time.time()-t0:.0f}s")

    out_tmp = args.out + ".partial.npz"
    print(f"[prebake-dt0p1s] saving → {args.out}")
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

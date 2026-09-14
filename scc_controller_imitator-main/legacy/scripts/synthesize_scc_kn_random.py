"""Phase 2: K=N SCC GT 합성 — 각 image frame 마다 random trajectory 세트 sampling.

Memory: scc-synth-k1-first.md
- Phase 1 (K=1): MultiAgnet 자체 wp_rel_gt 만
- Phase 2 (K=N): logged + (N-1) random trajectory (v0237 raw bag pool 에서)
  → 각 image frame 의 lead car 정보는 그대로 + trajectory 만 다양화 → 일반화 ↑

Args:
  --k 5            # 각 image 당 trajectory 수 (logged 1 + random K-1)
  --vx-tolerance 5 # random trajectory sampling 시 ego_vx ± 이 값 안의 trajectory 만
  --out scc_synth_kn_v1.pkl

trajectory pool 빌드:
  v0237 raw bag mpc_trajectory_generator_*.bag 의 ControlInputOutput.trajectory 시퀀스들
  → 미리 ml_con 에서 npz 로 변환 (별도 스크립트 build_trajectory_pool.py 필요)
  → 각 trajectory entry = (vehicle_speed, trajectory points)
  → vx-tolerance 로 ego.vx 와 비슷한 것만 sampling
"""
from __future__ import annotations
import os, sys, json, pickle, time, argparse
sys.path.insert(0, "/home/ailab-12/git/A1_Rule/devel/lib/python3/dist-packages")
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "acados_scc", "scripts")
sys.path.insert(0, SCRIPTS_DIR)
BUILD_CACHE = os.path.join(os.path.dirname(SCRIPTS_DIR), "build_cache")
os.makedirs(BUILD_CACHE, exist_ok=True)
os.chdir(BUILD_CACHE)

import numpy as np
from MPC.NMPC_class_delayed_kinematic_cat import (
    Nonlinear_Delayed_Kinematic_Cat_Model_Predictive_Controller as MPC_Class,
)
from synthesize_scc_k1 import (  # 공통 helper 재사용
    bev_bbox_to_ego_local, make_vs_msg, wp_rel_to_traj_msg,
    lead_to_objs_msg, run_cycle, downsample_81_to_30,
    pick_front_lead, get_image_paths,
    COMPACT_NPZ, COMPACT_META, RAW_IMG_BASE, INDEX_JSON,
)
from autohyu_msgs.msg import Trajectory, TrajectoryPoint, PredictObjects

TRAJ_POOL_NPZ = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/v0237_traj_pool.npz"
OUT_DIR = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_kn_v1"
N_ITER_RTI = 20  # was 8 — 충분 수렴 위해 ↑


def build_random_traj_msg(pool_traj_arr):
    """pool 의 trajectory (N, 4) [x, y, yaw, speed] → ROS Trajectory msg (ego-local frame)."""
    traj = Trajectory()
    for i in range(len(pool_traj_arr)):
        p = TrajectoryPoint()
        p.x = float(pool_traj_arr[i, 0]); p.y = float(pool_traj_arr[i, 1])
        p.yaw = float(pool_traj_arr[i, 2]); p.speed = float(pool_traj_arr[i, 3])
        p.time = i * 0.1
        traj.point.append(p)
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--vx-tolerance", type=float, default=5.0)
    ap.add_argument("--out", default="scc_synth_kn_v1.pkl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=999999)
    ap.add_argument("--out-dir-override", default=None)
    ap.add_argument("--lead-filter", choices=["all", "lead-only", "no-lead"], default="all",
                    help="lead-only: lead 있는 frame 만; no-lead: lead 없는 frame 만; all: 모두")
    ap.add_argument("--max-frames", type=int, default=999999,
                    help="필터 후 최대 frame 수 (random sample)")
    ap.add_argument("--frame-indices-file", default=None,
                    help="이 file 의 frame index (한 줄당 하나) 만 process. start/end/filter 무시")
    args = ap.parse_args()
    global OUT_DIR
    if args.out_dir_override:
        OUT_DIR = args.out_dir_override

    rng = np.random.RandomState(args.seed)

    print("loading compact npz + meta ...")
    z = np.load(COMPACT_NPZ)
    meta_list = json.loads(open(COMPACT_META).read())
    if args.frame_indices_file:
        with open(args.frame_indices_file) as f:
            wanted = set(int(line.strip()) for line in f if line.strip())
        meta_list = [m for m in meta_list if m["i"] in wanted]
        print(f"  {len(meta_list)} frames from file (wanted={len(wanted)})")
    else:
        meta_list = [m for m in meta_list if args.start <= m["i"] < args.end]
        print(f"  {len(meta_list)} frames in range [{args.start}, {args.end})")
    # lead filter
    def has_front_lead(m):
        for o in m["objects"]:
            xm, ym, _, _ = bev_bbox_to_ego_local(o["bbox"])
            if xm > 2.0 and abs(ym) < 3.5 and xm < 120.0:
                return True
        return False
    if args.lead_filter == "lead-only":
        meta_list = [m for m in meta_list if has_front_lead(m)]
    elif args.lead_filter == "no-lead":
        meta_list = [m for m in meta_list if not has_front_lead(m)]
    print(f"  after lead-filter [{args.lead_filter}]: {len(meta_list)} frames")
    if args.max_frames < len(meta_list):
        rng_local = np.random.RandomState(args.seed)
        idxs = rng_local.choice(len(meta_list), args.max_frames, replace=False)
        meta_list = [meta_list[int(i)] for i in sorted(idxs)]
        print(f"  after max-frames {args.max_frames}: {len(meta_list)} frames")

    print(f"\nloading trajectory pool: {TRAJ_POOL_NPZ}")
    if not os.path.exists(TRAJ_POOL_NPZ):
        raise FileNotFoundError(
            f"trajectory pool not built yet. run scripts/build_trajectory_pool.py first."
        )
    tp = np.load(TRAJ_POOL_NPZ)
    pool_trajs = tp["trajs"]      # (P, 40, 4) [x, y, yaw, speed]
    pool_v0s = tp["v0s"]          # (P,) initial speed of each trajectory
    print(f"  pool size: {len(pool_trajs)}")

    idx_data = json.loads(open(INDEX_JSON).read())
    run_dt = {e["run_id"]: float(e.get("dt_per_tick_ms", 10.0)) for e in idx_data}

    mpc = MPC_Class()
    print(f"MPC: N={mpc.N_MPC} Ts={mpc.Ts_MPC}\n")
    os.makedirs(OUT_DIR, exist_ok=True)

    from collections import defaultdict
    by_run = defaultdict(list)
    for m in meta_list:
        by_run[m["run_id"]].append(m)

    all_samples = []
    total = len(meta_list) * args.k
    done = 0
    t_start = time.time()

    for run_id in sorted(by_run.keys()):
        run_metas = sorted(by_run[run_id], key=lambda m: m["tick"])
        dt_per_tick = run_dt.get(run_id, 10.0)
        mpc.is_MPC_init = True
        mpc.pred_X = None; mpc.pred_U = None; mpc.X_init = None
        mpc.fallback_count = 0; mpc.last_mpc_status = None

        for meta in run_metas:
            i = meta["i"]
            ego = z[f"s{i}__ego_pose"]
            wp = z[f"s{i}__wp_rel_gt"]
            vx = float(ego[3])
            if vx < 0.5:
                done += args.k; continue
            curr_path, prev_path = get_image_paths(meta, dt_per_tick)
            if not os.path.exists(curr_path) or not os.path.exists(prev_path):
                done += args.k; continue
            lead = pick_front_lead(meta["objects"])

            # K trajectory: 1 logged + (K-1) random
            traj_set = []
            # 1) logged trajectory (자체 wp_rel_gt 누적)
            abs_xy = np.cumsum(wp[:, :2], axis=0)
            abs_yaw = np.cumsum(wp[:, 2], axis=0)
            logged = np.column_stack([abs_xy[:, 0], abs_xy[:, 1], abs_yaw, wp[:, 3]])
            traj_set.append(("logged", logged))
            # 2) (K-1) random from pool, vx 호환 필터
            mask = np.abs(pool_v0s - vx) < args.vx_tolerance
            valid_idx = np.where(mask)[0]
            if len(valid_idx) < args.k - 1:
                # tolerance 못 채우면 그냥 풀 전체에서 sample
                valid_idx = np.arange(len(pool_trajs))
            chosen = rng.choice(valid_idx, size=args.k - 1, replace=False)
            for c in chosen:
                traj_set.append((f"rand_{c}", pool_trajs[c]))

            # 각 trajectory 에 대해 MPC 호출
            for tag, traj_local in traj_set:
                ax_est = float((traj_local[0, 3] - vx) / 0.1) if vx > 0 else 0.0
                vs_msg = make_vs_msg(ego, ax=ax_est, ay=0.0, tire_angle=0.0)
                traj_msg = build_random_traj_msg(traj_local)
                objs_msg = lead_to_objs_msg(lead)
                cio = None
                # 같은 input N iter (warm chain 은 trajectory 바뀌면 끊김 — reset)
                mpc.is_MPC_init = True
                mpc.pred_X = None; mpc.pred_U = None; mpc.X_init = None
                for _ in range(N_ITER_RTI):
                    out = run_cycle(mpc, vs_msg, traj_msg, objs_msg)
                    _, _, _, _, cio_now = out
                    if cio_now is not None:
                        cio = {
                            'steer': np.asarray(cio_now['steer'], dtype=np.float64).copy(),
                            'acceleration': np.asarray(cio_now['acceleration'], dtype=np.float64).copy(),
                        }
                if cio is None:
                    done += 1; continue
                gt_steer_30 = downsample_81_to_30(cio['steer']).astype(np.float32)
                gt_accel_30 = downsample_81_to_30(cio['acceleration']).astype(np.float32)
                ego_5d = np.array([vx, float(ego[4]), ax_est, 0.0, float(ego[5])], dtype=np.float32)
                if lead is None:
                    lead_info = np.zeros(6, dtype=np.float32); lead_valid = False
                else:
                    lead_info = np.array([lead[0], lead[1], lead[4][0], lead[4][1], lead[2], lead[3]], dtype=np.float32)
                    lead_valid = True
                sample = {
                    "frame_id": meta["frame_id"], "traj_tag": tag,
                    "image_curr_path": curr_path, "image_prev_path": prev_path,
                    "ego_5d": ego_5d, "traj_30": traj_local.astype(np.float32),
                    "lead_info": lead_info, "lead_valid": bool(lead_valid),
                    "gt_steer_30": gt_steer_30, "gt_accel_30": gt_accel_30,
                }
                all_samples.append(sample)
                done += 1
                if done % 100 == 0:
                    elapsed = time.time() - t_start
                    rate = done / elapsed
                    eta = (total - done) / rate
                    # incremental save (acados segfault 방어)
                    out_pkl = os.path.join(OUT_DIR, args.out)
                    with open(out_pkl, "wb") as f:
                        pickle.dump({"samples": all_samples, "K": args.k}, f, protocol=4)
                    print(f"  [{done}/{total}] {rate:.1f}/s ETA {eta/60:.1f}min ok={len(all_samples)} saved={os.path.getsize(out_pkl)/1e6:.1f}MB")

    out_pkl = os.path.join(OUT_DIR, args.out)
    with open(out_pkl, "wb") as f:
        pickle.dump({"samples": all_samples, "K": args.k}, f, protocol=4)
    print(f"\nwrote {out_pkl} ({os.path.getsize(out_pkl)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

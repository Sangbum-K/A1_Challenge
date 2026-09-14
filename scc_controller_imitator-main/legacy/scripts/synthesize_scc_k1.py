"""K=1 SCC GT 합성 파이프라인.

각 jiwon_e2e MultiAgnet frame 에 대해:
  - image_curr_path, image_prev_path (raw camera 1 jpg, 100 ms 간격 prev tick)
  - ego (5-d): vx, vy, ax, ay, yaw_rate
  - trajectory (40, 4): wp_rel_gt 누적한 (x, y, yaw, speed) ego-local
  - objects: lead car 시계열 (40 step, 등속) — 가장 가까운 전방 lead 한 대 사용 (없으면 빈 PredictObjects)
  - GT control: MPC RTI 8 iter 결과의 control_io = (steer_seq_81, accel_seq_81)
                → 100 ms 간격 30 step 으로 downsample

Warm chain: 같은 run 내에서 frame 시간순으로 처리, reset 없이 warm.
K=1 (logged trajectory only — memory: scc-synth-k1-first).
"""
from __future__ import annotations
import os, sys, time, pickle, glob, json
sys.path.insert(0, "/home/ailab-12/git/A1_Rule/devel/lib/python3/dist-packages")
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "acados_scc", "scripts")
sys.path.insert(0, SCRIPTS_DIR)
BUILD_CACHE = os.path.join(os.path.dirname(SCRIPTS_DIR), "build_cache")
os.chdir(BUILD_CACHE)

import numpy as np
from MPC.NMPC_class_delayed_kinematic_cat import (
    Nonlinear_Delayed_Kinematic_Cat_Model_Predictive_Controller as MPC_Class,
)
from utils.trajectory_process import (
    npize_vehicle_state, npize_trajectory_local, npize_objects, resample_objects,
    generate_road_map, frenet_transform, frenet_mu_transform, global_to_local,
)
from autohyu_msgs.msg import (
    VehicleState, Trajectory, TrajectoryPoint,
    PredictObjects, PredictObject, Object3DState, ObjectDimension,
)

COMPACT_NPZ = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/multiagnet_compact.npz"
COMPACT_META = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/multiagnet_compact_meta.json"
RAW_IMG_BASE = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/data/raw/sangam_track/MultiAgnet"
OUT_DIR = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/scc_synth_v1"
INDEX_JSON = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/data/processed/index.json"

BEV_X_MIN, BEV_X_MAX = -20.0, 80.0
BEV_Y_MIN, BEV_Y_MAX = -25.0, 25.0
BEV_CELL = 0.5
N_ITER_RTI = 20  # was 8 — 충분 수렴 위해 ↑

# 차선 폭의 절반 — lead obj 가 |n| > 이 값 이면 MPC 시야에서 제외 (lane filter).
# 원본 ROS node 의 `filtering_objects_in_lane` 의도와 동일하나 (a) ±1 → ±2 로 완화,
# (b) s_max bug fix (reference_map[:, 3] 미초기화 대신 cum_s[-1] 사용).
LANE_HALF_WIDTH = 2.0


def bev_bbox_to_ego_local(bbox):
    """BEV pixel bbox → ego frame (jiwon_e2e _ego_xy_to_bev_pixel 의 역변환).

    Pixel convention (`src/data/map_raster.py::_ego_xy_to_bev_pixel`):
        col = (ego_y - y_min) / cell           # increasing left
        row = (x_max - ego_x) / cell           # row 0 = forward edge

    bbox = (cx_pix, cy_pix, w_pix, h_pix) where
        cx = mean col, cy = mean row
        w_pix = col-extent (y-direction → vehicle width)
        h_pix = row-extent (x-direction → vehicle length)

    Returns (ego_x, ego_y, L, W).
    """
    cx, cy, w_cells, h_cells = bbox
    ego_x = BEV_X_MAX - cy * BEV_CELL
    ego_y = BEV_Y_MIN + cx * BEV_CELL
    L = h_cells * BEV_CELL   # vehicle length (x extent)
    W = w_cells * BEV_CELL   # vehicle width  (y extent)
    return (ego_x, ego_y, L, W)


def make_vs_msg(ego_pose, ax=0.0, ay=0.0, tire_angle=0.0):
    vs = VehicleState()
    vs.x = 0.0; vs.y = 0.0; vs.yaw = 0.0
    vs.vx = float(ego_pose[3]); vs.vy = float(ego_pose[4]); vs.vz = 0.0
    vs.ax = float(ax); vs.ay = float(ay); vs.az = 0.0
    vs.roll = float(ego_pose[2]); vs.pitch = float(ego_pose[1])
    vs.yaw_vel = float(ego_pose[5])
    vs.vehicle_can.steering_tire_angle = float(tire_angle)
    return vs


def wp_rel_to_traj_msg(wp_rel_gt):
    abs_xy = np.cumsum(wp_rel_gt[:, :2], axis=0)
    abs_yaw = np.cumsum(wp_rel_gt[:, 2], axis=0)
    abs_v = wp_rel_gt[:, 3]
    traj = Trajectory()
    p0 = TrajectoryPoint()
    p0.x = 0.0; p0.y = 0.0; p0.yaw = 0.0; p0.speed = float(abs_v[0])
    traj.point.append(p0)
    for i in range(len(abs_xy)):
        p = TrajectoryPoint()
        p.x = float(abs_xy[i, 0]); p.y = float(abs_xy[i, 1])
        p.yaw = float(abs_yaw[i]); p.speed = float(abs_v[i])
        p.time = (i + 1) * 0.1
        traj.point.append(p)
    return traj


def lead_to_objs_msg(lead, n_states=40, dt=0.1):
    """lead = (x, y, length_m, width_m, velocity_2d_ego_frame)"""
    objs = PredictObjects()
    if lead is None:
        return objs
    o = PredictObject()
    o.id = 1; o.classification = 1; o.dynamic_state = 0
    o.dimension = ObjectDimension(length=lead[2], width=lead[3], height=1.5)
    lx, ly = lead[0], lead[1]
    lvx, lvy = float(lead[4][0]), float(lead[4][1])
    for i in range(n_states):
        t = i * dt
        s = Object3DState()
        s.x = lx + lvx * t; s.y = ly + lvy * t; s.z = 0.0
        s.yaw = 0.0; s.vx = lvx; s.vy = lvy
        o.state.append(s)
    objs.object.append(o)
    return objs


def run_cycle(mpc, vs_msg, traj_msg, objs_msg):
    vehicle_state, measured_state = npize_vehicle_state(vs_msg)
    target_trajectory, reference_map = npize_trajectory_local(traj_msg, vehicle_state)
    objects = npize_objects(objs_msg, max_objects_num=1)
    objects = resample_objects(objects, 0.1, 40, mpc.Ts_MPC, mpc.N_MPC + 1)
    road_map = generate_road_map(reference_map)
    vsg = vehicle_state.copy()
    vehicle_state[0, 0] = 0.0; vehicle_state[0, 1] = 0.0; vehicle_state[0, 2] = 0.0
    vehicle_state = frenet_transform(road_map, vehicle_state)
    vehicle_state = frenet_mu_transform(road_map, vehicle_state)
    trajectory_sn = frenet_transform(road_map, target_trajectory)
    trajectory_sn = frenet_mu_transform(road_map, trajectory_sn, start_mu=vehicle_state[0, 6])
    cum_s = np.zeros(len(reference_map))
    if len(reference_map) > 1:
        dxy = np.diff(reference_map[:, :2], axis=0)
        cum_s[1:] = np.cumsum(np.hypot(dxy[:, 0], dxy[:, 1]))
    s_max = float(cum_s[-1]) if len(cum_s) > 0 else 0.0
    ref_xy = reference_map[:, :2]
    R = np.array([[np.cos(-vsg[0, 2]), -np.sin(-vsg[0, 2])], [np.sin(-vsg[0, 2]), np.cos(-vsg[0, 2])]])
    sn_objects = np.zeros((objects.shape[0], objects.shape[1], 11))
    for i in range(objects.shape[0]):
        if objects[i, :, 4].all():
            obj = objects[i, :, :]
            for j in range(obj.shape[0]):
                lp = global_to_local(np.array([obj[j, 0], obj[j, 1]]), vsg, R)
                dxs = ref_xy[:, 0] - lp[0]; dys = ref_xy[:, 1] - lp[1]
                k = int(np.argmin(np.hypot(dxs, dys)))
                obj_s = cum_s[k]
                if k + 1 < len(ref_xy):
                    seg = ref_xy[k + 1] - ref_xy[k]
                else:
                    seg = ref_xy[k] - ref_xy[k - 1]
                seg_n = np.hypot(seg[0], seg[1])
                if seg_n > 1e-6:
                    seg /= seg_n
                    rel = np.array([lp[0] - ref_xy[k, 0], lp[1] - ref_xy[k, 1]])
                    obj_n = seg[0] * rel[1] - seg[1] * rel[0]
                else:
                    obj_n = 0.0
                # lane filter: |n| ≤ LANE_HALF_WIDTH AND 0 ≤ s ≤ s_max 인 state 만 활성.
                # 차선 밖 / s 범위 밖이면 sn_objects[i, j, :] = 0 (valid=False) + s=-1e6 로 invalidate.
                if abs(obj_n) <= LANE_HALF_WIDTH and 0.0 <= obj_s <= s_max:
                    sn_objects[i, j, :2] = lp
                    sn_objects[i, j, 2:8] = obj[j, 2:8]
                    sn_objects[i, j, 8] = obj_s; sn_objects[i, j, 9] = obj_n
                else:
                    sn_objects[i, j, 8] = -1e6
        else:
            sn_objects[i, :, 8] = -1e6
    return mpc.calculate_optimal(road_map, vehicle_state, sn_objects, trajectory_sn, measured_state)


def downsample_81_to_30(seq_81, target_len=30, target_dt=0.1, source_dt=0.05):
    """81-step (0.05 s) → 30-step (0.1 s) linear interp."""
    arr = np.asarray(seq_81, dtype=np.float64)
    src_t = np.arange(len(arr)) * source_dt
    tgt_t = np.arange(target_len) * target_dt
    return np.interp(tgt_t, src_t, arr)


def pick_front_lead(objects, max_x=120.0, max_y_abs=3.5):
    """objects (list of dict, compact npz 의 obj_list) → 가장 가까운 전방 lead 또는 None."""
    best = None
    best_x = max_x
    for o in objects:
        x_m, y_m, L, W = bev_bbox_to_ego_local(o["bbox"])
        if x_m > 2.0 and abs(y_m) < max_y_abs and x_m < best_x:
            best_x = x_m
            v = o.get("velocity", [0.0, 0.0])
            best = (x_m, y_m, L, W, np.asarray(v, dtype=np.float64))
    return best


def get_image_paths(meta, dt_per_tick_ms=10):
    """jiwon_e2e meta {track, policy, run_id, tick} → (curr_jpg, prev_jpg)."""
    run = meta["run_id"]; tick = int(meta["tick"])
    prev_tick = tick - int(round(100.0 / dt_per_tick_ms))
    base = f"{RAW_IMG_BASE}/{run}/CAMERA_1"
    curr = f"{base}/{tick}_Intensity.jpeg"
    prev = f"{base}/{prev_tick}_Intensity.jpeg"
    return curr, prev


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=999999)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    print(f"loading index.json for dt_per_tick_ms ...")
    idx_data = json.loads(open(INDEX_JSON).read())
    run_dt = {}
    for e in idx_data:
        run_dt[e["run_id"]] = float(e.get("dt_per_tick_ms", 10.0))
    print(f"  {len(run_dt)} runs")

    print(f"\nloading compact npz: {COMPACT_NPZ}")
    z = np.load(COMPACT_NPZ)
    print(f"  {len(z.files)} arrays")
    meta_list = json.loads(open(COMPACT_META).read())
    print(f"  {len(meta_list)} meta entries")
    # frame index range filter
    meta_list = [m for m in meta_list if args.start <= m["i"] < args.end]
    print(f"  range [{args.start}, {args.end}): {len(meta_list)} frames")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Group by run for warm chain
    from collections import defaultdict
    by_run = defaultdict(list)
    for m in meta_list:
        by_run[m["run_id"]].append(m)
    print(f"  {len(by_run)} unique runs")

    mpc = MPC_Class()
    print(f"\nMPC: N={mpc.N_MPC} Ts={mpc.Ts_MPC}\n")

    fp_warm = False  # 첫 frame 만 cold
    total = sum(len(v) for v in by_run.values())
    done = 0
    fail = 0
    skipped_no_image = 0
    skipped_load = 0
    t_start = time.time()

    # 누적 sample 들 (메모리 — 6877 frame 정도면 가능)
    all_samples = []

    for run_id in sorted(by_run.keys()):
        run_metas = sorted(by_run[run_id], key=lambda m: m["tick"])
        dt_per_tick = run_dt.get(run_id, 10.0)
        # 첫 frame 만 reset; warm chain
        mpc.is_MPC_init = True
        mpc.pred_X = None; mpc.pred_U = None; mpc.X_init = None
        mpc.fallback_count = 0; mpc.last_mpc_status = None

        for fi, meta in enumerate(run_metas):
            i = meta["i"]
            ego = z[f"s{i}__ego_pose"]
            wp = z[f"s{i}__wp_rel_gt"]
            vx = float(ego[3])
            if vx < 2.0:    # QP fail 줄이기 위해 stop frame 제외 강화
                skipped_load += 1; done += 1; continue

            curr_path, prev_path = get_image_paths(meta, dt_per_tick)
            if not os.path.exists(curr_path) or not os.path.exists(prev_path):
                skipped_no_image += 1; done += 1; continue

            lead = pick_front_lead(meta["objects"])
            ax_est = float((wp[0, 3] - vx) / 0.1)
            vs_msg = make_vs_msg(ego, ax=ax_est, ay=0.0, tire_angle=0.0)
            traj_msg = wp_rel_to_traj_msg(wp)
            objs_msg = lead_to_objs_msg(lead)

            # RTI 8 iter (warm chain)
            cio = None
            for _ in range(N_ITER_RTI):
                out = run_cycle(mpc, vs_msg, traj_msg, objs_msg)
                _, _, _, _, cio_now = out
                if cio_now is not None:
                    cio = {
                        'steer': np.asarray(cio_now['steer'], dtype=np.float64).copy(),
                        'acceleration': np.asarray(cio_now['acceleration'], dtype=np.float64).copy(),
                    }
            if cio is None:
                fail += 1; done += 1; continue

            # downsample 81 → 30
            gt_steer_30 = downsample_81_to_30(cio['steer'])
            gt_accel_30 = downsample_81_to_30(cio['acceleration'])

            # ego 5-d: vx, vy, ax, ay, yaw_rate
            ego_5d = np.array([vx, float(ego[4]), ax_est, 0.0, float(ego[5])], dtype=np.float32)
            # trajectory 30 step: wp_rel_gt 누적 (x, y, yaw, speed)
            traj_30 = np.zeros((30, 4), dtype=np.float32)
            traj_30[:, 0] = np.cumsum(wp[:, 0])
            traj_30[:, 1] = np.cumsum(wp[:, 1])
            traj_30[:, 2] = np.cumsum(wp[:, 2])
            traj_30[:, 3] = wp[:, 3]
            # lead info: (x, y, vx, vy, L, W) or all zeros if no lead
            if lead is None:
                lead_info = np.zeros(6, dtype=np.float32)
                lead_valid = False
            else:
                lead_info = np.array([lead[0], lead[1], lead[4][0], lead[4][1], lead[2], lead[3]], dtype=np.float32)
                lead_valid = True

            sample = {
                "frame_id": meta["frame_id"],
                "image_curr_path": curr_path,
                "image_prev_path": prev_path,
                "ego_5d": ego_5d,
                "traj_30": traj_30,
                "lead_info": lead_info,
                "lead_valid": bool(lead_valid),
                "gt_steer_30": gt_steer_30.astype(np.float32),
                "gt_accel_30": gt_accel_30.astype(np.float32),
            }
            all_samples.append(sample)
            done += 1

            if done % 100 == 0:
                elapsed = time.time() - t_start
                rate = done / elapsed
                eta = (total - done) / rate
                print(f"  [{done}/{total}] {rate:.1f}/s ETA {eta/60:.1f}min  "
                      f"fail={fail} no_img={skipped_no_image} stop={skipped_load}  ok={len(all_samples)}")

    print(f"\n=== done in {(time.time()-t_start)/60:.1f}min ===")
    print(f"  total processed: {done}")
    print(f"  success: {len(all_samples)}")
    print(f"  fail (MPC fallback): {fail}")
    print(f"  skipped no_image: {skipped_no_image}")
    print(f"  skipped (low speed): {skipped_load}")

    # Save as single pkl (chunk suffix)
    out_pkl = os.path.join(OUT_DIR, f"scc_synth_k1_v1{args.out_suffix}.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump({"samples": all_samples, "K": 1, "version": "v1"}, f, protocol=4)
    print(f"\nwrote {out_pkl} ({os.path.getsize(out_pkl)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

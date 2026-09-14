"""Workflow A sanity test.

목적: v0237 trajectory + jiwon_e2e MultiAgnet ego/lead-car 조합으로 acados_scc MPC 가
실제 의미 있는 control 출력을 만들어내는지 확인.

실행 (반드시 system py3.8 + ROS env):
    /usr/bin/python3 scripts/sanity_test_workflow_A.py

3 단계 점진 검증:
    Step 1: jiwon_e2e ego + jiwon_e2e own future trajectory + NO lead cars
            → MPC 가 단순 trajectory tracking 으로 풀리는지
    Step 2: jiwon_e2e ego + v0237 trajectory + NO lead cars
            → trajectory pool swap 이 동작하는지
    Step 3: jiwon_e2e ego + v0237 trajectory + jiwon_e2e lead cars
            → 풀 SCC 시나리오
"""
from __future__ import annotations

import os
import sys
import time
import pickle
import glob
import numpy as np

# --- path setup ---------------------------------------------------------------
SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "external", "acados_scc", "scripts",
)
sys.path.insert(0, "/home/ailab-12/git/A1_Rule/devel/lib/python3/dist-packages")
sys.path.insert(0, SCRIPTS_DIR)

# build_cache 디렉터리로 cd — acados c_generated_code 가 여기 생성됨
BUILD_CACHE = os.path.join(
    os.path.dirname(SCRIPTS_DIR), "build_cache"
)
os.makedirs(BUILD_CACHE, exist_ok=True)
os.chdir(BUILD_CACHE)

from MPC.NMPC_class_delayed_kinematic_cat import (
    Nonlinear_Delayed_Kinematic_Cat_Model_Predictive_Controller as MPC_Class,
)
from utils.trajectory_process import (
    generate_road_map,
    frenet_transform,
    frenet_mu_transform,
    resample_points,
)

# --- data paths ---------------------------------------------------------------
V0237_SAMPLE_NPZ = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/v0237_sample10.npz"
JIWON_E2E_FRAMES_GLOB = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/data/processed/frames/sangam_track__MultiAgnet__260513_10__*.pkl"


# --- jiwon_e2e helpers --------------------------------------------------------
def load_jiwon_frame(pkl_path):
    """Load one jiwon_e2e pkl. Returns dict with ego_pose, wp_rel_gt, objects."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def wp_rel_to_xy_v_yaw(wp_rel_gt):
    """jiwon_e2e wp_rel_gt (30, 4) [Δx,Δy,Δyaw,v] → cumulated (30, 4) [x,y,yaw,v] in ego frame."""
    abs_xy = np.cumsum(wp_rel_gt[:, :2], axis=0)
    abs_yaw = np.cumsum(wp_rel_gt[:, 2], axis=0)
    abs_v = wp_rel_gt[:, 3]
    return np.stack([abs_xy[:, 0], abs_xy[:, 1], abs_yaw, abs_v], axis=1)


def bev_obj_to_egolocal(obj_bbox, bev_x_min=-20.0, bev_y_min=-25.0, cell_m=0.5):
    """jiwon_e2e BEV bbox (cx, cy, w_cells, h_cells) → ego-local (x, y, length_m, width_m).

    Note: BEV grid is x=forward (200 cells from -20 to 80), y=left (100 cells from -25 to 25).
    """
    cx, cy, w, h = obj_bbox
    x_m = bev_x_min + cx * cell_m
    y_m = bev_y_min + cy * cell_m
    length_m = h * cell_m   # bbox h axis is along x (forward)? — check carefully
    width_m = w * cell_m
    return x_m, y_m, length_m, width_m


# --- MPC input builders -------------------------------------------------------
def build_reference_map_from_path(xy_path):
    """xy_path (N, 2) ego-local → reference_map (M, 5) [x, y, ...] padded forward.

    MPC requires reference_map to span enough s-distance. We:
      1. take xy_path
      2. linearly extend forward by ~50 m beyond last point
    """
    if len(xy_path) < 2:
        raise ValueError("xy_path needs ≥2 points")
    dx = xy_path[-1, 0] - xy_path[-2, 0]
    dy = xy_path[-1, 1] - xy_path[-2, 1]
    seg_len = max(np.hypot(dx, dy), 1e-3)
    direction = np.array([dx, dy]) / seg_len
    n_extend = 50
    extra = np.zeros((n_extend, 2))
    for i in range(n_extend):
        extra[i] = xy_path[-1] + direction * (i + 1) * seg_len
    full = np.vstack([xy_path, extra])
    ref = np.zeros((len(full), 5))
    ref[:, 0] = full[:, 0]
    ref[:, 1] = full[:, 1]
    return ref


def build_vehicle_state_from_jiwon(road_map, ego_pose, ego_pkl=None):
    """jiwon_e2e ego_pose [yaw, pitch, roll, vx, vy, wz] →
       vehicle_state (1, 7) [x=0, y=0, yaw=0, speed, s, n, mu] in ego-local + Frenet."""
    vx = float(ego_pose[3])
    speed = vx if vx > 0 else 0.01
    vs = np.zeros((1, 7))
    vs[0, 0] = 0.0
    vs[0, 1] = 0.0
    vs[0, 2] = 0.0
    vs[0, 3] = speed
    vs = frenet_transform(road_map, vs)
    vs = frenet_mu_transform(road_map, vs)
    return vs


def build_measured_state(ego_pose, tire_angle_rad=0.0):
    """(1, 8) [x, y, yaw, vx, vy, ax, ay, tire_angle]."""
    vx, vy, wz = float(ego_pose[3]), float(ego_pose[4]), float(ego_pose[5])
    ms = np.zeros((1, 8))
    ms[0, 3] = vx
    ms[0, 4] = vy
    # ax, ay 0 placeholder (jiwon_e2e pose 에 직접 없으므로 0)
    ms[0, 7] = tire_angle_rad
    return ms


def build_point_from_xy_path(road_map, xy_speed_path, N, Ts, ego_s):
    """xy_speed_path (M, 4) [x, y, yaw, speed] → point (N+1, 7) Frenet, resampled."""
    target_resampled = resample_points(xy_speed_path, ego_s, Ts, N + 1)
    p = frenet_transform(road_map, target_resampled)
    p = frenet_mu_transform(road_map, p)
    return p


# --- the sanity steps ---------------------------------------------------------
def run_step(label, mpc, road_map, vehicle_state, objects, point, measured_state):
    print(f"\n========== {label} ==========")
    print(f"vehicle_state s={vehicle_state[0,4]:.2f} n={vehicle_state[0,5]:.3f} mu={vehicle_state[0,6]:.3f} speed={vehicle_state[0,3]:.2f}")
    print(f"point[0]    x={point[0,0]:.2f} y={point[0,1]:.2f} speed={point[0,3]:.2f}")
    print(f"point[40]   x={point[40,0]:.2f} y={point[40,1]:.2f} speed={point[40,3]:.2f}")
    print(f"point[80]   x={point[80,0]:.2f} y={point[80,1]:.2f} speed={point[80,3]:.2f}")
    obj_valid = bool(objects[:, :, 4].any())
    if obj_valid:
        valid_steps = objects[0, :, 4].astype(bool)
        print(f"objects[0]: valid at {valid_steps.sum()} steps, first x={objects[0,0,0]:.2f} y={objects[0,0,1]:.2f}")
    else:
        print(f"objects: none")
    mpc.is_MPC_init = True  # reset
    t0 = time.time()
    out = mpc.calculate_optimal(road_map, vehicle_state, objects, point, measured_state)
    dt = time.time() - t0
    target_cart, target_pred, target_input, stats, control_io = out
    print(f"solve time: {dt*1000:.1f} ms")
    print(f"target_input [steer_deg, torque, accel_mps2]: {target_input}")
    if control_io is None:
        print("FAILED — control_io=None (fallback engaged)")
        return False
    print(f"steer seq [:10] (deg): {np.round(np.asarray(control_io['steer'][:10]), 3)}")
    print(f"accel seq [:10] (m/s²): {np.round(np.asarray(control_io['acceleration'][:10]), 3)}")
    nonzero = (np.abs(control_io['steer']) > 1e-6).any() or (np.abs(control_io['acceleration']) > 1e-6).any()
    print(f"non-trivial output: {nonzero}")
    return nonzero


# --- main --------------------------------------------------------------------
def main():
    print(f"loading v0237 samples from {V0237_SAMPLE_NPZ}")
    z = np.load(V0237_SAMPLE_NPZ)
    # reconstruct per-sample dicts from "s{i}__{key}" flat keys
    by_sample = {}
    for k in z.files:
        i_str, key = k.split("__", 1)
        idx = int(i_str[1:])
        by_sample.setdefault(idx, {})[key] = z[k]
    v0237_samples = [by_sample[i] for i in sorted(by_sample)]
    print(f"  {len(v0237_samples)} samples; first keys: {list(v0237_samples[0].keys())[:8]}")

    files = sorted(glob.glob(JIWON_E2E_FRAMES_GLOB))
    print(f"jiwon_e2e MultiAgnet pkl candidates: {len(files)}")
    # pick a frame with non-zero ego speed AND with objects
    chosen = None
    for fp in files:
        d = load_jiwon_frame(fp)
        if d["ego_pose"][3] > 1.0 and len(d["objects"]) > 0:
            chosen = (fp, d)
            break
    if chosen is None:
        # fallback: just first frame
        chosen = (files[0], load_jiwon_frame(files[0]))
    fp, jw_frame = chosen
    print(f"chosen frame: {os.path.basename(fp)}")
    print(f"  ego_pose: {jw_frame['ego_pose']}")
    print(f"  num objects: {len(jw_frame['objects'])}")

    print("\ninitializing MPC...")
    t0 = time.time()
    mpc = MPC_Class()
    print(f"  init: {time.time()-t0:.1f}s; N={mpc.N_MPC}, Ts={mpc.Ts_MPC}")

    # -------- Step 1: jiwon ego + jiwon own trajectory, no objects --------
    wp_abs = wp_rel_to_xy_v_yaw(jw_frame["wp_rel_gt"])  # (30, 4) [x,y,yaw,v]
    # need reference_map polyline (x,y) — use trajectory itself
    ref_map = build_reference_map_from_path(wp_abs[:, :2])
    road_map = generate_road_map(ref_map)

    vs = build_vehicle_state_from_jiwon(road_map, jw_frame["ego_pose"])
    ms = build_measured_state(jw_frame["ego_pose"])

    point = build_point_from_xy_path(
        road_map,
        np.column_stack([wp_abs[:, 0], wp_abs[:, 1], wp_abs[:, 2], wp_abs[:, 3]]),
        mpc.N_MPC, mpc.Ts_MPC, ego_s=vs[0, 4],
    )

    objects_empty = np.zeros((1, mpc.N_MPC + 1, 11))
    objects_empty[:, :, 8] = -1e6  # s = -inf so MPC ignores them

    # -------- Step 0: v0237 self-consistent (v0237 ego + v0237 traj) --------
    s0 = v0237_samples[0]
    traj0_local = s0["trajectory_points_local"]  # (40, 3) [dx, dy, vel]
    dxy0 = np.diff(traj0_local[:, :2], axis=0)
    yaws0 = np.concatenate([[0.0], np.arctan2(dxy0[:, 1], dxy0[:, 0])])
    xy_speed_self = np.column_stack([
        traj0_local[:, 0], traj0_local[:, 1], yaws0, traj0_local[:, 2],
    ])
    ref_self = build_reference_map_from_path(xy_speed_self[:, :2])
    road_self = generate_road_map(ref_self)
    # v0237 ego state — pose 는 origin (이미 ego-local), speed from ego_velocity[0]
    v0237_ego_pose = np.zeros(6)
    v0237_ego_pose[3] = float(s0["ego_velocity"][0])  # vx
    v0237_ego_pose[4] = float(s0["ego_velocity"][1])  # vy
    v0237_ego_pose[5] = float(s0["ego_yaw_rate"])      # wz
    vs_self = build_vehicle_state_from_jiwon(road_self, v0237_ego_pose)
    ms_self = build_measured_state(v0237_ego_pose, tire_angle_rad=float(s0["ego_tire_angle"]))
    # measured_state ax, ay 채우기
    ms_self[0, 5] = float(s0["ego_acceleration"][0])
    ms_self[0, 6] = float(s0["ego_acceleration"][1])
    point_self = build_point_from_xy_path(
        road_self, xy_speed_self, mpc.N_MPC, mpc.Ts_MPC, ego_s=vs_self[0, 4],
    )
    run_step("STEP 0: v0237 self-consistent (v0237 ego + v0237 traj)",
             mpc, road_self, vs_self, objects_empty, point_self, ms_self)

    run_step("STEP 1: jiwon ego + jiwon traj, NO lead car", mpc, road_map, vs, objects_empty, point, ms)

    # -------- Step 2: jiwon ego + v0237 trajectory, no objects --------
    s0 = v0237_samples[0]
    v0237_traj_local = s0["trajectory_points_local"]  # (40, 3) [dx, dy, vel]
    # synthesize yaw from successive points
    dxy = np.diff(v0237_traj_local[:, :2], axis=0)
    yaws = np.concatenate([[0.0], np.arctan2(dxy[:, 1], dxy[:, 0])])
    v0237_xy_speed = np.column_stack([
        v0237_traj_local[:, 0], v0237_traj_local[:, 1], yaws, v0237_traj_local[:, 2],
    ])
    ref_map_v = build_reference_map_from_path(v0237_xy_speed[:, :2])
    road_map_v = generate_road_map(ref_map_v)
    vs_v = build_vehicle_state_from_jiwon(road_map_v, jw_frame["ego_pose"])
    point_v = build_point_from_xy_path(
        road_map_v, v0237_xy_speed, mpc.N_MPC, mpc.Ts_MPC, ego_s=vs_v[0, 4],
    )
    run_step("STEP 2: jiwon ego + v0237 traj, NO lead car", mpc, road_map_v, vs_v, objects_empty, point_v, ms)

    # -------- Step 3: jiwon ego + v0237 traj + jiwon lead car (forward only) --------
    # pick lead car directly ahead (x > 0, |y| < 5)
    lead = None
    for o in jw_frame["objects"]:
        x_m, y_m, L, W = bev_obj_to_egolocal(o["bbox"])
        if x_m > 2.0 and abs(y_m) < 5.0:
            lead = (x_m, y_m, L, W, o.get("velocity", np.array([0.0, 0.0])))
            break
    if lead is None:
        # take first object regardless
        o = jw_frame["objects"][0]
        x_m, y_m, L, W = bev_obj_to_egolocal(o["bbox"])
        lead = (x_m, y_m, L, W, o.get("velocity", np.array([0.0, 0.0])))
        print(f"\n[note] no straight-ahead lead car; using first object at ({x_m:.1f}, {y_m:.1f})")
    print(f"\nlead car ego-local: x={lead[0]:.2f} y={lead[1]:.2f} L={lead[2]:.2f} W={lead[3]:.2f} v={lead[4]}")

    # build object trajectory (constant velocity assumption) over N+1 steps, Ts=0.05
    obj = np.zeros((1, mpc.N_MPC + 1, 11))
    vx_obj, vy_obj = float(lead[4][0]), float(lead[4][1])
    for k in range(mpc.N_MPC + 1):
        t_k = k * mpc.Ts_MPC
        obj[0, k, 0] = lead[0] + vx_obj * t_k
        obj[0, k, 1] = lead[1] + vy_obj * t_k
        obj[0, k, 2] = 0.0
        obj[0, k, 3] = 1.0       # id
        obj[0, k, 4] = 1.0       # valid
        obj[0, k, 5] = 1.5       # height (default)
        obj[0, k, 6] = float(lead[2])  # length
        obj[0, k, 7] = float(lead[3])  # width
    # Frenet 변환 (s, n)
    for k in range(mpc.N_MPC + 1):
        sn_in = obj[0, k:k+1, :7].copy()
        sn_in[0, 0] = obj[0, k, 0]
        sn_in[0, 1] = obj[0, k, 1]
        sn_in[0, 2] = obj[0, k, 2]
        sn_in[0, 3] = 0.0  # speed not needed
        try:
            sn_out = frenet_transform(road_map_v, sn_in)
            sn_out = frenet_mu_transform(road_map_v, sn_out)
            obj[0, k, 8] = sn_out[0, 4] if sn_out.shape[1] > 4 else 0.0
            obj[0, k, 9] = sn_out[0, 5] if sn_out.shape[1] > 5 else 0.0
            obj[0, k, 10] = sn_out[0, 6] if sn_out.shape[1] > 6 else 0.0
        except Exception as e:
            print(f"frenet transform failed for obj step {k}: {e}")
            obj[0, k, 8] = -1e6
            break

    run_step("STEP 3: jiwon ego + v0237 traj + lead car (constant vel)", mpc, road_map_v, vs_v, obj, point_v, ms)


if __name__ == "__main__":
    main()

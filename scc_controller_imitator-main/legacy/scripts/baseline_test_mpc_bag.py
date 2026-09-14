"""MPC.bag 의 실제 ROS 입력으로 MPC.calculate_optimal 동작 확인.

ROS node 의 변환 흐름을 그대로 따라간다:
  1. vs_global → vs_local (origin shifted, yaw rotated to 0)
  2. traj_global → traj_local (vehicle frame)
  3. reference_map = traj_local 의 (x, y)
  4. generate_road_map(reference_map)
  5. frenet_transform / frenet_mu_transform 으로 vs 와 trajectory 보강
  6. zero objects (lead car 없는 bag)
  7. MPC.calculate_optimal 호출
  8. NaN 안 나면 baseline 확립 → 우리 v0237 입력과 비교 가능
"""
from __future__ import annotations
import os, sys, time, numpy as np

SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "external", "acados_scc", "scripts",
)
sys.path.insert(0, "/home/ailab-12/git/A1_Rule/devel/lib/python3/dist-packages")
sys.path.insert(0, SCRIPTS_DIR)
BUILD_CACHE = os.path.join(os.path.dirname(SCRIPTS_DIR), "build_cache")
os.makedirs(BUILD_CACHE, exist_ok=True)
os.chdir(BUILD_CACHE)

from MPC.NMPC_class_delayed_kinematic_cat import (
    Nonlinear_Delayed_Kinematic_Cat_Model_Predictive_Controller as MPC_Class,
)
from utils.trajectory_process import (
    generate_road_map,
    frenet_transform,
    frenet_mu_transform,
    global_to_local,
)

BASELINE_NPZ = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/mpc_bag_baseline.npz"


def load_samples(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    by_idx = {}
    for k in z.files:
        i_str, key = k.split("__", 1)
        idx = int(i_str[1:])
        by_idx.setdefault(idx, {})[key] = z[k]
    return [by_idx[i] for i in sorted(by_idx)]


def build_mpc_inputs(s, N, Ts):
    """ROS node 변환 흐름 그대로."""
    vs_g = s["vs_global"]            # (1, 4) [x, y, yaw, vx]
    ms_g = s["ms_global"]            # (1, 8) [x, y, yaw, vx, vy, ax, ay, tire_angle]
    traj_g = s["traj_global"]        # (M, 4) [x, y, yaw, speed]

    # 1. vs_local
    vehicle_x = vs_g[0, 0]
    vehicle_y = vs_g[0, 1]
    vehicle_yaw = vs_g[0, 2]
    R = np.array([[np.cos(vehicle_yaw), -np.sin(vehicle_yaw)],
                  [np.sin(vehicle_yaw),  np.cos(vehicle_yaw)]]).T

    # 2. traj_local + reference_map (sharing trajectory.point xy)
    M = len(traj_g)
    point_local = np.zeros((M, 4))
    refmap = np.zeros((M, 5))
    for i in range(M):
        gp = np.array([traj_g[i, 0], traj_g[i, 1]])
        lp = global_to_local(gp, vs_g, R)
        point_local[i, 0] = lp[0]
        point_local[i, 1] = lp[1]
        point_local[i, 2] = traj_g[i, 2] - vehicle_yaw
        point_local[i, 3] = traj_g[i, 3]
        refmap[i, 0] = lp[0]
        refmap[i, 1] = lp[1]

    # 4. road_map
    road_map = generate_road_map(refmap)

    # vs_local at origin
    vs_local = np.zeros((1, 7))
    vs_local[0, 3] = float(vs_g[0, 3])  # speed
    vs_local = frenet_transform(road_map, vs_local)
    vs_local = frenet_mu_transform(road_map, vs_local)

    # trajectory_sn — ROS code calls frenet on point_local (local frame)
    trajectory_sn = frenet_transform(road_map, point_local)
    trajectory_sn = frenet_mu_transform(road_map, trajectory_sn, start_mu=vs_local[0, 6])

    # measured_state: x, y, yaw are global in ROS code (kept untouched after npize),
    # but acados_scc_control_node.py uses raw vs_msg.x/y/yaw which are global.
    # The MPC internally treats them as 'measurement' for delay estimation; just pass through.
    measured = ms_g.copy()

    # zero objects (no lead car in this bag)
    objects = np.zeros((1, N + 1, 11))
    objects[:, :, 8] = -1e6  # s = -inf so ignored

    return road_map, vs_local, objects, trajectory_sn, measured, point_local, refmap


def try_one(label, mpc, s, idx):
    N, Ts = mpc.N_MPC, mpc.Ts_MPC
    road_map, vs, objs, traj_sn, ms, point_local, refmap = build_mpc_inputs(s, N, Ts)
    print(f"\n========== {label} (sample {idx}) ==========")
    print(f"  vs_global x={s['vs_global'][0,0]:.1f} y={s['vs_global'][0,1]:.1f} yaw={s['vs_global'][0,2]:.3f} speed={s['vs_global'][0,3]:.2f}")
    print(f"  ms_global ax={s['ms_global'][0,5]:.3f} ay={s['ms_global'][0,6]:.3f} tire={s['ms_global'][0,7]:.4f}")
    print(f"  traj_local first 3:\n{point_local[:3]}")
    print(f"  traj_local last 3:\n{point_local[-3:]}")
    print(f"  vs after frenet [s,n,mu]: [{vs[0,4]:.4f}, {vs[0,5]:.4f}, {vs[0,6]:.4f}]")
    print(f"  trajectory_sn shape: {traj_sn.shape}, s range: [{traj_sn[:,4].min():.2f}, {traj_sn[:,4].max():.2f}]")
    print(f"  trajectory_sn speed range: [{traj_sn[:,3].min():.2f}, {traj_sn[:,3].max():.2f}]")
    print(f"  trajectory_sn n range: [{traj_sn[:,5].min():.3f}, {traj_sn[:,5].max():.3f}]")
    print(f"  trajectory_sn mu range: [{traj_sn[:,6].min():.3f}, {traj_sn[:,6].max():.3f}]")

    mpc.is_MPC_init = True
    t0 = time.time()
    out = mpc.calculate_optimal(road_map, vs, objs, traj_sn, ms)
    dt = time.time() - t0
    target_cart, target_pred, target_input, stats, control_io = out
    print(f"  solve time: {dt*1000:.1f} ms")
    print(f"  target_input [steer_deg, torque, accel]: {target_input}")
    if control_io is None:
        print(f"  FAILED — fallback (control_io=None)")
        try:
            print(f"  acados status: {mpc.acados_solver.get_status()}")
        except Exception:
            pass
        return False
    print(f"  control_io steer [:5]: {np.round(np.asarray(control_io['steer'][:5]), 3)}")
    print(f"  control_io accel [:5]: {np.round(np.asarray(control_io['acceleration'][:5]), 3)}")
    # compare with ground truth (control_io msg shape varies — be defensive)
    gt_steer = s.get("gt_steer")
    gt_accel = s.get("gt_accel")
    if gt_steer is not None and gt_steer.size > 0:
        gs0 = float(np.asarray(gt_steer).flatten()[0])
        os0 = float(np.asarray(control_io['steer']).flatten()[0])
        print(f"  GT steer[0]: {gs0:.3f}  vs ours[0]: {os0:.3f}")
    if gt_accel is not None and gt_accel.size > 0:
        ga0 = float(np.asarray(gt_accel).flatten()[0])
        oa0 = float(np.asarray(control_io['acceleration']).flatten()[0])
        print(f"  GT accel[0]: {ga0:.3f}  vs ours[0]: {oa0:.3f}")
    return True


def main():
    print(f"loading {BASELINE_NPZ}")
    samples = load_samples(BASELINE_NPZ)
    print(f"  {len(samples)} samples")

    print("\nMPC init...")
    t0 = time.time()
    mpc = MPC_Class()
    print(f"  init: {time.time()-t0:.1f} s; N={mpc.N_MPC}, Ts={mpc.Ts_MPC}")

    ok_count = 0
    for i, s in enumerate(samples):
        ok = try_one(f"baseline sample {i} (vx={s['vs_global'][0,3]:.1f} m/s)", mpc, s, i)
        if ok:
            ok_count += 1
    print(f"\n=== summary ===")
    print(f"  {ok_count}/{len(samples)} samples converged (non-fallback)")


if __name__ == "__main__":
    main()

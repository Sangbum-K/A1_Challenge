"""원인 격리 — baseline 정상 + v0237 fail. 어느 입력 요소가 NaN 유발하는지 격리.

3가지 hybrid 케이스로 격리:
  H1. baseline ms + baseline trajectory  → baseline (control)
  H2. baseline ms + v0237 trajectory      → trajectory 원인 여부 격리
  H3. v0237 ms    + baseline trajectory   → ms 원인 여부 격리
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
os.chdir(BUILD_CACHE)

from MPC.NMPC_class_delayed_kinematic_cat import (
    Nonlinear_Delayed_Kinematic_Cat_Model_Predictive_Controller as MPC_Class,
)
from utils.trajectory_process import (
    generate_road_map, frenet_transform, frenet_mu_transform, global_to_local,
)

BASELINE_NPZ = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/mpc_bag_baseline.npz"
V0237_NPZ = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/v0237_sample10.npz"


def load_v0237_sample(i=0):
    z = np.load(V0237_NPZ)
    return {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(f"s{i}__")}


def load_baseline_sample(i=0):
    z = np.load(BASELINE_NPZ, allow_pickle=True)
    return {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(f"s{i}__")}


def baseline_inputs(s, N, Ts):
    """Build MPC inputs from baseline (ROS bag) sample."""
    vs_g = s["vs_global"]
    ms_g = s["ms_global"].copy()
    traj_g = s["traj_global"]
    vehicle_yaw = vs_g[0, 2]
    R = np.array([[np.cos(vehicle_yaw), -np.sin(vehicle_yaw)],
                  [np.sin(vehicle_yaw),  np.cos(vehicle_yaw)]]).T
    M = len(traj_g)
    point_local = np.zeros((M, 4))
    refmap = np.zeros((M, 5))
    for i in range(M):
        gp = np.array([traj_g[i, 0], traj_g[i, 1]])
        lp = global_to_local(gp, vs_g, R)
        point_local[i, 0] = lp[0]; point_local[i, 1] = lp[1]
        point_local[i, 2] = traj_g[i, 2] - vehicle_yaw
        point_local[i, 3] = traj_g[i, 3]
        refmap[i, 0] = lp[0]; refmap[i, 1] = lp[1]
    road_map = generate_road_map(refmap)
    vs_local = np.zeros((1, 7))
    vs_local[0, 3] = float(vs_g[0, 3])
    vs_local = frenet_transform(road_map, vs_local)
    vs_local = frenet_mu_transform(road_map, vs_local)
    traj_sn = frenet_transform(road_map, point_local)
    traj_sn = frenet_mu_transform(road_map, traj_sn, start_mu=vs_local[0, 6])
    objs = np.zeros((1, N + 1, 11)); objs[:, :, 8] = -1e6
    return road_map, vs_local, objs, traj_sn, ms_g


def v0237_inputs(s, N, Ts):
    """Build MPC inputs from v0237 sample (already in ego-local frame)."""
    traj = s["trajectory_points_local"]  # (40, 3) [dx, dy, vel]
    # yaws from atan2 of diffs
    dxy = np.diff(traj[:, :2], axis=0)
    yaws = np.concatenate([[0.0], np.arctan2(dxy[:, 1], dxy[:, 0])])
    point_local = np.column_stack([traj[:, 0], traj[:, 1], yaws, traj[:, 2]])
    # extend reference_map forward
    dx_last = traj[-1, 0] - traj[-2, 0]
    dy_last = traj[-1, 1] - traj[-2, 1]
    seg = max(np.hypot(dx_last, dy_last), 0.1)
    dirn = np.array([dx_last, dy_last]) / seg
    extra = np.array([traj[-1, :2] + dirn * (i + 1) * seg for i in range(60)])
    refmap_xy = np.vstack([traj[:, :2], extra])
    refmap = np.zeros((len(refmap_xy), 5))
    refmap[:, 0] = refmap_xy[:, 0]; refmap[:, 1] = refmap_xy[:, 1]
    road_map = generate_road_map(refmap)
    vs_local = np.zeros((1, 7))
    vs_local[0, 3] = max(float(s["ego_velocity"][0]), 0.01)
    vs_local = frenet_transform(road_map, vs_local)
    vs_local = frenet_mu_transform(road_map, vs_local)
    traj_sn = frenet_transform(road_map, point_local)
    traj_sn = frenet_mu_transform(road_map, traj_sn, start_mu=vs_local[0, 6])
    ms = np.zeros((1, 8))
    ms[0, 3] = float(s["ego_velocity"][0])
    ms[0, 4] = float(s["ego_velocity"][1])
    ms[0, 5] = float(s["ego_acceleration"][0])
    ms[0, 6] = float(s["ego_acceleration"][1])
    ms[0, 7] = float(s["ego_tire_angle"])
    objs = np.zeros((1, N + 1, 11)); objs[:, :, 8] = -1e6
    return road_map, vs_local, objs, traj_sn, ms


def run(label, mpc, road_map, vs, objs, traj_sn, ms):
    print(f"\n========== {label} ==========")
    print(f"  vs_local speed={vs[0,3]:.2f} s={vs[0,4]:.3f} n={vs[0,5]:.3f} mu={vs[0,6]:.3f}")
    print(f"  ms vx={ms[0,3]:.3f} vy={ms[0,4]:.3f} ax={ms[0,5]:.3f} ay={ms[0,6]:.3f} tire={ms[0,7]:.4f}")
    print(f"  traj_sn pts={len(traj_sn)} s=[{traj_sn[:,4].min():.2f},{traj_sn[:,4].max():.2f}] "
          f"speed=[{traj_sn[:,3].min():.2f},{traj_sn[:,3].max():.2f}] "
          f"n=[{traj_sn[:,5].min():.3f},{traj_sn[:,5].max():.3f}] "
          f"mu=[{traj_sn[:,6].min():.3f},{traj_sn[:,6].max():.3f}]")
    mpc.is_MPC_init = True
    out = mpc.calculate_optimal(road_map, vs, objs, traj_sn, ms)
    target_cart, target_pred, target_input, stats, control_io = out
    print(f"  target_input: {target_input}")
    status = mpc.acados_solver.get_status()
    print(f"  acados status: {status}, sqp_iter: {mpc.acados_solver.get_stats('sqp_iter')}")
    return status == 0 or (control_io is not None and np.abs(np.asarray(control_io['steer'])).max() > 1e-3)


def main():
    print("loading samples...")
    s_base = load_baseline_sample(1)   # vx=13.6 m/s
    s_v0237 = load_v0237_sample(0)     # vx=4.8 m/s
    print(f"  baseline[1] vs=({s_base['vs_global'][0,3]:.2f} m/s)  ms={s_base['ms_global'].flatten()}")
    print(f"  v0237[0]    vx={s_v0237['ego_velocity'][0]:.2f} ax={s_v0237['ego_acceleration'][0]:.3f}")

    print("\ninitializing MPC...")
    t0 = time.time()
    mpc = MPC_Class()
    print(f"  init: {time.time()-t0:.1f}s")
    N, Ts = mpc.N_MPC, mpc.Ts_MPC

    # H1: baseline ms + baseline trajectory  (sanity control)
    rm_b, vs_b, objs_b, traj_b, ms_b = baseline_inputs(s_base, N, Ts)
    ok1 = run("H1: baseline ms + baseline traj (CONTROL)", mpc, rm_b, vs_b, objs_b, traj_b, ms_b)

    # H2: baseline ms + v0237 trajectory   (trajectory 격리)
    rm_v, vs_v, objs_v, traj_v, ms_v = v0237_inputs(s_v0237, N, Ts)
    # use baseline ms — but baseline ms is in global frame; that's a problem.
    # Actually ms is "measurement" with vx, vy, ax, ay etc.; the global x,y don't matter for cost.
    # The dynamic model uses vx, vy, ax, ay, tire_angle. Use baseline ms_b but with v0237's vs_v.
    print("\n--- H2: use baseline ms scalars (ax/ay/tire) + v0237 trajectory + v0237 vs ---")
    ms_blend = ms_v.copy()
    ms_blend[0, 5] = ms_b[0, 5]  # ax
    ms_blend[0, 6] = ms_b[0, 6]  # ay
    ms_blend[0, 7] = ms_b[0, 7]  # tire_angle
    ok2 = run("H2: ax/ay/tire from baseline, traj+vs from v0237", mpc, rm_v, vs_v, objs_v, traj_v, ms_blend)

    # H3: v0237 ms + baseline trajectory
    print("\n--- H3: v0237 ms scalars (ax/ay/tire) + baseline trajectory + baseline vs ---")
    ms_blend2 = ms_b.copy()
    ms_blend2[0, 5] = ms_v[0, 5]
    ms_blend2[0, 6] = ms_v[0, 6]
    ms_blend2[0, 7] = ms_v[0, 7]
    ok3 = run("H3: ax/ay/tire from v0237, traj+vs from baseline", mpc, rm_b, vs_b, objs_b, traj_b, ms_blend2)

    print("\n=== summary ===")
    print(f"  H1 baseline control: {'OK' if ok1 else 'FAIL'}")
    print(f"  H2 baseline ms + v0237 traj/vs: {'OK' if ok2 else 'FAIL'}")
    print(f"  H3 v0237 ms + baseline traj/vs: {'OK' if ok3 else 'FAIL'}")
    if not ok2 and ok3:
        print("\n  → Root cause: v0237 trajectory or vs (not ms).")
    elif ok2 and not ok3:
        print("\n  → Root cause: v0237 ms (ax/ay/tire).")
    elif not ok2 and not ok3:
        print("\n  → Both axes contribute. Likely interaction (e.g., short trajectory + small speed).")
    else:
        print("\n  → Neither isolated; check setup.")


if __name__ == "__main__":
    main()

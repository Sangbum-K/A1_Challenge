"""print_level=2 디버그 — Step 0 (v0237 self-consistent) 만 실행.

목적:
  1. acados 가 어디서 NaN 을 만드는지 식별
  2. MPC 입력 (vehicle_state / measured_state / point / objects) 의 모든 값 dump
  3. constraint bound 위반 여부 확인
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
    generate_road_map, frenet_transform, frenet_mu_transform, resample_points,
)

V0237_NPZ = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/v0237_sample10.npz"


def main():
    print("loading v0237 sample 0 ...")
    z = np.load(V0237_NPZ)
    s = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith("s0__")}
    print(f"  keys: {list(s.keys())[:10]}")
    print(f"  ego_velocity = {s['ego_velocity']}  (vx, vy)")
    print(f"  ego_acceleration = {s['ego_acceleration']}  (ax, ay)")
    print(f"  ego_yaw_rate = {s['ego_yaw_rate']}")
    print(f"  ego_tire_angle = {s['ego_tire_angle']}")
    traj = s["trajectory_points_local"]   # (40, 3) [dx, dy, vel]
    print(f"  trajectory_points_local[0:5] =\n{traj[:5]}")
    print(f"  trajectory_points_local speed range: [{traj[:,2].min():.3f}, {traj[:,2].max():.3f}]")

    print("\n=== initializing MPC (will print build messages) ===")
    mpc = MPC_Class()
    N, Ts = mpc.N_MPC, mpc.Ts_MPC
    print(f"  N={N}, Ts={Ts}, n_states={mpc.n_states}, n_inputs={mpc.n_inputs}")

    # Build inputs
    dxy = np.diff(traj[:, :2], axis=0)
    yaws = np.concatenate([[0.0], np.arctan2(dxy[:, 1], dxy[:, 0])])
    xy_speed = np.column_stack([traj[:, 0], traj[:, 1], yaws, traj[:, 2]])

    # reference_map = trajectory + forward extension
    dx_last = xy_speed[-1, 0] - xy_speed[-2, 0]
    dy_last = xy_speed[-1, 1] - xy_speed[-2, 1]
    seg = max(np.hypot(dx_last, dy_last), 0.1)
    dirn = np.array([dx_last, dy_last]) / seg
    extra = np.array([xy_speed[-1, :2] + dirn * (i + 1) * seg for i in range(60)])
    ref_xy = np.vstack([xy_speed[:, :2], extra])
    ref_map = np.zeros((len(ref_xy), 5))
    ref_map[:, 0] = ref_xy[:, 0]
    ref_map[:, 1] = ref_xy[:, 1]
    print(f"\n  reference_map shape: {ref_map.shape}")
    print(f"  ref_map first 3: {ref_map[:3, :2]}")
    print(f"  ref_map last 3: {ref_map[-3:, :2]}")

    road_map = generate_road_map(ref_map)

    # vehicle_state at origin, Frenet
    vs = np.zeros((1, 7))
    vs[0, 3] = max(float(s["ego_velocity"][0]), 0.01)   # speed
    vs = frenet_transform(road_map, vs)
    vs = frenet_mu_transform(road_map, vs, start_mu=0.0)
    print(f"\n  vehicle_state (1,7) after frenet:\n{vs}")
    print(f"    s={vs[0,4]:.4f}, n={vs[0,5]:.4f}, mu={vs[0,6]:.4f}")

    # measured_state — full from v0237
    ms = np.zeros((1, 8))
    ms[0, 3] = float(s["ego_velocity"][0])
    ms[0, 4] = float(s["ego_velocity"][1])
    ms[0, 5] = float(s["ego_acceleration"][0])
    ms[0, 6] = float(s["ego_acceleration"][1])
    # ms[0, 7] = float(s["ego_tire_angle"])  # ORIG: -0.067 rad
    ms[0, 7] = 0.0  # DEBUG: force tire_angle=0 (baseline always near 0)
    print(f"  measured_state (1,8): {ms}")
    print(f"    vx={ms[0,3]:.4f}, vy={ms[0,4]:.4f}, ax={ms[0,5]:.4f}, ay={ms[0,6]:.4f}, tire={ms[0,7]:.4f} rad")

    # point — match ROS node: use trajectory_sn (not resampled) + start_mu from vehicle
    # 단, trajectory 길이가 40 이므로 mpc 가 보간하도록 충분히 정보 제공
    point = frenet_transform(road_map, xy_speed)
    point = frenet_mu_transform(road_map, point, start_mu=vs[0, 6])
    print(f"\n  point shape: {point.shape}, columns: [x, y, yaw, speed, s, n, mu]")
    print(f"  point first 5 rows:\n{point[:5]}")
    print(f"  point speed range: [{point[:,3].min():.3f}, {point[:,3].max():.3f}]")
    print(f"  point s range: [{point[:,4].min():.3f}, {point[:,4].max():.3f}]")
    print(f"  point n range: [{point[:,5].min():.3f}, {point[:,5].max():.3f}]")
    print(f"  point mu range: [{point[:,6].min():.3f}, {point[:,6].max():.3f}]")

    # objects: none
    objs = np.zeros((1, N + 1, 11))
    objs[:, :, 8] = -1e6

    print("\n=== calling calculate_optimal (print_level=2 → expect per-iter KKT) ===")
    t0 = time.time()
    out = mpc.calculate_optimal(road_map, vs, objs, point, ms)
    print(f"\nsolve time: {(time.time()-t0)*1000:.1f} ms")
    target_cart, target_pred, target_input, stats, control_io = out
    print(f"target_input: {target_input}")
    print(f"control_io: {'None (fallback)' if control_io is None else 'OK'}")

    # Inspect acados solver status directly via interface
    try:
        # the AcadosOcpSolver instance is mpc.acados_solver
        status_codes = {
            0: "SUCCESS",
            1: "FAILURE",
            2: "MAX_ITER",
            3: "MIN_STEP",
            4: "QP_FAILURE",
            5: "READY",
        }
        s_code = mpc.acados_solver.get_status()
        print(f"\n  acados final status: {s_code} ({status_codes.get(s_code, 'unknown')})")
        sqp_iter = mpc.acados_solver.get_stats('sqp_iter')
        qp_iter = mpc.acados_solver.get_stats('qp_iter')
        time_tot = mpc.acados_solver.get_stats('time_tot')
        print(f"  sqp_iter: {sqp_iter}")
        print(f"  qp_iter:  {qp_iter}")
        print(f"  time_tot: {time_tot*1000:.2f} ms")
    except Exception as e:
        print(f"  acados stats unavailable: {e}")


if __name__ == "__main__":
    main()

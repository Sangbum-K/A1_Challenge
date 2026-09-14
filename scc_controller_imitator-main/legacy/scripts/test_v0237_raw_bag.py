"""v0237 raw bag 추출 sample 로 MPC 직접 호출.

만약 정상 동작:
  - v0237 pkl 의 preprocessing 에서 어떤 변형이 NaN 만든 것 (가능성 높음)
만약 fail:
  - 우리 코드의 처리 방식 자체 문제 (vehicle_state 처리 등)
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

NPZ = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/v0237_raw_bag_sample10.npz"


def load_samples(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    by = {}
    for k in z.files:
        i_str, key = k.split("__", 1)
        i = int(i_str[1:])
        by.setdefault(i, {})[key] = z[k]
    return [by[i] for i in sorted(by)]


def build_mpc_inputs(s, N, Ts):
    """Same flow as baseline_test_mpc_bag.py: global → local + frenet."""
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
    objs = np.zeros((1, N + 1, 11))
    objs[:, :, 8] = -1e6
    return road_map, vs_local, objs, traj_sn, ms_g, point_local


def run_one(label, mpc, s):
    N = mpc.N_MPC
    rm, vs, objs, traj_sn, ms, point_local = build_mpc_inputs(s, N, mpc.Ts_MPC)
    print(f"\n========== {label} ==========")
    print(f"  vs_global x={s['vs_global'][0,0]:.3f} y={s['vs_global'][0,1]:.3f} "
          f"yaw={s['vs_global'][0,2]:.4f} vx={s['vs_global'][0,3]:.2f}")
    print(f"  ms ax={s['ms_global'][0,5]:.3f} ay={s['ms_global'][0,6]:.3f} tire={s['ms_global'][0,7]:.4f}")
    print(f"  traj_local first 3:\n{point_local[:3]}")
    print(f"  traj_local last 3:\n{point_local[-3:]}")
    print(f"  vs after frenet [s,n,mu]: [{vs[0,4]:.4f}, {vs[0,5]:.4f}, {vs[0,6]:.4f}]")
    print(f"  traj_sn s=[{traj_sn[:,4].min():.2f},{traj_sn[:,4].max():.2f}] "
          f"speed=[{traj_sn[:,3].min():.2f},{traj_sn[:,3].max():.2f}] "
          f"mu=[{traj_sn[:,6].min():.3f},{traj_sn[:,6].max():.3f}]")
    mpc.is_MPC_init = True
    t0 = time.time()
    out = mpc.calculate_optimal(rm, vs, objs, traj_sn, ms)
    dt = time.time() - t0
    target_cart, target_pred, target_input, stats, control_io = out
    status = mpc.acados_solver.get_status()
    print(f"  solve {dt*1000:.0f} ms, status={status}, target_input={target_input}")
    if control_io is None:
        print(f"  → FAIL fallback")
        return False
    print(f"  control_io steer[:5]: {np.round(np.asarray(control_io['steer'][:5]), 3)}")
    print(f"  control_io accel[:5]: {np.round(np.asarray(control_io['acceleration'][:5]), 3)}")
    gt_steer = s.get("gt_steer")
    gt_accel = s.get("gt_acceleration")
    if gt_steer is not None and gt_steer.size > 0:
        print(f"  GT_steer[:5]={np.round(gt_steer[:5], 3)}")
    if gt_accel is not None and gt_accel.size > 0:
        print(f"  GT_accel[:5]={np.round(gt_accel[:5], 3)}")
    nonzero = (np.abs(control_io['steer']) > 1e-6).any() or (np.abs(control_io['acceleration']) > 1e-6).any()
    return nonzero


def main():
    samples = load_samples(NPZ)
    print(f"loaded {len(samples)} samples")
    t0 = time.time()
    mpc = MPC_Class()
    print(f"\nMPC init: {time.time()-t0:.1f}s; N={mpc.N_MPC}, Ts={mpc.Ts_MPC}")
    ok = 0
    for i, s in enumerate(samples):
        if run_one(f"v0237 raw bag sample {i} (vx={s['vs_global'][0,3]:.1f})", mpc, s):
            ok += 1
    print(f"\n=== summary === {ok}/{len(samples)} OK")


if __name__ == "__main__":
    main()

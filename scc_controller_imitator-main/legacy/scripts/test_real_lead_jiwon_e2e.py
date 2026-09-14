"""jiwon_e2e MultiAgnet 의 실차 ego + trajectory + lead car 로 MPC 호출.

모든 입력이 같은 frame (시간/공간 정합) 에서 추출 → MPC 가 reasonable 한 SCC 출력 만들 것.

jiwon_e2e pkl 형식:
  ego_pose: [yaw, pitch, roll, vx, vy, wz]
  wp_rel_gt: (30, 4) [Δx, Δy, Δyaw, v]   ego-local
  objects: [{bbox=(cx,cy,w_cells,h_cells), velocity=(vx,vy), ...}]
            bbox 는 BEV cell. real ego-local: (-20+cx*0.5, -25+cy*0.5)
            BEV grid: H=200 (x forward -20~80m), W=100 (y left -25~25m), cell=0.5m

변환:
  vehicle_state ← (x=0, y=0, yaw=0, vx, vy, ...)
  trajectory.point ← cumsum(wp_rel_gt) → (x, y, yaw, speed) in ego-local
  PredictObjects ← lead car 시계열 (등속 가정, 40 step × 0.1s)
"""
from __future__ import annotations
import os, sys, time, pickle, glob
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
    generate_road_map, frenet_transform, frenet_mu_transform,
    global_to_local, resample_points,
)
from autohyu_msgs.msg import (
    VehicleState, Trajectory, TrajectoryPoint,
    PredictObjects, PredictObject, Object3DState, ObjectDimension,
)
from std_msgs.msg import Header

PKL_GLOB = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/data/processed/frames/sangam_track__MultiAgnet__*.pkl"

# BEV grid params (jiwon_e2e default.yaml)
BEV_X_MIN, BEV_Y_MIN, BEV_CELL = -20.0, -25.0, 0.5


def bev_bbox_to_ego_local(bbox):
    """bbox (cx, cy, w_cells, h_cells) → (x_m, y_m, length_m, width_m) in ego-local."""
    cx, cy, w_cells, h_cells = bbox
    x_m = BEV_X_MIN + cx * BEV_CELL
    y_m = BEV_Y_MIN + cy * BEV_CELL
    length_m = h_cells * BEV_CELL   # h cell axis is x-forward
    width_m = w_cells * BEV_CELL
    return x_m, y_m, length_m, width_m


def make_vs_msg(ego_pose, ax=0.0, ay=0.0, tire_angle=0.0):
    """ego_pose=[yaw,pitch,roll,vx,vy,wz] (jiwon_e2e SI) → VehicleState msg (ego-local frame, x=y=yaw=0)."""
    vs = VehicleState()
    vs.x = 0.0
    vs.y = 0.0
    vs.yaw = 0.0
    vs.vx = float(ego_pose[3])
    vs.vy = float(ego_pose[4])
    vs.vz = 0.0
    vs.ax = float(ax)
    vs.ay = float(ay)
    vs.az = 0.0
    vs.roll = float(ego_pose[2])
    vs.pitch = float(ego_pose[1])
    vs.yaw_vel = float(ego_pose[5])
    vs.vehicle_can.steering_tire_angle = float(tire_angle)
    return vs


def wp_rel_to_traj_msg(wp_rel_gt):
    """wp_rel_gt (30, 4) [Δx, Δy, Δyaw, v] → Trajectory msg (ego-local, prepend (0,0,0,vx_now))."""
    abs_xy = np.cumsum(wp_rel_gt[:, :2], axis=0)
    abs_yaw = np.cumsum(wp_rel_gt[:, 2], axis=0)
    abs_v = wp_rel_gt[:, 3]
    traj = Trajectory()
    # prepend ego origin point (0,0,0,v0) so MPC has the start
    p0 = TrajectoryPoint()
    p0.x = 0.0; p0.y = 0.0; p0.yaw = 0.0; p0.speed = float(abs_v[0])
    p0.acceleration = 0.0; p0.curvature = 0.0; p0.time = 0.0
    p0.distance = 0.0
    traj.point.append(p0)
    for i in range(len(abs_xy)):
        p = TrajectoryPoint()
        p.x = float(abs_xy[i, 0])
        p.y = float(abs_xy[i, 1])
        p.yaw = float(abs_yaw[i])
        p.speed = float(abs_v[i])
        p.acceleration = 0.0; p.curvature = 0.0
        p.time = (i + 1) * 0.1
        traj.point.append(p)
    return traj


def lead_to_objects_msg(lead_x, lead_y, lead_vx_global, lead_vy_global, length_m, width_m,
                       n_states=40, dt=0.1):
    """Lead car 정보를 PredictObjects msg 시계열로 (등속 가정, ego-local frame).

    ego_pose 의 yaw=0 (ego-local) 이므로 lead velocity 도 ego-local frame.
    """
    objs = PredictObjects()
    o = PredictObject()
    o.id = 1
    o.classification = 1
    o.dynamic_state = 0
    o.dimension = ObjectDimension(length=length_m, width=width_m, height=1.5)
    for i in range(n_states):
        s = Object3DState()
        t = i * dt
        s.x = lead_x + lead_vx_global * t
        s.y = lead_y + lead_vy_global * t
        s.z = 0.0
        s.yaw = 0.0
        s.vx = lead_vx_global
        s.vy = lead_vy_global
        s.vz = 0.0
        s.ax = 0.0; s.ay = 0.0; s.az = 0.0
        s.roll = 0.0; s.pitch = 0.0
        s.roll_rate = 0.0; s.pitch_rate = 0.0; s.yaw_rate = 0.0
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
    # 직접 nearest-trajectory frenet for objects
    cum_s = np.zeros(len(reference_map))
    if len(reference_map) > 1:
        dxy = np.diff(reference_map[:, :2], axis=0)
        cum_s[1:] = np.cumsum(np.hypot(dxy[:, 0], dxy[:, 1]))
    ref_xy = reference_map[:, :2]
    R = np.array([[np.cos(-vsg[0, 2]), -np.sin(-vsg[0, 2])], [np.sin(-vsg[0, 2]), np.cos(-vsg[0, 2])]])
    sn_objects = np.zeros((objects.shape[0], objects.shape[1], 11))
    for i in range(objects.shape[0]):
        if objects[i, :, 4].all():
            obj = objects[i, :, :]
            for j in range(obj.shape[0]):
                gp = np.array([obj[j, 0], obj[j, 1]])
                lp = global_to_local(gp, vsg, R)
                dxs = ref_xy[:, 0] - lp[0]
                dys = ref_xy[:, 1] - lp[1]
                dists = np.hypot(dxs, dys)
                k = int(np.argmin(dists))
                obj_s = cum_s[k]
                if k + 1 < len(ref_xy):
                    seg = ref_xy[k + 1] - ref_xy[k]
                else:
                    seg = ref_xy[k] - ref_xy[k - 1]
                seg_n = np.hypot(seg[0], seg[1])
                if seg_n > 1e-6:
                    seg = seg / seg_n
                    rel = np.array([lp[0] - ref_xy[k, 0], lp[1] - ref_xy[k, 1]])
                    obj_n = seg[0] * rel[1] - seg[1] * rel[0]
                else:
                    obj_n = dists[k]
                sn_objects[i, j, :2] = lp
                sn_objects[i, j, 2:8] = obj[j, 2:8]
                sn_objects[i, j, 8] = obj_s
                sn_objects[i, j, 9] = obj_n
                sn_objects[i, j, 10] = 0.0
        else:
            sn_objects[i, :, :] = 0
            sn_objects[i, :, 8] = -1e6
    return mpc.calculate_optimal(road_map, vehicle_state, sn_objects, trajectory_sn, measured_state), sn_objects


def repeat(mpc, vs_msg, traj_msg, objs_msg, n_iter=10, reset=False):
    """reset=True 면 cold start. False 면 warm start (이전 prev_X 사용)."""
    if reset:
        mpc.is_MPC_init = True
        mpc.pred_X = None; mpc.pred_U = None; mpc.X_init = None
        mpc.fallback_count = 0; mpc.last_mpc_status = None
    last_cio, last_of = None, None
    for k in range(n_iter):
        out, of = run_cycle(mpc, vs_msg, traj_msg, objs_msg)
        _, _, _, _, cio = out
        if cio is not None:
            # cio['steer'/'acceleration'] 가 self.pred_X view 일 수 있어 copy
            last_cio = {
                'steer': np.asarray(cio['steer'], dtype=np.float64).copy(),
                'acceleration': np.asarray(cio['acceleration'], dtype=np.float64).copy(),
            }
        last_of = of.copy() if of is not None else None
    return last_cio, last_of


def main():
    files = sorted(glob.glob(PKL_GLOB))
    print(f"jiwon_e2e MultiAgnet pkl: {len(files)}")
    # pick frames with non-zero ego speed AND at least one lead car in front
    picks = []
    for fp in files:
        with open(fp, 'rb') as f:
            d = pickle.load(f)
        vx = float(d["ego_pose"][3])
        if vx < 2.0:
            continue
        # 가장 가까운 전방 객체 (x>2, |y|<3 ego-local)
        front_lead = None
        best_x = 999
        for o in d["objects"]:
            x_m, y_m, L, W = bev_bbox_to_ego_local(o["bbox"])
            if x_m > 2.0 and abs(y_m) < 3.0 and x_m < best_x:
                best_x = x_m
                front_lead = (x_m, y_m, L, W, np.asarray(o["velocity"]))
        if front_lead is None:
            continue
        picks.append((fp, d, front_lead))
        if len(picks) >= 5:
            break
    print(f"picked {len(picks)} frames with front lead car\n")

    mpc = MPC_Class()
    print(f"MPC: N={mpc.N_MPC} Ts={mpc.Ts_MPC}\n")

    # 첫 호출만 cold start; 그 후 모두 warm
    first_call_done = False

    for i, (fp, d, lead) in enumerate(picks):
        print(f"========== {os.path.basename(fp)} ==========")
        ego = d["ego_pose"]
        vx, vy, wz = float(ego[3]), float(ego[4]), float(ego[5])
        lx, ly, L, W, lvel = lead
        print(f"  ego: vx={vx:.2f}, vy={vy:.3f}, wz={wz:.3f}")
        print(f"  lead local: ({lx:.1f}, {ly:.1f}), L={L:.1f}, W={W:.1f}, v_ego_frame=({lvel[0]:.2f}, {lvel[1]:.2f})")
        wp = d["wp_rel_gt"]
        ax_est = (wp[0, 3] - vx) / 0.1 if vx > 0 else 0.0
        vs_msg = make_vs_msg(ego, ax=ax_est, ay=0.0, tire_angle=0.0)
        traj_msg = wp_rel_to_traj_msg(wp)

        # 1) no lead — first frame cold, 나머지 warm
        cio_a, of_a = repeat(mpc, vs_msg, traj_msg, PredictObjects(),
                             n_iter=8, reset=not first_call_done)
        first_call_done = True
        # 2) with lead — warm (직전 (a) 결과로 시작 → lead 추가만 점진 적응)
        objs = lead_to_objects_msg(lx, ly, float(lvel[0]), float(lvel[1]), L, W)
        cio_b, of_b = repeat(mpc, vs_msg, traj_msg, objs, n_iter=8, reset=False)

        if cio_a is None:
            print("  (a) no lead: FAIL")
        else:
            ca = np.asarray(cio_a['acceleration'])[:8]
            print(f"  (a) no lead:  accel[:8]={[round(x,2) for x in ca]}")
        if cio_b is None:
            print("  (b) +lead:   FAIL")
        else:
            cb = np.asarray(cio_b['acceleration'])[:8]
            obj_s = of_b[0, 0, 8] if of_b is not None else None
            print(f"  (b) +lead:    accel[:8]={[round(x,2) for x in cb]}  (lead first_s={obj_s:.1f})")
        if cio_a is not None and cio_b is not None:
            ca = np.asarray(cio_a['acceleration'])[:8]
            cb = np.asarray(cio_b['acceleration'])[:8]
            diff = cb - ca
            print(f"  diff (b-a):   {[round(x,2) for x in diff]}  (negative = lead 가 더 강한 감속)")
        print()


if __name__ == "__main__":
    main()

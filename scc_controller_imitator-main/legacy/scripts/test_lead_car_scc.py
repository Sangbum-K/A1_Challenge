"""SCC 효과 검증: v0237 vs+traj + 가짜 lead car → MPC 가 감속 명령 만드는지 확인.

각 cycle 마다:
  (a) lead car 없음 (baseline) — RTI 15 iter
  (b) lead car 전방 15m, 5 m/s (vehicle 보다 느림) — RTI 15 iter
  두 결과의 accel/steer 시퀀스 비교 → (b) 가 감속 더 강해지면 SCC 동작.

다양한 lead 시나리오:
  - "far slow"      (전방 30m, 0 m/s)        — 거리 멀고 정차한 lead → 약한 감속 예상
  - "near slow"     (전방 12m, 2 m/s)        — 가깝고 느림 → 강한 감속 예상
  - "close stopped" (전방 8m, 0 m/s)         — 매우 가까이 정차 → 매우 강한 감속
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, "/home/ailab-12/git/A1_Rule/devel/lib/python3/dist-packages")
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "acados_scc", "scripts")
sys.path.insert(0, SCRIPTS_DIR)
BUILD_CACHE = os.path.join(os.path.dirname(SCRIPTS_DIR), "build_cache")
os.chdir(BUILD_CACHE)

import numpy as np
import rosbag
from MPC.NMPC_class_delayed_kinematic_cat import (
    Nonlinear_Delayed_Kinematic_Cat_Model_Predictive_Controller as MPC_Class,
)
from utils.trajectory_process import (
    npize_vehicle_state, npize_trajectory_local, npize_objects, resample_objects,
    generate_road_map, frenet_transform, frenet_mu_transform,
    global_to_local, resample_points, filtering_objects_in_lane,
)
from autohyu_msgs.msg import PredictObjects, PredictObject, Object3DState, ObjectDimension

BAG = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/controller_imitator/data/Raw_bag_multi/v0237_latlon_separate_with_without_random_rotation_combined/mpc_trajectory_generator_no_random_rotation.bag"


def make_lead_car_msg(vs_msg, distance_m, lead_vx, n_states=40, dt=0.1):
    """vehicle 전방 distance_m 에 lead car. global frame 으로 vehicle.x,y 기준 forward 방향에 둠.

    vehicle yaw 방향으로 distance_m 떨어진 위치 + 동일 yaw + lead_vx (forward) 속도.
    n_states 단계 (dt=0.1s) 의 시계열 — 등속 직진 가정.
    """
    yaw = vs_msg.yaw
    cos_y = np.cos(yaw); sin_y = np.sin(yaw)
    lead_x0 = vs_msg.x + distance_m * cos_y
    lead_y0 = vs_msg.y + distance_m * sin_y
    lead_vx_global_x = lead_vx * cos_y
    lead_vx_global_y = lead_vx * sin_y

    objs = PredictObjects()
    o = PredictObject()
    o.id = 1
    o.classification = 1
    o.dynamic_state = 0
    o.dimension = ObjectDimension(length=4.5, width=2.0, height=1.5)
    for i in range(n_states):
        s = Object3DState()
        t = i * dt
        s.x = lead_x0 + lead_vx_global_x * t
        s.y = lead_y0 + lead_vx_global_y * t
        s.z = 0.0
        s.yaw = yaw
        s.vx = lead_vx
        s.vy = 0.0
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
    # libfrenet ToFrenet 가 spline 외부 점에서 s=0 반환 버그 — 직접 nearest trajectory
    # point 기반으로 (s, n) 계산.
    cum_s = np.zeros(len(reference_map))
    if len(reference_map) > 1:
        dxy = np.diff(reference_map[:, :2], axis=0)
        cum_s[1:] = np.cumsum(np.hypot(dxy[:, 0], dxy[:, 1]))
    ref_xy = reference_map[:, :2]
    R = np.array([[np.cos(-vsg[0, 2]), -np.sin(-vsg[0, 2])], [np.sin(-vsg[0, 2]),  np.cos(-vsg[0, 2])]])
    sn_objects = np.zeros((objects.shape[0], objects.shape[1], 11))
    for i in range(objects.shape[0]):
        if objects[i, :, 4].all():
            obj = objects[i, :, :]
            for j in range(obj.shape[0]):
                obj_g = np.array([obj[j, 0], obj[j, 1]])
                obj_l = global_to_local(obj_g, vsg, R)
                # direct nearest trajectory point projection
                dxs = ref_xy[:, 0] - obj_l[0]
                dys = ref_xy[:, 1] - obj_l[1]
                dists = np.hypot(dxs, dys)
                k = int(np.argmin(dists))
                obj_s = cum_s[k]
                # signed lateral distance: cross product of segment dir × obj displacement
                if k + 1 < len(ref_xy):
                    seg = ref_xy[k + 1] - ref_xy[k]
                else:
                    seg = ref_xy[k] - ref_xy[k - 1]
                seg_n = np.hypot(seg[0], seg[1])
                if seg_n > 1e-6:
                    seg = seg / seg_n
                    rel = np.array([obj_l[0] - ref_xy[k, 0], obj_l[1] - ref_xy[k, 1]])
                    obj_n = seg[0] * rel[1] - seg[1] * rel[0]  # cross product z
                else:
                    obj_n = dists[k]
                # row layout: [x, y, yaw, id, valid, h, l, w, s, n, mu]
                sn_objects[i, j, 0] = obj_l[0]
                sn_objects[i, j, 1] = obj_l[1]
                sn_objects[i, j, 2] = obj[j, 2]   # yaw
                sn_objects[i, j, 3] = obj[j, 3]   # id
                sn_objects[i, j, 4] = obj[j, 4]   # valid
                sn_objects[i, j, 5] = obj[j, 5]   # h
                sn_objects[i, j, 6] = obj[j, 6]   # l
                sn_objects[i, j, 7] = obj[j, 7]   # w
                sn_objects[i, j, 8] = obj_s
                sn_objects[i, j, 9] = obj_n
                sn_objects[i, j, 10] = 0.0
        else:
            sn_objects[i, :, :] = 0
            sn_objects[i, :, 8] = -1e6
    # lane filter skip — sn_objects 그대로 (n 1m 제한이 트랙폭에 맞지 않아 거의 모두 mask).
    # invalid (valid=False) 객체는 우리가 이미 s=-1e6 패딩으로 처리됨.
    objects_frenet = sn_objects
    return mpc.calculate_optimal(road_map, vehicle_state, objects_frenet, trajectory_sn, measured_state), objects_frenet


def repeat_rti(mpc, vs_msg, traj_msg, objs_msg, n_iter=15):
    # 완전 reset — pred_X, pred_U, X_init 모두 None 으로
    mpc.is_MPC_init = True
    mpc.pred_X = None
    mpc.pred_U = None
    mpc.X_init = None
    mpc.fallback_count = 0
    mpc.last_mpc_status = None
    last_cio = None
    last_of = None
    for k in range(n_iter):
        out, of = run_cycle(mpc, vs_msg, traj_msg, objs_msg)
        _, _, _, _, cio = out
        if cio is not None:
            last_cio = cio
        last_of = of
    return last_cio, last_of


def main():
    bag = rosbag.Bag(BAG)
    info = bag.get_type_and_topic_info()
    total = info.topics["/app/con/control_io"].message_count
    K = 3
    step = max(1, total // 100)
    picks = []
    idx = 0
    for topic, msg, t in bag.read_messages(topics=["/app/con/control_io"]):
        idx += 1
        if idx % step != 0: continue
        # 비교 명확하게: 일정 속도 + 직진 trajectory 가 좋음
        if msg.vehicle_state.vx < 8.0 or msg.vehicle_state.vx > 18.0: continue
        # check trajectory 가 직선 가까운지: 첫 점 yaw 와 마지막 점 yaw 차이
        if abs(msg.trajectory.point[-1].yaw - msg.trajectory.point[0].yaw) > 0.2: continue
        picks.append(msg)
        if len(picks) >= K: break
    bag.close()
    print(f"picked {len(picks)} candidate samples (straight, 8-18 m/s)")

    mpc = MPC_Class()
    print(f"\nMPC: N={mpc.N_MPC}, Ts={mpc.Ts_MPC}\n")

    empty = PredictObjects()
    scenarios = [
        ("(a) no lead car",         None),
        ("(b) far  40m, 12 m/s",    (40.0, 12.0)),   # 살짝 느린 lead, far
        ("(c) mid  25m, 6 m/s",     (25.0, 6.0)),    # 절반 속도
        ("(d) near 18m, 3 m/s",     (18.0, 3.0)),    # 거의 정차에 가까움
    ]

    iter_choices = [1, 3, 5, 8]
    for ci, msg in enumerate(picks[:1]):  # cycle 0 만으로 iter sweep
        vx = msg.vehicle_state.vx
        print(f"========== cycle {ci}, vx={vx:.2f} m/s ==========")
        for n_it in iter_choices:
            print(f"\n  -- n_iter={n_it} --")
            for name, params in scenarios:
                if params is None:
                    objs = empty
                else:
                    d, lv = params
                    objs = make_lead_car_msg(msg.vehicle_state, d, lv)
                cio, of = repeat_rti(mpc, msg.vehicle_state, msg.trajectory, objs, n_iter=n_it)
                if cio is None:
                    print(f"    {name:30s} FAIL fallback")
                    continue
                ca = np.asarray(cio['acceleration'])
                obj_info = ""
                if of is not None and len(of) > 0 and of[0, :, 4].any():
                    obj_info = f"  s={of[0,0,8]:.1f}"
                print(f"    {name:30s} accel[:6]=[{ca[0]:.1f},{ca[1]:.1f},{ca[2]:.1f},{ca[3]:.1f},{ca[4]:.1f},{ca[5]:.1f}]{obj_info}")


if __name__ == "__main__":
    main()

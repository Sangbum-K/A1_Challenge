"""RTI 가 cold start 에서 수렴 못해 steer=0 출력 → 같은 입력 N 번 반복 호출 로 수렴 유도."""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, "/home/ailab-12/git/A1_Rule/devel/lib/python3/dist-packages")
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "acados_scc", "scripts")
sys.path.insert(0, SCRIPTS_DIR)
BUILD_CACHE = os.path.join(os.path.dirname(SCRIPTS_DIR), "build_cache")
os.chdir(BUILD_CACHE)

import numpy as np
import rosbag
from MPC.NMPC_class_delayed_kinematic_cat import Nonlinear_Delayed_Kinematic_Cat_Model_Predictive_Controller as MPC_Class
from utils.trajectory_process import (
    npize_vehicle_state, npize_trajectory_local, npize_objects, resample_objects,
    generate_road_map, frenet_transform, frenet_mu_transform,
    global_to_local, resample_points, filtering_objects_in_lane,
)
from autohyu_msgs.msg import PredictObjects

BAG = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/controller_imitator/data/Raw_bag_multi/v0237_latlon_separate_with_without_random_rotation_combined/mpc_trajectory_generator_no_random_rotation.bag"


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
    R = np.array([[np.cos(-vsg[0, 2]), -np.sin(-vsg[0, 2])], [np.sin(-vsg[0, 2]),  np.cos(-vsg[0, 2])]])
    sn_objects = np.zeros((objects.shape[0], objects.shape[1], 11))
    for i in range(objects.shape[0]):
        if objects[i, :, 4].all():
            obj = objects[i, :, :]
            for j in range(obj.shape[0]):
                obj_g = np.array([obj[j, 0], obj[j, 1]])
                obj_l = global_to_local(obj_g, vsg, R)
                obj[j, 0] = obj_l[0]; obj[j, 1] = obj_l[1]
            obj_sn = frenet_transform(road_map, obj); obj_sn = frenet_mu_transform(road_map, obj_sn)
            if np.isnan(obj_sn).any(): obj_sn[np.isnan(obj_sn)] = 0
            sn_objects[i, :, :] = obj_sn
        else:
            sn_objects[i, :, :] = np.zeros((objects.shape[1], 11))
            sn_objects[i, :, 8] = -1e6
    objects_frenet = filtering_objects_in_lane(sn_objects, reference_map)
    return mpc.calculate_optimal(road_map, vehicle_state, objects_frenet, trajectory_sn, measured_state)


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
        if msg.vehicle_state.vx < 1.0: continue
        picks.append(msg)
        if len(picks) >= K: break
    bag.close()
    print(f"picked {len(picks)} msgs")

    mpc = MPC_Class()
    print(f"N={mpc.N_MPC}, Ts={mpc.Ts_MPC}")

    empty_objs = PredictObjects()
    for ci, msg in enumerate(picks):
        print(f"\n========== cycle {ci} (vx={msg.vehicle_state.vx:.2f}) ==========")
        gt_steer = list(msg.command_steer)
        gt_accel = list(msg.command_acceleration)
        print(f"  GT_steer[:5]: {[round(x,2) for x in gt_steer[:5]]}")
        print(f"  GT_accel[:5]: {[round(x,2) for x in gt_accel[:5]]}")
        # 첫 호출 only init
        mpc.is_MPC_init = True
        for iter_n in range(15):
            out = run_cycle(mpc, msg.vehicle_state, msg.trajectory, empty_objs)
            target_cart, target_pred, target_input, stats, control_io = out
            status = mpc.acados_solver.get_status()
            if control_io is None:
                print(f"  iter {iter_n}: status={status}, FALLBACK")
                continue
            cs = list(control_io['steer'])
            ca = list(control_io['acceleration'])
            print(f"  iter {iter_n}: status={status}, ours_steer[:5]={[round(x,2) for x in cs[:5]]}, ours_accel[:5]={[round(x,2) for x in ca[:5]]}")


if __name__ == "__main__":
    main()

"""acados_scc_control_node.py 의 loop() 한 cycle 을 v0237 raw bag 의 msg 객체로 정확히 재현.

ROS 환경 없이 (roscore 없이) bag.read_messages 로 ControlInputOutput msg 를 가져와서
그 안의 vehicle_state, trajectory 를 node 의 callback 결과처럼 그대로 사용.

핵심: numpy 변환을 우리가 직접 하지 않고, ROS msg → npize_* 함수에 그대로 전달.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, "/home/ailab-12/git/A1_Rule/devel/lib/python3/dist-packages")

SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "external", "acados_scc", "scripts",
)
sys.path.insert(0, SCRIPTS_DIR)
BUILD_CACHE = os.path.join(os.path.dirname(SCRIPTS_DIR), "build_cache")
os.makedirs(BUILD_CACHE, exist_ok=True)
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
from autohyu_msgs.msg import PredictObjects

BAG = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/controller_imitator/data/Raw_bag_multi/v0237_latlon_separate_with_without_random_rotation_combined/mpc_trajectory_generator_no_random_rotation.bag"


def run_node_cycle(mpc, vs_msg, traj_msg, objects_msg):
    """ROS node loop() 한 cycle 의 전처리 + calculate_optimal 호출. acados_scc_control_node.py line 175~246 그대로."""
    # Vehicle State: [x, y, yaw, speed]; Measured: [x, y, yaw, speed, vy, ax, ay, tire]
    vehicle_state, measured_state = npize_vehicle_state(vs_msg)

    # target_trajectory: [x, y, yaw, speed]; reference_map: [x, y, ...]
    target_trajectory, reference_map = npize_trajectory_local(traj_msg, vehicle_state)

    # objects
    objects = npize_objects(objects_msg, max_objects_num=1)
    objects = resample_objects(objects, 0.1, 40, mpc.Ts_MPC, mpc.N_MPC + 1)

    road_map = generate_road_map(reference_map)

    # Frenet transform vehicle_state
    vehicle_state_global = vehicle_state.copy()
    vehicle_state[0, 0] = 0.0
    vehicle_state[0, 1] = 0.0
    vehicle_state[0, 2] = 0.0
    vehicle_state = frenet_transform(road_map, vehicle_state)
    vehicle_state = frenet_mu_transform(road_map, vehicle_state)

    # Resampled target trajectory (not used in MPC call but ROS node computes)
    target_traj_resampled = resample_points(target_trajectory, vehicle_state[0, 4], mpc.Ts_MPC, mpc.N_MPC + 1)
    target_traj_resampled_frenet_sn = frenet_transform(road_map, target_traj_resampled)
    target_traj_resampled_frenet = frenet_mu_transform(road_map, target_traj_resampled_frenet_sn, start_mu=vehicle_state[0, 6])

    # trajectory_sn (this is what's passed to MPC)
    trajectory_sn = frenet_transform(road_map, target_trajectory)
    trajectory_sn = frenet_mu_transform(road_map, trajectory_sn, start_mu=vehicle_state[0, 6])

    # objects → frenet
    rotation_matrix = np.array([[np.cos(-vehicle_state_global[0, 2]), -np.sin(-vehicle_state_global[0, 2])],
                                [np.sin(-vehicle_state_global[0, 2]),  np.cos(-vehicle_state_global[0, 2])]])
    sn_objects = np.zeros((objects.shape[0], objects.shape[1], 11))
    for i in range(objects.shape[0]):
        if objects[i, :, 4].all():
            obj = objects[i, :, :]
            for j in range(obj.shape[0]):
                obj_g = np.array([obj[j, 0], obj[j, 1]])
                obj_l = global_to_local(obj_g, vehicle_state_global, rotation_matrix)
                obj[j, 0] = obj_l[0]
                obj[j, 1] = obj_l[1]
            obj_sn = frenet_transform(road_map, obj)
            obj_sn = frenet_mu_transform(road_map, obj_sn)
            if np.isnan(obj_sn).any():
                obj_sn[np.isnan(obj_sn)] = 0
            sn_objects[i, :, :] = obj_sn
        else:
            sn_objects[i, :, :] = np.zeros((objects.shape[1], 11))
            sn_objects[i, :, 8] = -1e6
    objects_frenet = filtering_objects_in_lane(sn_objects, reference_map)

    # MPC call
    out = mpc.calculate_optimal(road_map, vehicle_state, objects_frenet, trajectory_sn, measured_state)
    return out, vehicle_state, trajectory_sn, target_trajectory, reference_map


def main():
    print(f"opening bag (3min) ...")
    t0 = time.time()
    bag = rosbag.Bag(BAG)
    info = bag.get_type_and_topic_info()
    total = info.topics["/app/con/control_io"].message_count
    print(f"  open {time.time()-t0:.0f}s, total {total} msgs")

    # collect first K msgs with vx > 1.0
    K = 5
    picks = []
    step = max(1, total // 100)
    idx = 0
    for topic, msg, t in bag.read_messages(topics=["/app/con/control_io"]):
        idx += 1
        if idx % step != 0:
            continue
        if msg.vehicle_state.vx < 1.0:
            continue
        picks.append(msg)
        if len(picks) >= K:
            break
    bag.close()
    print(f"  picked {len(picks)} msgs")

    print("\nMPC init...")
    t0 = time.time()
    mpc = MPC_Class()
    print(f"  init: {time.time()-t0:.1f}s")

    empty_objects = PredictObjects()
    for i, msg in enumerate(picks):
        print(f"\n========== cycle {i} (vx={msg.vehicle_state.vx:.2f}) ==========")
        print(f"  vs_global x={msg.vehicle_state.x:.3f} y={msg.vehicle_state.y:.3f} yaw={msg.vehicle_state.yaw:.4f}")
        print(f"  vs ax={msg.vehicle_state.ax:.3f} ay={msg.vehicle_state.ay:.3f} "
              f"tire={msg.vehicle_state.vehicle_can.steering_tire_angle:.4f}")
        print(f"  traj points: {len(msg.trajectory.point)}, first xy=({msg.trajectory.point[0].x:.2f}, {msg.trajectory.point[0].y:.2f})")
        try:
            mpc.is_MPC_init = True
            out, vs_local, traj_sn, traj_local, ref_map = run_node_cycle(mpc, msg.vehicle_state, msg.trajectory, empty_objects)
            target_cart, target_pred, target_input, stats, control_io = out
            status = mpc.acados_solver.get_status()
            print(f"  vs_local s={vs_local[0,4]:.4f}, n={vs_local[0,5]:.4f}, mu={vs_local[0,6]:.4f}")
            print(f"  traj_local first 3 xy: {traj_local[:3, :2].tolist()}")
            print(f"  traj_sn s=[{traj_sn[:,4].min():.2f},{traj_sn[:,4].max():.2f}] mu=[{traj_sn[:,6].min():.3f},{traj_sn[:,6].max():.3f}]")
            print(f"  status={status}, target_input={target_input}")
            if control_io is not None:
                print(f"  ours steer[:5]: {[round(x,3) for x in control_io['steer'][:5]]}")
                print(f"  ours accel[:5]: {[round(x,3) for x in control_io['acceleration'][:5]]}")
                gts = list(msg.command_steer)
                gta = list(msg.command_acceleration)
                print(f"  GT_steer[:5]:   {[round(x,3) for x in gts[:5]]}")
                print(f"  GT_accel[:5]:   {[round(x,3) for x in gta[:5]]}")
        except Exception as e:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

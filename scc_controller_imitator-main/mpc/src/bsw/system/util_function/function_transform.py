import os
import numpy as np
import torch

def global_to_local(ego_origin, ego_yaw, trajectory_points_global):
    # trajectory (local)
    trajectory_points_local = np.zeros_like(trajectory_points_global)
    # rotation matrix
    rotation_matrix = np.array([[np.cos(ego_yaw),  np.sin(ego_yaw)], [-np.sin(ego_yaw),  np.cos(ego_yaw)]])
    relative_positions = trajectory_points_global[:, 0:2] - ego_origin
    trajectory_points_local[:, 0:2] = np.dot(relative_positions, rotation_matrix.T)
    trajectory_points_local[:, 2] = trajectory_points_global[:, 2] - ego_yaw
    trajectory_points_local[:, 3] = trajectory_points_global[:, 3]
    return trajectory_points_local

def radian_to_degree(radian):
    # radian to degree 
    return radian * 180 / np.pi

def degree_to_radian(degree):
    # degree to radian
    return degree * np.pi / 180
def ax_to_steer_brake(ax):
    ax = np.asarray(ax)
    
    # numpy의 vectorized 연산을 이용해 조건에 따라 값을 선택합니다.
    accel = np.where(ax > 0, ax, 0)
    brake = np.where(ax < 0, -ax, 0)
    
    return accel, brake

def bag_time_sync(bag, topics, dst_path, data_type):
    ## extract key-value
    print("extract key-value")
    print("Data will save at: {}".format(dst_path))
    for idx, (topic, msg, t) in enumerate(bag.read_messages(topics=topics)):
        # vehicle_state
        ego_origin = np.array([msg.vehicle_state.x, msg.vehicle_state.y])
        ego_yaw = msg.vehicle_state.yaw
        ego_velocity = np.array([msg.vehicle_state.vx, msg.vehicle_state.vy])
        ego_acceleration = np.array([msg.vehicle_state.ax, msg.vehicle_state.ay])
        ego_tire_angle = msg.vehicle_state.vehicle_can.steering_tire_angle
        ego_yaw_rate = msg.vehicle_state.yaw_vel
        
        # target_value (ax / steer)
        target_value = msg.command_output
        
        # trajectory (global)
        num_of_points = len(msg.trajectory.point)
        trajectory_points_global = np.zeros((num_of_points, 4)) # x, y, yaw, speed
        for point_idx in range(num_of_points):
            trajectory_points_global[point_idx, 0] = msg.trajectory.point[point_idx].x
            trajectory_points_global[point_idx, 1] = msg.trajectory.point[point_idx].y
            trajectory_points_global[point_idx, 2] = msg.trajectory.point[point_idx].yaw
            trajectory_points_global[point_idx, 3] = msg.trajectory.point[point_idx].speed
            
        # trajectory (local)
        trajectory_points_local = global_to_local(ego_origin, ego_yaw, trajectory_points_global)
        
        dict_data = {
            'ego_origin': ego_origin,
            'ego_yaw': ego_yaw,
            'ego_velocity': ego_velocity,
            'ego_acceleration': ego_acceleration,
            'ego_tire_angle': ego_tire_angle,
            'ego_yaw_rate': ego_yaw_rate,
            'trajectory_points_local': trajectory_points_local,
            'target_value': target_value
        }
        
        torch.save(dict_data, os.path.join(dst_path, "{}_{:05d}.pt".format(data_type, idx)))
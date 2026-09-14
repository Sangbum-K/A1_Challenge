import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import Akima1DInterpolator
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

import math
import time
from acados_scc_control import libfrenet_scc as frenet
from geometry_msgs.msg import Point, Pose, Quaternion

def npize_vehicle_state(vehicle_state):
    x0 = np.zeros((1, 4)) # x,y,yaw,speed
    x0[0, 0] = vehicle_state.x
    x0[0, 1] = vehicle_state.y
    x0[0, 2] = vehicle_state.yaw
    x0[0, 3] = vehicle_state.vx

    measured_state = np.zeros((1, 8)) # x,y,yaw,speed,s,n,mu
    measured_state[0, 0] = vehicle_state.x
    measured_state[0, 1] = vehicle_state.y
    measured_state[0, 2] = vehicle_state.yaw
    measured_state[0, 3] = vehicle_state.vx
    measured_state[0, 4] = vehicle_state.vy
    measured_state[0, 5] = vehicle_state.ax
    measured_state[0, 6] = vehicle_state.ay
    measured_state[0, 7] = vehicle_state.vehicle_can.steering_tire_angle

    return x0, measured_state
    

def npize_trajectory(trajectory):
    point           = np.zeros((len(trajectory.point), 4))          # x,y,yaw,speed
    reference_map   = np.zeros((len(trajectory.reference_map), 5))  # x,y
    right_boundary  = np.zeros((len(trajectory.right_boundary), 3)) # x,y
    left_boundary   = np.zeros((len(trajectory.left_boundary), 3))  # x,y
    
    # point: [x, y, yaw, speed]
    for i in range(len(trajectory.point)):
        point[i, 0] = trajectory.point[i].x
        point[i, 1] = trajectory.point[i].y
        point[i, 2] = trajectory.point[i].yaw
        point[i, 3] = trajectory.point[i].speed

    # reference_map: [x, y]
    for i in range(len(trajectory.reference_map)):
        reference_map[i, 0] = trajectory.reference_map[i].x
        reference_map[i, 1] = trajectory.reference_map[i].y

    # right_boundary: [x, y, yaw], yaw is calculated from x, y
    for i in range(len(trajectory.right_boundary)):
        right_boundary[i, 0] = trajectory.right_boundary[i].x
        right_boundary[i, 1] = trajectory.right_boundary[i].y
    
    # Right Boundary 스무딩
    # right_boundary[:, 0:2] = smooth_boundary(right_boundary[:, 0:2])
    right_boundary[:, 0:2] = gaussian_smooth(right_boundary[:, 0:2], sigma=5.0)

    right_boundary = compute_yaw(right_boundary)

    # left_boundary: [x, y, yaw], yaw is calculated from x, y
    for i in range(len(trajectory.left_boundary)):
        left_boundary[i, 0] = trajectory.left_boundary[i].x
        left_boundary[i, 1] = trajectory.left_boundary[i].y

    # Left Boundary 스무딩
    # left_boundary[:, 0:2] = smooth_boundary(left_boundary[:, 0:2])
    left_boundary[:, 0:2] = gaussian_smooth(left_boundary[:, 0:2], sigma=5.0)

    left_boundary = compute_yaw(left_boundary)


    return point,reference_map,right_boundary,left_boundary

def global_to_local(global_point, vehicle_state, rotation_matrix):
    vehicle_x = vehicle_state[0,0]
    vehicle_y = vehicle_state[0,1]
    vehicle_yaw = vehicle_state[0,2]
    local_point = global_point - np.array([vehicle_x, vehicle_y])
    local_point = rotation_matrix @ local_point
    return local_point

def local_to_global(local_point, vehicle_state, rotation_matrix):
    vehicle_x = vehicle_state[0,0]
    vehicle_y = vehicle_state[0,1]
    vehicle_yaw = vehicle_state[0,2]
    global_point = rotation_matrix @ local_point + np.array([vehicle_x, vehicle_y])
    return global_point

def npize_trajectory_local(trajectory, vehicle_state):
    vehicle_x = vehicle_state[0,0]
    vehicle_y = vehicle_state[0,1]
    vehicle_yaw = vehicle_state[0,2]
    # numpy 회전 횡렬
    rotation_matrix = np.array([[np.cos(vehicle_yaw), -np.sin(vehicle_yaw)],
                                [np.sin(vehicle_yaw), np.cos(vehicle_yaw)]]).T
        
    point           = np.zeros((len(trajectory.point), 4))          # x,y,yaw,speed
    reference_map   = np.zeros((len(trajectory.point), 5))  # x,y
    
    # point: [x, y, yaw, speed]
    for i in range(len(trajectory.point)):
        global_point = np.array([trajectory.point[i].x, trajectory.point[i].y])
        local_point = global_to_local(global_point, vehicle_state, rotation_matrix)
        point[i, 0] = local_point[0]
        point[i, 1] = local_point[1]
        point[i, 2] = trajectory.point[i].yaw - vehicle_yaw
        point[i, 3] = trajectory.point[i].speed

    # reference_map: [x, y]
    for i in range(len(trajectory.point)):
        global_point = np.array([trajectory.point[i].x, trajectory.point[i].y])
        local_point = global_to_local(global_point, vehicle_state, rotation_matrix)
        reference_map[i, 0] = local_point[0]
        reference_map[i, 1] = local_point[1]

    return point,reference_map


def resample_points(point, ego_s, dt, sample_num):
    """
    Breif:
        입력된 point에 대해 동일 시간 간격 (dt)를 기준으로 샘플링합니다.
        ego_s 시작지점 기준으로 sample_num개의 포인트를 샘플링합니다.
    Input: 
        point: N x 6 배열 [x, y, yaw, speed s, n] 이라고 가정하나,
           여기서는 s가 없다면 arc-length로 s를 계산.
           (만약 point[:,4]에 s가 이미 있다면 arc-length 계산은 생략 가능.)
    
        ego_s: 시작 지점의 s 값 (arc-length)
        dt   : 시간 간격 (초 단위)
        sample_num: 샘플링할 포인트 개수
    
    Output:
        resampled_point [x, y, yaw, speed]: 샘플링된 포인트 배열
    """
    
    x_orig = point[:, 0]
    y_orig = point[:, 1]
    yaw_orig = point[:, 2]
    speed_orig = point[:, 3]  # 각 포인트에서의 속도
    
    # s_orig 계산: arc-length
    dx = np.diff(x_orig)
    dy = np.diff(y_orig)
    dist = np.sqrt(dx**2 + dy**2)
    s_orig = np.zeros(len(x_orig))
    s_orig[1:] = np.cumsum(dist)
    
    # 구간 시간 계산 시 평균 속도 사용
    # 구간 i->i+1에 대해 평균 속도 = (speed_orig[i] + speed_orig[i+1]) / 2
    avg_speed = np.maximum((speed_orig[:-1] + speed_orig[1:]) / 2.0, 0.001)
    segment_time = dist / avg_speed
    
    # 각 포인트에 해당하는 누적 시간 t_orig
    t_orig = np.zeros_like(s_orig)
    t_orig[1:] = np.cumsum(segment_time)
    
    # ego_s에 해당하는 시간 t_ego_s
    t_ego_s = np.interp(ego_s, s_orig, t_orig)
    
    # t_ego_s부터 dt 간격으로 sample_num개 시간 생성
    t_new = t_ego_s + np.arange(sample_num)*dt
    
    # t_new -> s_new 보간: t축에서 s 축으로
    s_new = np.interp(t_new, t_orig, s_orig)
    
    # s_new에 대해 x, y, yaw, speed 보간
    x_new = np.interp(s_new, s_orig, x_orig)
    y_new = np.interp(s_new, s_orig, y_orig)
    yaw_new = np.interp(s_new, s_orig, yaw_orig)
    speed_new = np.interp(s_new, s_orig, speed_orig)
    
    # 리샘플링된 포인트 배열
    # 만약 결과에 s,n 등이 필요하다면 추가 보간 또는 계산 필요
    # 여기서는 [x, y, yaw, speed]만 리턴
    resampled_point = np.zeros((sample_num, 4))
    resampled_point[:,0] = x_new
    resampled_point[:,1] = y_new
    resampled_point[:,2] = yaw_new
    resampled_point[:,3] = speed_new
    
    return resampled_point


def npize_objects(objects, max_objects_num):
    '''
    Input:
        objects: [x, y, yaw, id, True(valid), height, length, width]
    Return:
        all_data: [x, y, yaw, id, True(valid), height, length, width, s, n, mu]
    '''
    all_data = []
    objects_num = 0
    
    # 먼저 실제 objects에 있는 만큼 변환
    for obj in objects.object:
        obj_id = obj.id
        length = obj.dimension.length
        width = obj.dimension.width
        height = obj.dimension.height

        # Extract positions and orientations into NumPy arrays
        positions = np.array([[state.x, state.y] for state in obj.state])  # Shape: (num_states, 2)
        yaws = np.array([state.yaw for state in obj.state])                # Shape: (num_states,)
        
        num_states = positions.shape[0]
        
        obj_data = np.column_stack((
            positions,                            # Transformed x and y positions
            yaws,                                 # Transformed yaw angles
            np.full(num_states, obj_id),          # Object ID
            np.full(num_states, True),            # True
            np.full(num_states, height),          # Object height
            np.full(num_states, length),          # Object length
            np.full(num_states, width)            # Object width
        ))

        all_data.append(obj_data)
        objects_num += 1

    # objects_num이 max_objects_num보다 많은 경우, 상위 max_objects_num개로 제한
    if objects_num > max_objects_num:
        all_data = all_data[:max_objects_num]
        objects_num = max_objects_num
    # objects_num이 max_objects_num보다 적은 경우, 나머지 부분을 zero-padding
    elif objects_num < max_objects_num:
        # 부족한 개수 만큼 채워넣기
        for i in range(max_objects_num - objects_num):
            obj_data = np.column_stack((
                np.full(40, 0),      # Transformed x 
                np.full(40, 0),      # Transformed y positions
                np.full(40, 0),      # Transformed yaw angles
                np.full(40, 0),      # Object ID
                np.full(40, False),  # False
                np.full(40, 0),      # Object height
                np.full(40, 0),      # Object length
                np.full(40, 0)       # Object width
            ))
            all_data.append(obj_data)

    return np.array(all_data)

def resample_objects(objects, origin_dt, origin_sample_num, target_dt, target_sample_num):
    '''
    Input:
        objects: [object_num, timestep, features] 형태의 numpy array
        origin_dt: 원본 시간 간격
        origin_sample_num: 원본 샘플 개수
        target_dt: 목표 시간 간격
        target_sample_num: 목표 샘플 개수
    Output:
        resampled_objects: [object_num, target_sample_num, features] 형태의 numpy array
    '''
    object_num = objects.shape[0]
    feature_num = objects.shape[2]
    
    # 원본 시간 포인트
    t_original = np.linspace(0, origin_dt * (origin_sample_num-1), origin_sample_num)
    # 목표 시간 포인트
    t_target = np.linspace(0, target_dt * (target_sample_num-1), target_sample_num)
    
    # 결과를 저장할 배열
    resampled_objects = np.zeros((object_num, target_sample_num, feature_num))
    
    # 각 객체와 특성에 대해 보간 수행
    for obj_idx in range(object_num):
        for feat_idx in range(feature_num):
            # 현재 객체의 현재 특성에 대한 시계열 데이터
            current_feature = objects[obj_idx, :, feat_idx]
            
            # 선형 보간 수행
            resampled_feature = np.interp(t_target, t_original, current_feature)
            
            # 결과 저장
            resampled_objects[obj_idx, :, feat_idx] = resampled_feature
            
    return resampled_objects


def generate_road_map(reference_map):
    '''
    reference_map: [x, y]
    '''
    traj_x = reference_map[:, 0]
    traj_y = reference_map[:, 1]
    return frenet.RoadMap(traj_x.tolist(), traj_y.tolist())

def frenet_transform(road_map, point):
    '''
    Input: 
        road_map: Frenet Road Map
        point: [x, y, yaw, speed]
    Output:
        point: [x, y, yaw, speed, s, n]
    '''
    x = point[:,0]
    y = point[:,1]
    
    sn = np.array(road_map.ToFrenet(x.tolist(), y.tolist()))
    return np.hstack((point, sn.T)) 

def frenet_mu_transform(road_map, point, start_mu = 0):
    '''
    Input: 
        road_map: Frenet Road Map
        point: [x, y, yaw, speed, s, n]
    Output:
        point: [x, y, yaw, speed, s, n, mu]
    '''
    yaw = point[:,  2] 
    s   = point[:, -2]
    ref_yaw = np.array(road_map.GetYaw(s.tolist()))
    mu = (yaw - ref_yaw).reshape(-1, 1)

    mu_with_start = np.vstack(([start_mu], mu))
    mu_unwrap_with_start = np.unwrap(mu_with_start, axis=0)

    mu_unwrap = mu_unwrap_with_start[1:] 
    return np.hstack((point, mu_unwrap)) 

def frenet_to_cartesian(road_map, pred_X):
    
    s_pred  = pred_X[:,0]
    n_pred  = pred_X[:,1]
    mu_pred = pred_X[:,2]
    xy = np.array(road_map.ToCartesian(s_pred.tolist(),n_pred.tolist()))
    ref_yaw = np.array(road_map.GetYaw(s_pred.tolist()))

    yaw = ref_yaw + mu_pred
    return np.vstack((xy,yaw))

def yaw_to_quaternion(yaw):
    # Convert a yaw angle (in radians) to a Quaternion message
    q = Quaternion()
    q.w = np.cos(yaw / 2)
    q.x = 0.0
    q.y = 0.0
    q.z = np.sin(yaw / 2)
    return q

def calculate_path_length(boundary):
    # x와 y 좌표 추출
    x = boundary[:, 0]
    y = boundary[:, 1]
    
    # 연속된 점들 사이의 거리 계산
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    
    # 총 길이 반환
    return np.sum(distances)


def find_closest(b_sn, target_s):
    #180x3 s n mu
    b_s = b_sn[:,0]

    # Ensure array is a NumPy array and sorted
    b_s = np.asarray(b_s)
    target_s = np.asarray(target_s)
    
    # 각 pred_s 값에 대해 closest index 찾기
    indices = np.searchsorted(b_s, target_s, side="left")
    
    # 배열 범위 밖의 인덱스 처리 (맨 처음과 끝 경계)
    indices = np.clip(indices, 1, len(b_s) - 1)
    
    # left와 right 값 추출
    left = b_s[indices - 1]
    right = b_s[indices]
    
    # values에서 left와 right 중 더 가까운 값을 선택
    mask = np.abs(target_s - left) < np.abs(target_s - right)
    
    # 더 가까운 값에 따라 left 또는 right의 인덱스를 선택
    closest_indices = np.where(mask, indices - 1, indices)
    

    sn_closest = np.vstack((b_sn[:,0][closest_indices], b_sn[:,1][closest_indices], b_sn[:,2][closest_indices])).T

    
    return sn_closest

def compute_tangent_lines(sn_closest, shift_distance, side):
    """
    각 (s, n, mu) 포인트에 대해 접선인 일차 방정식 y = m * x + c을 계산하고,
    지정된 거리만큼 평행 이동합니다.

    Parameters:
    ----------
    sn_closest : np.ndarray
        shape (N, 3), 각 행이 (s, n, mu) 포인트를 나타냅니다.
        s는 프레넷 s 좌표, n은 프레넷 n 좌표, mu는 도로의 방향각입니다.
    shift_distance : float
        직선을 평행 이동할 거리 (양수).
    side : str
        'left' 또는 'right'로, 차선의 방향을 지정합니다.

    Returns:
    -------
    m : np.ndarray
        각 포인트에서의 기울기.
    c : np.ndarray
        각 포인트에서의 평행 이동된 절편.
    """
    if not isinstance(sn_closest, np.ndarray):
        raise TypeError("sn_closest는 NumPy 배열이어야 합니다.")
    if sn_closest.shape[0] < 1:
        raise ValueError("sn_closest에는 최소 1개의 포인트가 있어야 합니다.")

    # s, n, mu 분리
    s = sn_closest[:, 0]
    n = sn_closest[:, 1]
    mu = sn_closest[:, 2]
    
    # 기울기 m 계산 (mu를 사용)
    m = np.tan(mu)

    # 원래 절편 c 계산
    c = n - m * s

    # 평행 이동량 계산
    shift = shift_distance * np.sqrt(1 + m**2)

    # 이동 방향에 따른 절편 조정
    if side == 'left':
        # 왼쪽 차선은 n 축의 음의 방향으로 이동
        c_new = c - shift
    elif side == 'right':
        # 오른쪽 차선은 n 축의 양의 방향으로 이동
        c_new = c + shift
    else:
        raise ValueError("side 매개변수는 'left' 또는 'right'여야 합니다.")

    return m, c_new


def compute_yaw(boundary_array):
    """
    경계 배열에 대해 yaw를 계산하여 세 번째 열에 추가합니다.

    Parameters:
    ----------
    boundary_array : np.ndarray
        shape (N, 3), 각 행이 (x, y, yaw)를 나타냅니다.
        yaw는 계산 후 세 번째 열에 저장됩니다.

    Returns:
    -------
    boundary_array : np.ndarray
        yaw가 계산되어 세 번째 열에 추가된 배열.
    """
    x = boundary_array[:, 0]
    y = boundary_array[:, 1]

    # dx와 dy 계산
    dx = np.diff(x)
    dy = np.diff(y)

    # yaw 계산 (N - 1 개의 값)
    yaw = np.arctan2(dy, dx)

    # yaw를 boundary_array에 추가
    boundary_array[:-1, 2] = yaw

    # 마지막 점의 yaw는 이전 yaw와 동일하게 설정
    boundary_array[-1, 2] = boundary_array[-2, 2]

    return boundary_array


def smooth_boundary(boundary_points, window_length=5, polyorder=3):
    """
    경계 포인트를 스무딩합니다.

    Parameters:
    ----------
    boundary_points : np.ndarray
        shape (N, 2), 각 행이 (x, y)를 나타냅니다.
    window_length : int
        스무딩에 사용할 윈도우 길이 (홀수여야 함).
    polyorder : int
        다항식 차수 (window_length보다 작아야 함).

    Returns:
    -------
    smoothed_points : np.ndarray
        스무딩된 경계 포인트 배열.
    """
    # 윈도우 길이는 홀수여야 하며, 데이터 길이보다 작아야 함
    if len(boundary_points) < window_length:
        window_length = len(boundary_points) if len(boundary_points) % 2 != 0 else len(boundary_points) - 1
    if window_length < 3:
        window_length = 3  # 최소 윈도우 길이

    x = boundary_points[:, 0]
    y = boundary_points[:, 1]

    # Savitzky-Golay 필터를 적용하여 스무딩
    x_smooth = savgol_filter(x, window_length=window_length, polyorder=polyorder)
    y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)

    smoothed_points = np.column_stack((x_smooth, y_smooth))

    return smoothed_points


def gaussian_smooth(data, sigma=2.0):
    """
    가우시안 필터를 사용하여 데이터를 스무딩합니다.

    Parameters:
    ----------
    data : np.ndarray
        shape (N, 2), 각 행이 (x, y)를 나타냅니다.
    sigma : float
        가우시안 커널의 표준 편차.

    Returns:
    -------
    smoothed_data : np.ndarray
        스무딩된 데이터 배열.
    """
    x = data[:, 0]
    y = data[:, 1]

    x_smooth = gaussian_filter1d(x, sigma=sigma)
    y_smooth = gaussian_filter1d(y, sigma=sigma)

    smoothed_data = np.column_stack((x_smooth, y_smooth))

    return smoothed_data

def compute_object_gap_positions(objects, extra_gap = 0.0):
    """
    Compute front, center, and rear bumper positions for obstacles.
    
    Args:
        objects: np.ndarray with shape (max_objects_num, num_states, 11)
                 columns: [x, y, yaw, obj_id, bool, height, length, width, s, n, mu]
    
    Returns:
        bumpers: np.ndarray with shape (max_objects_num, num_states, 6)
                 columns: [s_ofront, n_ofront, s_ocenter, n_ocenter, s_orear, n_orear]
    """

    # 인덱스 매핑
    length_col = 6
    s_col = 8
    n_col = 9
    mu_col = 10

    l_o = objects[:, :, length_col] + extra_gap  # Length of the obstacle
    s_o = objects[:, :, s_col]      # Frenet 's' position of the obstacle
    n_o = objects[:, :, n_col]      # Frenet 'n' position of the obstacle
    mu_o = objects[:, :, mu_col]    # Orientation of the obstacle in Frenet coordinates

    # 각 범퍼 위치 계산
    s_ofront = s_o + (l_o / 2) * np.cos(mu_o)
    n_ofront = n_o + (l_o / 2) * np.sin(mu_o)

    s_ocenter = s_o
    n_ocenter = n_o

    s_orear = s_o - (l_o / 2) * np.cos(mu_o)
    n_orear = n_o - (l_o / 2) * np.sin(mu_o)

    # 하나의 배열로 합치기 (shape: (max_objects_num, num_states, 6))
    bumpers = np.stack([s_ofront, n_ofront, s_ocenter, n_ocenter, s_orear, n_orear], axis=-1)

    return bumpers


def filtering_objects_in_lane(objects, reference_map):
    """
    objects: (N,40,10) array
             Each object: 40 states, each state: [x,y,yaw,id,valid,height,length,width,s,n]
    left_boundary: shape (L, ...), [x,y,yaw,s,n], 여기서 s=3, n=4 인덱스를 사용
    right_boundary: shape (R, ...), [x,y,yaw,s,n], 여기서 s=3, n=4 인덱스를 사용
    
    Returns:
        filtered_objects: 동일 shape (N,40,10)
                          차선 밖 또는 s 범위 밖 상태는 0으로 패딩
    """
    s_max = np.max(reference_map[:, 3])
    
    filtered_objects = objects.copy()

    N = objects.shape[0]
    num_states = objects.shape[1]

    # 각 객체에 대해 차선 내부 및 s 범위 내 상태만 유지
    for i in range(N):
        for j in range(num_states):
            state = filtered_objects[i, j, :]
            valid = state[4]  # valid 필드 가정
            s_val = state[8]
            n_val = state[9]

            # s 범위 검사: 0 <= s_val <= s_max
            if valid and (0 <= s_val <= s_max):
                # Object n 값이 -1 <= n_val <= 1 범위에 있는지 확인
                if not (-1 <= n_val <= 1):
                    # 차선 밖이면 해당 state를 0으로 패딩
                    filtered_objects[i, j, :] = 0
            else:
                # valid==False 이거나 s 범위 밖이면 0 패딩
                filtered_objects[i, j, :] = 0

    return filtered_objects


def calculate_weight(initial_weight, final_weight, step, total_steps):
    """Linearly decrease the weight from initial_weight to final_weight over the steps."""
    return initial_weight - (initial_weight - final_weight) * (step / total_steps)

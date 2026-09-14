import numpy as np
import casadi as cs
import configparser
import os
import time
from MPC.NMPC_STM_acados_settings_delayed_kinematic_cat import acados_settings      
import matplotlib.pyplot as plt

from utils.trajectory_process import *
from MPC.fallback_controller import FallbackController
from MPC.understeer_gradient_compensator import UnderSteerGradientCompensator

# Vehicle State Index
IDX_VS_X = 0
IDX_VS_Y = 1
IDX_VS_YAW = 2
IDX_VS_SPEED = 3
IDX_VS_S = 4
IDX_VS_N = 5
IDX_VS_MU = 6

# MPC State Index
IDX_MPC_X = 0
IDX_MPC_Y = 1
IDX_MPC_YAW = 2
IDX_MPC_SPEED = 3
IDX_MPC_DELTA = 4
IDX_MPC_DELTA1 = 5
IDX_MPC_DELTA2 = 6
IDX_MPC_DELTA3 = 7
IDX_MPC_DELTA4 = 8
IDX_MPC_DELTA5 = 9
IDX_MPC_DELTA6 = 10
IDX_MPC_ACC = 11
IDX_MPC_A1 = 12
IDX_MPC_A2 = 13
IDX_MPC_A3 = 14
IDX_MPC_A4 = 15
IDX_MPC_A5 = 16
IDX_MPC_A6 = 17
IDX_MPC_S = 18

class Nonlinear_Delayed_Kinematic_Cat_Model_Predictive_Controller:

    def __init__(self):
        self.loader = WeightLoader("weights.ini")
        self.weights, self.weights_e, self.constraints, self.general = self.loader.get_weights_ini()
        general = self.general
        self.use_uniform_discretization = general['use_uniform_discretization']
        self.TF_MPC = float(general['fianl_time'])
        if self.use_uniform_discretization:
            self.Ts_MPC = float(general['uniform_mpc_dt'])
            self.N_MPC = int(self.TF_MPC / self.Ts_MPC)
            self.shooting_nodes = np.zeros(self.N_MPC+1)
            for i in range(1, self.N_MPC+1):
                self.shooting_nodes[i] = self.shooting_nodes[i-1] + self.Ts_MPC
            print(f"N_MPC: {self.N_MPC}, TF_MPC: {self.TF_MPC}, Ts_MPC: {self.Ts_MPC}")
        else:
            self.N_MPC = int(general['non_uniform_mpc_steps'])
            self.N_small_dt_steps = int(general['non_uniform_small_dt_steps'])
            self.small_dt = float(general['non_uniform_small_dt'])
            self.shooting_nodes = np.zeros(self.N_MPC+1)
            for i in range(1, self.N_small_dt_steps + 1):
                self.shooting_nodes[i] = self.shooting_nodes[i-1] + self.small_dt
                print(f"i: {i}, shooting_nodes: {self.shooting_nodes[i]}")

            remain_dt = (self.TF_MPC - self.shooting_nodes[self.N_small_dt_steps]) / float(self.N_MPC - self.N_small_dt_steps)

            for i in range(self.N_small_dt_steps+1, self.N_MPC+1):
                self.shooting_nodes[i] = self.shooting_nodes[i-1] + remain_dt
                print(f"i: {i}, shooting_nodes: {self.shooting_nodes[i]}")

            # 최종 시간(Tf)에 맞게 스케일링
            self.shooting_nodes = self.shooting_nodes * (self.TF_MPC / self.shooting_nodes[-1])
            self.Ts_MPC = self.shooting_nodes[1]
            print(f"N_MPC: {self.N_MPC}, TF_MPC: {self.TF_MPC}, initial Ts_MPC: {self.small_dt}, long Ts_MPC: {remain_dt}, final Ts_MPC: {self.shooting_nodes[-1]}, dead_time: {self.shooting_nodes[4]}")


        ## --- Load MPC formulation ---
        self.acados_solver, self.n_states, self.n_inputs = acados_settings(True, self.N_MPC, self.TF_MPC, general['ego_frame_rear'], self.shooting_nodes)
        self.is_MPC_init = True
        self.last_operation_time = None
        self.pred_X = None
        self.X_init = None
        self.pred_U = None
        self.prev_s = np.linspace(0, 10, self.N_MPC + 1).reshape(-1, 1)
        self.pprev_s = None
        self.prev_lag_tau = None
        self.is_vis_init = False

        
        self.start_time = time.time()
        self.output_idx = 0
        self.last_mpc_status = None
        self.target_cart = None
        self.target_pred = None
        # Fallback controller 관련 변수 초기화
        self.fallback_controller = FallbackController()
        self.stats = None
        self.fallback_count = 0

        # Understeer gradient compensator
        self.understeer_compensator = UnderSteerGradientCompensator(understeer_gradient=general['understeer_gradient'])
        return
    
    def association_s_distance(self, source_s, target_s):
        """
        두 s 배열 간의 연관성(매핑) 찾기
        source_s의 각 값에 대해 가장 가까운 target_s의 인덱스와 보간 비율 계산
        
        Args:
            source_s (np.ndarray): 소스 s 값 배열
            target_s (np.ndarray): 타겟 s 값 배열 (오름차순 정렬 가정)
            
        Returns:
            tuple: (인덱스 배열, 보간 비율 배열)
        """
        indices = np.zeros(len(source_s), dtype=int)
        ratios = np.zeros(len(source_s))
        
        # 타겟 배열이 비어있는 경우 처리
        if len(target_s) == 0:
            return indices, ratios
        
        # 마지막으로 찾은 타겟 인덱스 (검색 최적화)
        last_index = 0
        
        for i, s in enumerate(source_s):
            # 경계 처리
            if s <= target_s[0]:
                indices[i] = 0
                ratios[i] = 0.0
                continue
            
            if s >= target_s[-1]:
                indices[i] = len(target_s) - 1
                ratios[i] = 1.0
                continue
            
            # last_index부터 검색 시작 (s 값이 오름차순이라는 가정)
            for j in range(last_index, len(target_s)):
                if target_s[j] >= s:
                    if j > 0:
                        indices[i] = j - 1
                        s0, s1 = target_s[j-1], target_s[j]
                        ratios[i] = (s - s0) / (s1 - s0)
                    else:
                        indices[i] = 0
                        ratios[i] = 0.0
                    
                    last_index = j
                    break
        
        return indices, ratios

    def interpolate_by_s(self, source_s, target_data, s_column_idx, replace_s=True):
        """
        s 값을 기준으로 데이터를 선형 보간하는 함수
        
        Args:
            source_s (np.ndarray): 보간할 s 값 배열 (새로운 s 좌표)
            target_data (np.ndarray): 보간 대상 데이터 배열 
            s_column_idx (int): target_data에서 s 값이 있는 열 인덱스
            replace_s (bool): True이면 보간 결과의 s 값을 source_s로 대체
            
        Returns:
            np.ndarray: 보간된 데이터 배열
        """
        # 대상 데이터에서 s 값 추출
        target_s = target_data[:, s_column_idx]
        
        # s 값 매핑 계산
        indices, ratios = self.association_s_distance(source_s, target_s)
        
        # 선형보간 수행: source_s에 해당하는 target_data 값 계산
        interp_data = np.zeros((len(source_s), target_data.shape[1]))
        
        for i in range(len(source_s)):
            idx = indices[i]
            ratio = ratios[i]
            
            # 경계 처리
            if idx >= len(target_s) - 1:
                interp_data[i] = target_data[-1]
                continue
            
            # 각 열에 대해 선형보간 수행
            for col in range(target_data.shape[1]):
                # s 값은 그대로 source_s 값 사용 (옵션에 따라)
                if col == s_column_idx and replace_s:
                    interp_data[i, col] = source_s[i]
                else:
                    # 선형보간: v = v1 + ratio * (v2 - v1)
                    interp_data[i, col] = target_data[idx, col] + ratio * (target_data[idx+1, col] - target_data[idx, col])
        
        return interp_data

    def estimate_initial_states(self, delta_time, prev_X, prev_U):
        '''
        delta_time: 지연 시간
        prev_X: 이전 상태 [s, n, mu, v, delta, delta1, delta2, delta3, acc, a1, a2, a3]
        prev_U: 이전 제어 [delta, acc]

        Returns:
            X0: 초기 상태 [s, n, mu, v, delta, delta1, delta2, delta3, acc, a1, a2, a3]
        '''
        """
        지연 시간을 고려하여 초기 상태를 추정하는 함수
        
        Args:
            delta_time (float): 지연 시간 (초)
            prev_X (np.ndarray): 이전 상태 배열 [s, n, mu, v, delta, delta1, delta2, delta3, delta4, delta5, delta6, acc, a1, a2, a3, a4, a5, a6]
            prev_U (np.ndarray): 이전 제어 입력 배열 [delta_dot, jerk]
            
        Returns:
            np.ndarray: 추정된 초기 상태 배열
        """
        # MPC 타임스텝 계산
        MPC_dt = self.Ts_MPC
        num_step = len(prev_X)
        
        # 지연 시간을 MPC 타임스텝으로 정규화
        delta_time_normalized = delta_time / MPC_dt
        
        # 전체 time horizon에 대한 예측 상태 초기화
        predicted_states = np.zeros_like(prev_X)
        
        # 각 time step에 대해 지연된 상태 계산
        for t in range(num_step):
            # 현재 time step에 해당하는 지연 시간 인덱스 계산
            current_delay_idx = delta_time_normalized + t
            
            if current_delay_idx < num_step - 1:  # 예측 범위 내에 있는 경우
                # 하한값과 상한값 인덱스 계산
                idx_lower = int(current_delay_idx)
                idx_upper = idx_lower + 1
                
                # 보간 가중치 계산
                alpha = current_delay_idx - idx_lower
                
                # 선형 보간을 통한 지연된 상태 계산
                predicted_states[t] = (1 - alpha) * prev_X[idx_lower] + alpha * prev_X[idx_upper]
            else:  # 예측 범위를 벗어난 경우 마지막 상태 사용
                predicted_states[t] = prev_X[-1]
        
        # 속도가 음수인 경우 처리 (최소값 설정)
        predicted_states[:, IDX_MPC_SPEED] = np.maximum(predicted_states[:, IDX_MPC_SPEED], 0.01)
        
        # 전체 time horizon에 대한 예측된 상태 반환
        return predicted_states

    def predict_initial_trajectory(self, vehicle_state, point, road_map, constraints):
        '''
        vehicle_state: 차량 상태 [x, y, yaw, speed, s, n, mu]
        point: 경로 포인트 [x, y, yaw, speed, s, n, mu] * N
        constraints: 제약 조건 (max_delta 등)

        Returns:
            pred_X: 예측된 상태 [s, n, mu, v, delta, delta1, delta2, delta3, acc, a1, a2, a3] * Time steps
        '''
        # 초기 상태 설정
        pred_X = np.zeros((self.N_MPC + 1, self.n_states))
        # Initial pred_X define
        pred_X[0, IDX_MPC_X] = 0.0
        pred_X[0, IDX_MPC_Y] = 0.0
        pred_X[0, IDX_MPC_YAW] = 0.0
        pred_X[0, IDX_MPC_SPEED] = vehicle_state[0, IDX_VS_SPEED]
        steering_angle = self.fallback_controller._stanley_control(vehicle_state, point, constraints)
        pred_X[0, IDX_MPC_DELTA] = steering_angle
        pred_X[0, IDX_MPC_DELTA1] = steering_angle
        pred_X[0, IDX_MPC_DELTA2] = steering_angle
        pred_X[0, IDX_MPC_DELTA3] = steering_angle
        pred_X[0, IDX_MPC_DELTA4] = steering_angle
        pred_X[0, IDX_MPC_DELTA5] = steering_angle
        pred_X[0, IDX_MPC_DELTA6] = steering_angle
        pred_X[0, IDX_MPC_S] = vehicle_state[0, IDX_VS_S]
        
        tmp_vehicle_state = vehicle_state.copy()
        v  = vehicle_state[0, IDX_VS_SPEED] if vehicle_state[0, IDX_VS_SPEED] > 3.0 else 3.0
        s = vehicle_state[0, IDX_VS_S]
        for i in range(1, self.N_MPC + 1):
            # 차량 상태 업데이트
            prev_steering_angle = pred_X[i-1, IDX_MPC_DELTA]
            s = s + v * self.Ts_MPC
            tmp_vehicle_state[0, IDX_VS_X] += v * np.cos(tmp_vehicle_state[0, IDX_VS_YAW]) * self.Ts_MPC
            tmp_vehicle_state[0, IDX_VS_Y] += v * np.sin(tmp_vehicle_state[0, IDX_VS_YAW]) * self.Ts_MPC
            tmp_vehicle_state[0, IDX_VS_YAW] += v * prev_steering_angle / 3.0 * self.Ts_MPC
            tmp_vehicle_state[0, IDX_VS_SPEED] = v
            tmp_vehicle_state[0, IDX_VS_S] = s
            tmp_vehicle_state[0, IDX_VS_N] = 0.0
            tmp_vehicle_state[0, IDX_VS_MU] = 0.0
            curr_steering_angle = self.fallback_controller._stanley_control(tmp_vehicle_state, point, constraints)

            # N 계산
            x = tmp_vehicle_state[0, IDX_VS_X]
            y = tmp_vehicle_state[0, IDX_VS_Y]
            yaw = tmp_vehicle_state[0, IDX_VS_YAW]
            speed = tmp_vehicle_state[0, IDX_VS_SPEED]
            tmp_point = np.array([[x, y, yaw, speed]])
            tmp_point = frenet_transform(road_map, tmp_point) # [x, y, yaw, speed, s, n]
            tmp_point = frenet_mu_transform(road_map, tmp_point, start_mu=tmp_vehicle_state[0, IDX_VS_MU]) # [x, y, yaw, speed, s, n, mu]

            # pred_X 업데이트
            pred_X[i, IDX_MPC_X] = x
            pred_X[i, IDX_MPC_Y] = y
            pred_X[i, IDX_MPC_YAW] = yaw
            pred_X[i, IDX_MPC_SPEED] = v
            pred_X[i, IDX_MPC_DELTA] = pred_X[i-1, IDX_MPC_DELTA1]
            pred_X[i, IDX_MPC_DELTA1] = pred_X[i-1, IDX_MPC_DELTA2]
            pred_X[i, IDX_MPC_DELTA2] = pred_X[i-1, IDX_MPC_DELTA3]
            pred_X[i, IDX_MPC_DELTA3] = pred_X[i-1, IDX_MPC_DELTA4]
            pred_X[i, IDX_MPC_DELTA4] = pred_X[i-1, IDX_MPC_DELTA5]
            pred_X[i, IDX_MPC_DELTA5] = pred_X[i-1, IDX_MPC_DELTA6]
            pred_X[i, IDX_MPC_DELTA6] = curr_steering_angle
            pred_X[i, IDX_MPC_ACC] = 0.0
            pred_X[i, IDX_MPC_A1] = 0.0
            pred_X[i, IDX_MPC_A2] = 0.0
            pred_X[i, IDX_MPC_A3] = 0.0
            pred_X[i, IDX_MPC_A4] = 0.0
            pred_X[i, IDX_MPC_A5] = 0.0
            pred_X[i, IDX_MPC_A6] = 0.0
            pred_X[i, IDX_MPC_S] = s
        return pred_X

    def calculate_optimal(self, road_map,
                                vehicle_state, # x,y,yaw,speed,s,n,mu
                                objects,       # x,y,yaw,id,length,width,height,valid,s,n
                                point,         # x,y,yaw,speed,s,n,mu
                                measured_state
                            ):
        # Load weights
        try:
            self.weights, self.weights_e, self.constraints, self.general = self.loader.get_weights_ini()
            weights = self.weights
            weights_e = self.weights_e
            constraints = self.constraints
            general = self.general
        except Exception as e:
            print(f"Error loading weights: {e}")
            if not hasattr(self, 'weights'):
                return
        if self.fallback_count == 0 and (self.last_mpc_status == 'success' and not self.general['use_prediction_output']):
            self.start_time = time.time()

        # MPC 파라미터 설정
        num_step = self.N_MPC + 1

        # 이전 예측치 저장 공간 생성
        if self.pred_X is None or self.is_MPC_init:
            self.pred_X = np.zeros(( num_step, self.n_states)) # initialize empty array with shape (N,13)
            self.pred_U = np.zeros(( num_step, self.n_inputs))
            self.X_init = np.zeros(( num_step, self.n_states))
        # MPC 초기 상태 설정
        if self.is_MPC_init: # 초기 상태라면 MPC를 차량 상태로 초기화
            # X0 = [x, y, yaw, speed, delta, delta1, delta2, delta3, delta4, delta5, delta6, acc, a1, a2, a3, a4, a5, a6, s]
            v  = vehicle_state[0, IDX_VS_SPEED] if vehicle_state[0, IDX_VS_SPEED] > 0 else 0.01
            self.X_init = self.predict_initial_trajectory(vehicle_state, point, road_map, constraints)
            self.X_init[:, IDX_MPC_DELTA:IDX_MPC_ACC] = self.pred_X[:, IDX_MPC_DELTA:IDX_MPC_ACC]
            X0 = self.X_init[0, :]
            print("MPC get init")
            print(f"Vehicle state: {vehicle_state[0, IDX_VS_S]:.3f}, {vehicle_state[0, IDX_VS_N]:.3f}, {vehicle_state[0, IDX_VS_MU]:.3f}, {vehicle_state[0, IDX_VS_SPEED]:.3f}")
            self.last_operation_time = time.time()
        else: # 초기 상태가 아니라면 이전 최적 제어값에서 시작
            # X0 = [x, y, yaw, speed, delta, delta1, delta2, delta3, delta4, delta5, delta6, acc, a1, a2, a3, a4, a5, a6, s]
            current_time = time.time()
            delta_time = current_time - self.last_operation_time
            self.last_operation_time = current_time

            # 초기 상태 추정
            self.X_init = self.estimate_initial_states(delta_time, self.pred_X, self.pred_U)

            MPC_dt = self.Ts_MPC
            
            # 지연 시간 조정을 위한 빠른 계산
            delta_time_normalized = delta_time / MPC_dt
            delta_index_lower = max(0, min(int(delta_time_normalized), num_step-2))
            delta_index_upper = max(1, min(delta_index_lower+1, num_step-1))
            
            # 보간 가중치
            alpha = delta_time_normalized - delta_index_lower
            # 선형 보간을 통한 지연된 상태 계산 - 벡터화된 연산으로 최적화
            one_minus_alpha = 1 - alpha
            indices = [IDX_MPC_DELTA, IDX_MPC_DELTA1, IDX_MPC_DELTA2, IDX_MPC_DELTA3, 
                      IDX_MPC_DELTA4, IDX_MPC_DELTA5, IDX_MPC_DELTA6, IDX_MPC_ACC, IDX_MPC_A1, IDX_MPC_A2, IDX_MPC_A3, IDX_MPC_A4, IDX_MPC_A5, IDX_MPC_A6]
            
            interpolated_values = one_minus_alpha * self.pred_X[delta_index_lower, indices] + \
                                 alpha * self.pred_X[delta_index_upper, indices]

            # 속도 처리
            v = vehicle_state[0, IDX_VS_SPEED] if vehicle_state[0, IDX_VS_SPEED] > 0 else 0.01

            X0 = np.array([0.0,  # x
                           0.0,  # y
                           0.0,  # yaw
                           v,                          # speed
                           interpolated_values[0],     # delta
                           interpolated_values[1],     # delta1
                           interpolated_values[2],     # delta2
                           interpolated_values[3],     # delta3
                           interpolated_values[4],     # delta4
                           interpolated_values[5],     # delta5
                           interpolated_values[6],     # delta6
                           interpolated_values[7],     # acc
                           interpolated_values[8],     # a1
                           interpolated_values[9],     # a2
                           interpolated_values[10],    # a3
                           interpolated_values[11],    # a4
                           interpolated_values[12],    # a5
                           interpolated_values[13],    # a6
                           vehicle_state[0,IDX_VS_S]])   # s

        # point 선형보간 (기준 경로를 prev_s 위치에 맞게 보간)
        s_for_point = self.prev_s
        point = self.interpolate_by_s(s_for_point, point, IDX_VS_S)
        s_list = self.prev_s.flatten().tolist()
        kappas = road_map.GetCurvature(s_list)

        # Point 속도 프로파일 계산
        ## 커버쳐 기반 속도 프로파일 계산
        for i in range(0, num_step):
            # 비선형 제약조건 설정 [lane_boundary, ay, jy, distance_to_front_obj, friction_circle_constraint]
            kappa_for_max_lat_acc = max(1e-5, abs(kappas[i]))
            v_lim_from_lat_acc = np.sqrt(constraints['ay_max'] / kappa_for_max_lat_acc)
            point[i, IDX_VS_SPEED] = min(v_lim_from_lat_acc, point[i, IDX_VS_SPEED])
        ## 순방향 가속도 제한
        max_acc = constraints['ax_max']
        min_acc = constraints['ax_min']
        dt = self.Ts_MPC
        for i in range(0, num_step-2):
            curr_v = point[i, IDX_VS_SPEED]
            next_speed = curr_v + max_acc * dt
            point[i+1, IDX_VS_SPEED] = min(next_speed, point[i+1, IDX_VS_SPEED])
        ## 역방향 가속도 제한
        for i in range(num_step-1, 1, -1):
            curr_v = point[i, IDX_VS_SPEED]
            prev_speed = curr_v + min_acc * (-dt)
            point[i-1, IDX_VS_SPEED] = min(prev_speed, point[i-1, IDX_VS_SPEED])
                
        # 선행차량 위치 계산
        bumpers = compute_object_gap_positions(objects, extra_gap = constraints['bumper_extra_gap'])

        # 상태 제약조건 설정 [Velocity, Steering]
        lbx = np.array([constraints['min_speed'], -constraints['max_delta']])
        ubx = np.array([constraints['max_speed'], +constraints['max_delta']])
        
        

        # ---- Solver 파라미터 설정 ----
        # self.acados_solver.reset()
        for i in range(num_step):
            delta_w = weights['delta_w']
            accel_w = weights['accel_w']

            # 제어 제약조건 설정 [Steering acceleration, Jerk]
            jx_min = constraints['jx_min']
            cur_acc = self.pred_X[i, IDX_MPC_ACC]
            if cur_acc < -1.0: # Deceleration
                jx_max = -jx_min # In case of deceleration, jx_max has same amount of jx_min
            else: # Acceleration
                jx_max = constraints['jx_max']
            lbu = np.array([-constraints['max_ddelta'], jx_min]) # Nagative jerk is not bounded for safty
            ubu = np.array([+constraints['max_ddelta'], jx_max])

            # 비선형 제약조건 설정 [lane_boundary, ay, jy, distance_to_front_obj, friction_circle_constraint]
            kappa_for_max_lat_acc = max(1e-5, abs(kappas[i]))
            v_lim_from_lat_acc = np.sqrt(constraints['ay_max'] / kappa_for_max_lat_acc)
            v_lim = min(v_lim_from_lat_acc, point[i, IDX_VS_SPEED])
            lh = np.array([
                -constraints['lane_safety_gap'], # lane_boundary
                -v_lim, #-constraints['ay_max'], # ay
                -constraints['jy_max'], # jy
                0.0, # distance_to_front_obj
                -1.0, # friction_circle_constraint
                ]) 
            uh = np.array([
                constraints['lane_safety_gap'], # lane_boundary
                v_lim, #constraints['ay_max'], # ay
                constraints['jy_max'], # jy
                1e9, # distance_to_front_obj
                constraints['super_ellipse_radius'], # friction_circle_constraint
                ]) 
            lh_e = np.array([
                -constraints['lane_safety_gap'], # lane_boundary
                -v_lim, #constraints['ay_max'], # ay
                0.0, # distance_to_front_obj
                -1.0 # friction_circle_constraint
                ]) # 최종 제약조건은 jy 조건 없음
            uh_e = np.array([
                constraints['lane_safety_gap'], # lane_boundary
                v_lim, #constraints['ay_max'], # ay
                1e9, # distance_to_front_obj
                constraints['super_ellipse_radius'] # friction_circle_constraint
                ])  # 최종 제약조건은 jy 조건 없음


            if i == 0: # 초기 상태 설정
                if self.use_uniform_discretization:
                    dt = self.Ts_MPC
                else:
                    dt = self.shooting_nodes[i+1] - self.shooting_nodes[i] # len(shooting_nodes) = N_MPC + 1
                p = np.array([point[i, IDX_VS_X], point[i, IDX_VS_Y], point[i, IDX_VS_YAW], point[i, IDX_VS_SPEED],
                              constraints['ax_max'], constraints['ax_min'], constraints['ay_max'], constraints['super_ellipse_exponent'],
                              kappas[i],
                              weights['s_w'], weights['d_w'], weights['d_dot_w'], weights['mu_w'], weights['speed_w'], 
                              delta_w, accel_w, weights['ddelta_w'], weights['jerk_w'], weights['obj_w'], weights['dddelta_w'],
                              *bumpers[:, i, :].flatten(), constraints['thw'],
                              dt,
                              general['lag_tau']
                             ])
                # 초기 상태 설정
                # DEBUG hook — set env SCC_DEBUG_X0=1 to inspect X0 first time
                import os as _os
                # debug hook removed for clean output
                self.acados_solver.constraints_set(i, "lbx", X0)
                self.acados_solver.constraints_set(i, "ubx", X0)
                self.acados_solver.set(i, "x", X0)
                self.acados_solver.constraints_set(i, "lbu", lbu) 
                self.acados_solver.constraints_set(i, "ubu", ubu) 

            elif i == num_step - 1: # 마지막 상태 설정
                if self.use_uniform_discretization:
                    dt = self.Ts_MPC
                else:
                    dt = self.shooting_nodes[i] - self.shooting_nodes[i-1] # len(shooting_nodes) = N_MPC + 1
                p = np.array([point[i, IDX_VS_X], point[i, IDX_VS_Y], point[i, IDX_VS_YAW], point[i, IDX_VS_SPEED],
                              constraints['ax_max'], constraints['ax_min'], constraints['ay_max'], constraints['super_ellipse_exponent'],
                              kappas[i-1],
                              weights_e['s_w'], weights_e['d_w'], weights_e['d_dot_w'], weights_e['mu_w'], weights_e['speed_w'], 
                              weights_e['delta_w'], weights_e['accel_w'], weights_e['ddelta_w'], weights_e['jerk_w'], weights_e['obj_w'], weights_e['dddelta_w'],
                              *bumpers[:, i, :].flatten(), constraints['thw'],
                              dt,
                              general['lag_tau']
                             ])
                
                self.acados_solver.constraints_set(i, "lbx", lbx)
                self.acados_solver.constraints_set(i, "ubx", ubx)
                self.acados_solver.constraints_set(i, "lh", lh_e)
                self.acados_solver.constraints_set(i, "uh", uh_e)
            else: # 중간 상태 설정
                if self.use_uniform_discretization:
                    dt = self.Ts_MPC
                else:
                    dt = self.shooting_nodes[i+1] - self.shooting_nodes[i] # len(shooting_nodes) = N_MPC + 1
                p = np.array([point[i,IDX_VS_X], point[i, IDX_VS_Y], point[i, IDX_VS_YAW], point[i, IDX_VS_SPEED],
                              constraints['ax_max'], constraints['ax_min'], constraints['ay_max'], constraints['super_ellipse_exponent'],
                              kappas[i],
                              weights['s_w'], weights['d_w'], weights['d_dot_w'], weights['mu_w'], weights['speed_w'], 
                              delta_w, accel_w, weights['ddelta_w'], weights['jerk_w'], weights['obj_w'], weights['dddelta_w'],
                              *bumpers[:, i, :].flatten(), constraints['thw'],
                              dt,
                              general['lag_tau']
                             ])
                self.acados_solver.constraints_set(i, "lbx", lbx)
                self.acados_solver.constraints_set(i, "ubx", ubx)
                self.acados_solver.constraints_set(i, "lh", lh)
                self.acados_solver.constraints_set(i, "uh", uh)
                self.acados_solver.constraints_set(i, "lbu", lbu) 
                self.acados_solver.constraints_set(i, "ubu", ubu) 
            self.acados_solver.set(i, "p", p)
            if self.is_MPC_init:
                if i != 0:
                    self.acados_solver.set(i, "x", self.pred_X[i, :])

        solve_time_start = time.time()      

        status = self.acados_solver.solve()
        
        ## Solver status
        # 0: Success (ACADOS_SUCCESS)
        # 1: NaN detected (ACADOS_NAN_DETECTED)
        # 2: Maximum number of iterations reached (ACADOS_MAXITER)
        # 3: Minimum step size reached (ACADOS_MINSTEP)
        # 4: QP solver failed (ACADOS_QP_FAILURE)
        # 5: Solver created (ACADOS_READY)
        # 6: Problem unbounded (ACADOS_UNBOUNDED)
        # 7: Solver timeout (ACADOS_TIMEOUT)
        status_str = "Success" if (status == 0) else "NaN detected" if (status == 1) else "Maximum number of iterations reached" if (status == 2) else "Minimum step size reached" if (status == 3) else "QP solver failed" if (status == 4) else "Solver created" if (status == 5) else "Problem unbounded" if (status == 6) else "Solver timeout"
        if (status == 0 or status == 2 or status == 3):
            self.is_MPC_init = False
            self.last_mpc_status = 'success'
            for j in range(num_step):
                x = self.acados_solver.get(j,"x")
                x = np.array(x)
                if general['use_prediction_output'] == 0 and self.fallback_count == 0:
                    self.pred_X[j, :] = x
                elif self.output_idx >= self.N_MPC:
                    self.pred_X[j, :] = x
                    self.start_time = time.time()
                # Print the predicted state: time: s, n, mu, v, delta, delta1, delta2, delta3, acc, a1, a2, a3
                if j != num_step - 1:
                    u = self.acados_solver.get(j,"u") 
                    u = np.array(u)
                    self.pred_U[j, :] = u
                else:
                    self.pred_U[j, :] = self.pred_U[j-1, :]
                if(general['print_states'] == 1 and j < num_step):
                    print(f"t_idx: {j}, \ts: {x[IDX_MPC_S]:.3f}, \tx: {x[IDX_MPC_X]:.3f}, \ty: {x[IDX_MPC_Y]:.3f}, \tyaw: {x[IDX_MPC_YAW]:.3f}, \tv: {x[IDX_MPC_SPEED]:.3f}, \tdelta: {np.rad2deg(x[IDX_MPC_DELTA]):+.3f}, \tdelta1: {np.rad2deg(x[IDX_MPC_DELTA1]):+.3f}, \tdelta3: {np.rad2deg(x[IDX_MPC_DELTA3]):+.3f}, \tdelta6: {np.rad2deg(x[IDX_MPC_DELTA6]):+.3f}, \tacc: {x[IDX_MPC_ACC]:+.3f}, \ta1: {x[IDX_MPC_A1]:+.3f}, \ta3: {x[IDX_MPC_DELTA3]:+.3f}, \ta6: {x[IDX_MPC_A6]:+.3f}, \tddelta: {np.rad2deg(u[0]):+.3f}, \tjerk: {u[1]:+.3f} ")
        else:
            self.last_mpc_status = 'failed'
            self.acados_solver.reset()
            self.is_MPC_init = True
            print("MPC not solved, reset")
            print(f"status: {status_str}")
        
        # MPC Solver stats: [Cost, Time, Iter, QP Iter, Status, Solve Time]
        stats  = np.zeros(6)
        stats[0] = self.acados_solver.get_cost()
        stats[1] = 1000.0*self.acados_solver.get_stats('time_tot')
        stats[2] = self.acados_solver.get_stats('nlp_iter')
        stats[3] = np.max(self.acados_solver.get_stats('qp_iter'))
        stats[4] = status
        solve_time_end = time.time()
        stats[5] = 1000.0*(solve_time_end-solve_time_start)

        # Print stats
        if(general['print'] == 1):
            # print(f"Cost: {stats[0]:>7.3f}, Time: {stats[1]:>7.3f}ms, n-it: {stats[2]:>7.3f}, QP it: {stats[3]:>7.3f}, Stat: {stats[4]}, toT: {stats[5]:>7.3f}ms", end="\r")
            print(f"Cost: {stats[0]:>7.3f}, Time: {stats[1]:>5.3f}ms, QP it: {stats[3]}, Stat: {stats[4]}        ", end="\r")
        
        # Evaluate Local minimum: Compare trajectory and predicted trajectory
        # 마지막 predicted trajectory의 s가 -5이하이면 Fail
        if self.fallback_count > 0 or not self.last_mpc_status == 'success':
            print(f"\n{self.fallback_count} Fail: Local minimum, switching to fallback controller")
            self.fallback_count += 1
            self.prev_s = self.pprev_s
            
            # Fallback controller는 발동되려면 10회 이상 발동
            if self.fallback_count >= 10 and not self.last_mpc_status == 'success':
                self.fallback_count = 0
                # Fallback controller 초기화 및 실행
                steering_angle, torque = self.fallback_controller.calculate_optimal(vehicle_state, point, *bumpers[:, 0, 0].flatten(), constraints)
                    
                # Fallback controller 결과를 target_input으로 설정
                target_steer = np.rad2deg(steering_angle)
                target_input = [target_steer, torque, torque / (2300.0 * 0.36)]
                
                # 이전 값이 없는 경우 기본값 설정
                if self.target_cart is None:
                    self.target_cart = np.zeros((num_step, 6))
                if self.target_pred is None:
                    self.target_pred = np.zeros((num_step, 8))
                if self.stats is None:
                    self.stats = np.zeros(6)
                final_s = vehicle_state[0, IDX_VS_SPEED] * self.TF_MPC
                self.prev_s = np.linspace(0, final_s, self.N_MPC + 1).reshape(-1, 1)
                return self.target_cart, self.target_pred, target_input, self.stats, None
            elif self.fallback_count >= 20:
                self.fallback_count = 0
                
        else:
            self.fallback_count = 0

        # 정상적인 MPC 실행
        x_arr = self.pred_X[:, IDX_MPC_X]
        y_arr = self.pred_X[:, IDX_MPC_Y]
        yaw_arr = self.pred_X[:, IDX_MPC_YAW]
        target_cart = np.array([x_arr, y_arr, yaw_arr])
        target_cart = np.hstack((target_cart.T, self.pred_X[:, IDX_MPC_SPEED:IDX_MPC_ACC]))
        
        target_pred = np.hstack((self.pred_X, self.pred_U)) 

        # 현재 값을 저장
        self.target_cart = target_cart
        self.target_pred = target_pred
        self.stats = stats

        self.pprev_s  = self.prev_s
        self.prev_s   = self.pred_X[:, IDX_MPC_S]
        
        # 타겟 가속도 및 토크 계산
        output_time = time.time() - self.start_time
        output_time_position = output_time / self.Ts_MPC
        output_time_idx = int(output_time_position)
        if self.fallback_count > 0 or general['use_prediction_output']:
            print(f"\noutput_time: {output_time:.3f}s, output_time_idx: {output_time_idx}")

        future_select_idx = int(general['future_select_idx'])
        self.output_idx = min(output_time_idx + future_select_idx, self.N_MPC)
        
        mass_of_ioniq5 = 2100.0
        wheel_radius = 0.36
        target_accel = self.pred_X[self.output_idx, IDX_MPC_A6]
        target_torque = target_accel*(mass_of_ioniq5)*(wheel_radius)        
        target_steer = np.rad2deg(self.pred_X[self.output_idx, IDX_MPC_DELTA6])

        # 언더스티어 그래디언트 보상
        if general['understeer_gradient'] != 0.0:
            self.understeer_compensator.set_parameters(understeer_gradient=general['understeer_gradient'], lpf_alpha=general['lpf_alpha'])
            target_steer = self.understeer_compensator.compensate(self.pred_X[self.output_idx, IDX_MPC_SPEED], target_steer, current_time=0.0)
        
        target_input = [target_steer, target_torque, target_accel]
        
        control_io = {}
        control_io['steer'] = np.rad2deg(self.pred_X[:, IDX_MPC_DELTA6])
        control_io['acceleration'] = self.pred_X[:, IDX_MPC_A6]

        return target_cart, target_pred, target_input, stats, control_io

class WeightLoader:
    def __init__(self, file_name="weights.ini"):
        # 같은 디렉토리에 있는 파일 경로 설정
        self.ini_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
        self.ini_config = configparser.ConfigParser()
        
        self.last_modified_time = None
        self.weights = {}
        self.weights_e = {}
        self.constraints = {}

        self._load_weights_ini()

    def _load_weights_ini(self):
        # UTF-8-BOM 인코딩으로 파일 읽기
        with open(self.ini_file_path, 'r', encoding='utf-8-sig') as f:
            self.ini_config.read_file(f)
        
        self.general = {}
        self.weights = {}
        self.weights_e = {}
        self.constraints = {}
        
        for key, value in self.ini_config.items('general'):
            self.general[key] = float(value)
        for key, value in self.ini_config.items('weights'):
            self.weights[key] = float(value)
        for key, value in self.ini_config.items('weights_e'):
            self.weights_e[key] = float(value)
        for key, value in self.ini_config.items('constraints'):            self.constraints[key] = float(value)
        
        self.last_modified_time = os.path.getmtime(self.ini_file_path)

    def get_weights_ini(self):
        current_modified_time = os.path.getmtime(self.ini_file_path)
        if current_modified_time != self.last_modified_time:
            self._load_weights_ini()
            print ("Weights loaded from ini file")
        return self.weights, self.weights_e, self.constraints, self.general

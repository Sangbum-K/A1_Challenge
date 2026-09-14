import numpy as np
from typing import Tuple, List
import math

class FallbackController:
    def __init__(self):
        # Pure Pursuit 파라미터
        self.min_look_ahead = 6.0  # 최소 전방주시거리
        self.look_ahead_gain = 0.5  # 속도에 따른 전방주시거리 게인

        # Stanley 제어 파라미터
        self.k_e = 0.5  # 횡방향 오차 계수
        
        # PID 파라미터
        self.vel_kp = 2.5
        self.vel_ki = 0.0
        self.vel_kd = 0.0
        self.max_i_error = 10
        self.ref_speed_lpf = 0.1

        self.ref_speed_gain = 1.5/5.0 # 차량 최대 가속도 1.5m/s^2
        self.time_headway = 3.0 # 전방 차량 시간 간격
        self.min_distance = 10.0 # 최소 차량 간격
        
        # 상태 변수
        self.sum_speed_error = 0.0
        self.prev_speed_error = 0.0
        
        # 차량 파라미터
        self.wheel_base = 3.0  # 휠베이스
        self.vehicle_mass = 2300.0  # 차량 질량
        self.wheel_radius = 0.321  # 휠 반경
        self.gravity = 9.81  # 중력가속도
        self.rolling_resistance = 0.015  # 구름저항계수
        self.air_density = 1.225  # 공기밀도
        self.front_area = 2.5  # 전면적
        self.drag_coeff = 0.3  # 공기저항계수

    def calculate_optimal(self, 
                         vehicle_state,  # x,y,yaw,speed,s,n,mu
                         point,          # x,y,yaw,speed,s,n,mu
                         bumpers,
                         constraints
                         ) -> Tuple[float, float]:
        """
        Fallback controller 계산
        Returns:
            Tuple[float, float]: (steering_angle, torque)
        """
            
        # point가 1차원 배열인 경우 2차원으로 변환
        if len(point.shape) == 1:
            point = point.reshape(1, -1)
            
        # 1. Pure Pursuit으로 조향각 계산
        # steering_angle = self._pure_pursuit(vehicle_state, point, constraints)
        steering_angle = self._stanley_control(vehicle_state, point, constraints)
        
        # 2. PID로 토크 계산
        torque = self._pid_control(vehicle_state, point, bumpers, constraints)

        print(f"steering_angle: {steering_angle}, torque: {torque}")
        return steering_angle, torque

    def _pure_pursuit(self, vehicle_state, point, constraints) -> float:
        """Pure Pursuit 알고리즘으로 조향각 계산"""
        # 전방주시거리 계산
        look_ahead_distance = self.min_look_ahead + self.look_ahead_gain * vehicle_state[0, 3]
        
        # 가장 가까운 경로점 찾기
        closest_idx = self._find_closest_point(vehicle_state, point)
        
        # 전방주시점 찾기
        look_ahead_point = self._find_look_ahead_point(point, closest_idx, look_ahead_distance)
        
        if look_ahead_point is None:
            return 0.0
        
        # 차량 기준 좌표계로 변환
        look_ahead_point_x_in_global = look_ahead_point[0] - vehicle_state[0, 0]
        look_ahead_point_y_in_global = look_ahead_point[1] - vehicle_state[0, 1]

        vehicle_yaw = vehicle_state[0, 2]
        l_x = math.cos(-vehicle_yaw) * look_ahead_point_x_in_global - math.sin(-vehicle_yaw) * look_ahead_point_y_in_global
        l_y = math.sin(-vehicle_yaw) * look_ahead_point_x_in_global + math.cos(-vehicle_yaw) * look_ahead_point_y_in_global

        # 조향각 계산
        ld = math.sqrt(l_x**2 + l_y**2)
        steering_angle = math.atan2(2.0 * self.wheel_base * l_y/ld, 
                                  ld)
        
        # 조향각 제약조건
        steering_angle = np.clip(steering_angle, 
                                 -constraints['max_delta'], 
                                 constraints['max_delta'])
        
        return steering_angle

    def _stanley_control(self, vehicle_state, point, constraints) -> float:
        """Stanley 제어로 조향각 계산"""
        """
        Stanley 제어 알고리즘으로 조향각 계산
        
        Parameters:
            vehicle_state: 차량 상태 [x, y, yaw, speed, ...]
            point: 경로 포인트 [x, y, yaw, speed, ...]
            constraints: 제약 조건 (max_delta 등)
            
        Returns:
            float: 계산된 조향각
        """
        # 가장 가까운 경로점 찾기
        closest_idx = self._find_closest_point(vehicle_state, point)
        
        # 현재 차량 상태
        x_current = vehicle_state[0, 0]
        y_current = vehicle_state[0, 1]
        yaw_current = vehicle_state[0, 2]
        v_current = max(vehicle_state[0, 3], 0.1)  # 0으로 나누기 방지
        
        # 가장 가까운 경로점
        x_target = point[closest_idx, 0]
        y_target = point[closest_idx, 1]
        yaw_path = point[closest_idx, 2]
        
        # 횡방향 오차 계산
        # 경로에 수직인 오차 계산
        dx = x_target - x_current
        dy = y_target - y_current
        
        # 차량 좌표계에서의 횡방향 오차
        e_y = -math.sin(yaw_current) * dx + math.cos(yaw_current) * dy
        
        # 경로와 차량의 방향 오차
        yaw_error = self._normalize_angle(yaw_path - yaw_current)
        
        # Stanley 제어 공식
        # 횡방향 오차에 따른 조향각
        crosstrack_term = math.atan2(self.k_e * e_y, v_current)
        
        # 최종 조향각 = 방향 오차 + 횡방향 오차 보정
        steering_angle = yaw_error + crosstrack_term
        
        # 조향각 제약조건 적용
        steering_angle = np.clip(steering_angle, 
                               -constraints['max_delta'], 
                               constraints['max_delta'])
        
        return steering_angle
    
    def _normalize_angle(self, angle):
        """
        각도를 -pi에서 pi 사이로 정규화
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _pid_control(self, vehicle_state, point, bumpers, constraints) -> float:
        """PID 제어로 토크 계산"""
        # 목표 속도 계산 (전방주시점의 속도)
        target_idx = min(10, len(point)-1)  # 전방 10개 포인트 중 하나 선택
        ref_speed = point[target_idx, 3]

        # 속도 제약조건
        ref_speed = min(ref_speed, constraints['max_speed'])
        
        # 현재 속도
        current_speed = vehicle_state[0, 3]

        # 전방 차량 속도
        vehicle_distance = vehicle_state[0, 4]
        rel_distance = bumpers - vehicle_distance - self.min_distance # 최소 차량 간격 고려
        if rel_distance > -self.min_distance:
            vehicle_speed = vehicle_state[0, 3]

            # 0으로 나누기 오류 방지
            if np.isscalar(vehicle_speed) and vehicle_speed > 0.001:
                rel_time = rel_distance / vehicle_speed
                
                # 배열 비교 오류 방지
                if np.isscalar(rel_time) and rel_time < self.time_headway:
                    # print(f"rel_distance: {rel_distance}, rel_time: {rel_time}, time_headway: {self.time_headway}")
                    ref_speed = (1-(self.time_headway - rel_time) / self.time_headway) * vehicle_state[0, 3]

        # 속도 오차
        speed_error = ref_speed - current_speed
        
        # PID 제어
        self.sum_speed_error += speed_error
        self.sum_speed_error = np.clip(self.sum_speed_error, 
                                     -self.max_i_error, 
                                     self.max_i_error)
        
        # PID 출력
        pid_output = (self.vel_kp * speed_error + 
                     self.vel_ki * self.sum_speed_error + 
                     self.vel_kd * (speed_error - self.prev_speed_error))
        
        # 저속 오버슈트 보상
        if abs(current_speed) < 40/3.6 and speed_error > 0:
            pid_output = 1.5 * speed_error
            
        # 가속도 계산
        control_accel = pid_output
        
        # 구름저항, 공기저항 고려한 토크 계산
        rolling_resistance = (self.vehicle_mass * self.gravity * 
                            self.rolling_resistance)
        air_resistance = (0.5 * self.air_density * self.front_area * 
                         self.drag_coeff * current_speed**2)
        
        control_force = (self.vehicle_mass * control_accel + 
                        rolling_resistance + 
                        air_resistance)
        
        # 토크로 변환
        control_torque = control_force * self.wheel_radius
        
        # print(f"current_speed: {current_speed}, ref_speed: {ref_speed}, speed_error: {speed_error}, pid_output: {pid_output}, control_torque: {control_torque}")
        self.prev_speed_error = speed_error
        return control_torque

    def _find_closest_point(self, vehicle_state, point) -> int:
        """가장 가까운 경로점 찾기"""
        min_distance = float('inf')
        closest_idx = 0

        vehicle_x = vehicle_state[0, 0]
        vehicle_y = vehicle_state[0, 1]
        
        for i in range(len(point)):
            point_x = point[i,0]
            point_y = point[i,1]
            distance = np.sqrt((vehicle_x - point_x)**2 + 
                             (vehicle_y - point_y)**2)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i

        return closest_idx

    def _find_look_ahead_point(self, point, 
                              closest_idx, 
                              look_ahead_distance):
        """전방주시점 찾기"""
        closest_x = point[closest_idx,0]
        closest_y = point[closest_idx,1]
        for i in range(closest_idx, len(point)):
            point_x = point[i,0]
            point_y = point[i,1]
            distance = np.sqrt((point_x - closest_x)**2 + 
                             (point_y - closest_y)**2)
            if distance >= look_ahead_distance:
                return point[i]
        return point[i]

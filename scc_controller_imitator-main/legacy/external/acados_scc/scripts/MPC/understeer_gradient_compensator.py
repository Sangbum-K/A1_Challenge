#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class UnderSteerGradientCompensator:
    """
    차량의 언더스티어 특성을 보상하는 클래스입니다.
    언더스티어 그래디언트와 차량 속도, 조향각을 입력받아 보상된 조향각을 계산합니다.
    """
    
    def __init__(self, understeer_gradient=0.0, wheelbase=3.0):
        """
        초기화 함수
        
        Args:
            understeer_gradient (float): 언더스티어 그래디언트 값 (deg/G)
            wheelbase (float): 차량 휠베이스 길이 (m)
        """
        self.understeer_gradient = understeer_gradient
        self.wheelbase = wheelbase
        
        # 저역 통과 필터(LPF) 파라미터
        self.lpf_alpha = 0.1
        self.prev_compensated_steer = 0.0
        
        # 조향각 변화율 제한 파라미터
        self.max_steer_rate = 10.0  # deg/s
        self.prev_time = 0.0
        
        # 속도에 따른 보상 게인
        self.kv = 1.0
        
    def set_parameters(self, understeer_gradient=None, lpf_alpha=None, max_steer_rate=None, kv=None):
        """
        파라미터 설정 함수
        
        Args:
            understeer_gradient (float, optional): 언더스티어 그래디언트 값
            lpf_alpha (float, optional): 저역 통과 필터 계수 (0~1)
            max_steer_rate (float, optional): 최대 조향각 변화율 (deg/s)
            kv (float, optional): 속도에 따른 보상 게인
        """
        if understeer_gradient is not None:
            self.understeer_gradient = understeer_gradient
        if lpf_alpha is not None:
            self.lpf_alpha = lpf_alpha
        if max_steer_rate is not None:
            self.max_steer_rate = max_steer_rate
        if kv is not None:
            self.kv = kv
    
    def calculate_lateral_acceleration(self, velocity, steering_angle):
        """
        횡가속도 계산 함수
        
        Args:
            velocity (float): 차량 속도 (m/s)
            steering_angle (float): 조향각 (deg)
            
        Returns:
            float: 횡가속도 (m/s^2)
        """
        # 단위 변환 (deg -> rad)
        steering_rad = np.deg2rad(steering_angle)
        
        # 아칸만 조향 모델에 따른 횡가속도 계산
        # ay = v^2 / R, R = L / tan(δ)
        if abs(steering_rad) < 1e-6:
            return 0.0
        
        turn_radius = self.wheelbase / np.tan(steering_rad)
        lateral_accel = (velocity ** 2) / turn_radius
        
        return lateral_accel
    
    def compensate(self, velocity, steering_angle, current_time=None):
        """
        언더스티어 보상 함수
        
        Args:
            velocity (float): 차량 속도 (m/s)
            steering_angle (float): 현재 조향각 (deg)
            current_time (float, optional): 현재 시간 (s), 변화율 제한에 사용
            
        Returns:
            float: 보상된 조향각 (deg)
        """
        # 속도가 매우 낮을 경우 보상 불필요
        if velocity < 0.5:
            return steering_angle
        
        # 단위 변환 (deg -> rad)
        steering_rad = np.deg2rad(steering_angle)
        
        # 차량의 회전 반경 계산
        if abs(steering_rad) < 1e-6:
            return steering_angle
        
        turn_radius = self.wheelbase / np.tan(steering_rad)
        
        # 아커만 각도 공식 적용
        # δf = L/R + Kv*vx²/R
        ideal_steering_rad = self.wheelbase / turn_radius
        understeer_term = np.deg2rad(self.understeer_gradient) * (velocity**2) / turn_radius
        compensated_steering_rad = ideal_steering_rad + understeer_term
        
        # 라디안에서 각도로 변환
        compensated_steer = np.rad2deg(compensated_steering_rad)
        
        # 저역 통과 필터 적용
        filtered_compensated_steer = (1 - self.lpf_alpha) * self.prev_compensated_steer + self.lpf_alpha * compensated_steer

        # 이전 값 저장
        self.prev_compensated_steer = filtered_compensated_steer
        
        return filtered_compensated_steer

    def reset(self):
        """
        상태 초기화 함수
        """
        self.prev_compensated_steer = 0.0
        self.prev_time = 0.0


# 사용 예시
if __name__ == "__main__":
    # 인스턴스 생성
    compensator = UnderSteerGradientCompensator(understeer_gradient=5.0)  # 5.0 deg/G의 언더스티어 그래디언트
    
    # 파라미터 설정
    compensator.set_parameters(lpf_alpha=0.5, kv=1.5)
    
    # 보상 테스트
    velocity = 20.0  # m/s (약 72 km/h)
    steering_angle = 10.0  # deg
    
    compensated_steer = compensator.compensate(velocity, steering_angle, current_time=0.0)
    print(f"원래 조향각: {steering_angle} deg")
    print(f"보상된 조향각: {compensated_steer} deg")
    
    # 시뮬레이션 예시
    print("\n간단한 시뮬레이션:")
    compensator.reset()
    
    for t in range(10):
        time = t * 0.1
        velocity = 15.0 + t * 0.5  # 속도 증가
        steering_angle = 5.0 + t * 0.5  # 조향각 증가
        
        compensated_steer = compensator.compensate(velocity, steering_angle, current_time=time)
        print(f"시간: {time:.1f}s, 속도: {velocity:.1f} m/s, 원래 조향각: {steering_angle:.1f} deg, 보상된 조향각: {compensated_steer:.1f} deg")

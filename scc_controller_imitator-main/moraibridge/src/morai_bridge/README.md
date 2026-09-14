# MORAI Bridge

MORAI 시뮬레이터와 통신하여 차량 및 객체 상태를 관리하는 ROS 패키지입니다.

## 노드 구성

1. Vehicle Driver Node
   - MORAI 시뮬레이터의 차량 상태를 구독하고 AutoHYU 메시지 형식으로 변환하여 발행
   - ROS 토픽 또는 UDP 직접 통신 지원
   - 실시간 좌표계 변환 (MORAI UTM -> WGS84 -> ENU)
   - 차량 CG 위치에서 차량 뒷축 기반 상태 (속도, 가속도) 정보 보정 가능
      - cg_position_x_from_rear: 차량 뒷축 위치 기준 차량 CG 위치 (m), 0 인경우 보정 없음
      - 각속도 효과를 고려한 속도/가속도 계산

2. Object Driver Node
   - MORAI 시뮬레이터의 객체 상태를 구독하고 AutoHYU 메시지 형식으로 변환하여 발행
   - 객체 정보 관리 및 좌표계 변환
   - 실시간 객체 추적 및 상태 업데이트

3. Vehicle Control Node
   - AutoHYU 제어 명령을 MORAI 시뮬레이터 형식으로 변환하여 전송
   - ROS 토픽 또는 UDP 직접 통신 지원
   - 조향각, 페달 입력 등 차량 제어 명령 처리

## 의존성

- ROS Noetic
- MORAI Simulator
- MORAI C++ Networks (>= 0.2.7)
   - git clone https://github.com/Onlyti/MORAI_CPP_Networks
   - 패키지 빌드시 자동 다운로드
- Lanelet2 (Core, IO, Projection, Routing, Traffic Rules)
- Eigen3
- Boost

## 설정 방법

### simulation.ini 파일 설정

1. [MoraiBridge] 섹션
   - communication_control_type: "ros" 또는 "udp" (차량 제어 통신 방식)
   - communication_pubsubser_type: "ros" 또는 "udp" (차량/객체 상태 구독 통신 방식)
   - communication_sim_type: "ros" 또는 "udp" (시뮬레이터 설정 통신 방식)
   - udp_source_ip: UDP 통신 시 소스 IP (시뮬레이터 IP)
   - udp_destination_ip: UDP 통신 시 목적지 IP (자율주행 시스템 IP)

2. [MoraiObjectDriver] 섹션
   - udp_object_source_port: 객체 정보 송신 포트
   - udp_object_destination_port: 객체 정보 수신 포트
   - udp_traffic_light_source_port: 신호등 정보 송신 포트
   - udp_traffic_light_destination_port: 신호등 정보 수신 포트
   - udp_morai_ros_republish: UDP 메시지의 ROS morai_msgs 토픽 ("/udp/Object_topic", "/udp/GetTrafficLightStatus") 재발행 여부 (true/false)

3. [MoraiVehicleControl] 섹션
   - control_type: 차량 제어 명령 타입 (1: 페달, 2: 속도, 3: 가속도)
   - max_accel: 최대 가속도 제한 (m/s^2)
   - max_decel: 최대 감속도 제한 (m/s^2)
   - udp_ego_ctl_source_port: 차량 제어 명령 송신 포트
   - udp_ego_ctl_destination_port: 차량 제어 명령 수신 포트
   - udp_morai_ros_multi_publish: ROS morai_msgs ("/ctrl_cmd") 토픽 동시 발행 여부 (true/false), UDP 통신 시 사용 활성화 되는 옵션
   - udp_morai_autonomous_mode: 차량 제어 명령 모드 (0: 수동, 1: 자율)

4. [MoraiVehicleDriver] 섹션
   - udp_vehicle_info_source_port: 차량 정보 송신 포트
   - udp_vehicle_info_destination_port: 차량 정보 수신 포트
   - udp_gps_source_port: GPS 정보 송신 포트
   - udp_gps_destination_port: GPS 정보 수신 포트
   - udp_morai_ros_republish: UDP 메시지의 ROS morai_msgs 토픽 ("/udp/Ego_topic") 재발행 여부 (true/false)
   - cg_position_x_from_rear: 후륜 축으로부터 CG까지의 거리 (m)

### ROS 파라미터 설정
- ref_lat: 기준점 위도 (도 단위)
- ref_lon: 기준점 경도 (도 단위)
- proj_mode: "utm" 또는 "local_cartesian" (좌표계 투영 방식)
- location: 맵 위치 ("kcity", "katri", "pangyo", "sangam", "deagu")
- task_period/period_vehicle_driver: 노드 실행 주기 (초 단위, 기본값: 1.0)

## 토픽 구조

### Vehicle Driver Node
Subscribe: (ROS 통신 시)
- /Ego_topic (morai_msgs/EgoVehicleStatus): 차량 상태 정보
- /gps (morai_msgs/GPSMessage): GPS 위치 정보

Publish:
- /app/loc/vehicle_state (autohyu_msgs/VehicleState): 변환된 차량 상태
- /app/loc/reference (autohyu_msgs/Reference): 좌표계 변환 기준점
- /udp/Ego_topic (morai_msgs/EgoVehicleStatus): UDP 메시지 ROS 재발행 (UDP 통신 및 republish 설정된 경우)

### Object Driver Node
Subscribe: (ROS 통신 시)
- /Object_topic (morai_msgs/ObjectStatusList): 주변 객체 정보

Publish:
- /app/loc/track_objects (autohyu_msgs/TrackObjects): 변환된 객체 정보
- /udp/Object_topic (morai_msgs/ObjectStatusList): UDP 메시지 ROS 재발행 (UDP 통신 및 republish 설정된 경우)

### Vehicle Control Node
Subscribe:
- /app/con/command_steer (std_msgs/Float32): 조향 제어 명령 [degree, -40 ~ 40]
- /app/con/command_accel (std_msgs/Float32): 가속 제어 명령 [m/s^2]
- /app/pla/target_speed (std_msgs/Float32): 목표 속도 제어 명령 [m/s]
- /app/con/command_pedal_accel (std_msgs/Float32): 페달 가속 제어 명령 [m/s^2, 0 ~ 1]
- /app/con/command_pedal_brake (std_msgs/Float32): 페달 감속 제어 명령 [m/s^2, 0 ~ 1]

Publish:
- /ctrl_cmd (morai_msgs/CtrlCmd): MORAI 형식 제어 명령 (ROS 통신 시)
- UDP: MORAI 형식 제어 명령 (UDP 통신 시)

## 실행 방법

1. 패키지 빌드
   ```bash
   catkin_make --only-pkg-with-deps morai_bridge
   ```

2. 전체 실행
   ```bash
   roslaunch morai_bridge morai_bridge.launch
   ```

3. 개별 노드 실행
   ```bash
   roslaunch morai_bridge vehicle_driver.launch
   roslaunch morai_bridge object_driver.launch
   roslaunch morai_bridge vehicle_control.launch
   ```

## 주의사항

1. UDP 통신
   - <span style="color: red">(**Linux 환경에서 1024 이하 포트 사용 시 포트 포워딩 또는 1024 이상의 포트로 설정 필요** !)</span>
   - 방화벽 설정 확인
      - ufw disable : 방화벽 비활성화
   - IP 주소 및 포트 번호 설정 확인
   - UDP 수신 타임아웃 설정 가능

2. 좌표계 변환
   - 정확한 좌표 변환을 위해 올바른 레퍼런스 포인트 설정 필요
   - 맵 변경 시 location 파라미터 업데이트 필요
   - UTM/WGS84/ENU 좌표계 간 변환 지원

3. 통신 방식 설정
   - simulation.ini 파일 [MoraiBridge] 섹션에서 통신 종류별 설정 가능
     - communication_control_type: 차량 제어 통신 방식 (ROS/UDP)
     - communication_pubsubser_type: 차량/객체 상태 통신 방식 (ROS/UDP) 
     - communication_sim_type: 시뮬레이터 설정 통신 방식 (ROS/UDP)
   - 실행 중 통신 방식 변경 시 자동으로 이전 통신 연결 해제
   - 새로운 통신 방식으로 재연결
   - 통신 방식 전환 시 메시지 손실 최소화

4. 차량 상태 정보
   - 모든 상태 정보는 후륜 중심 기준으로 계산
      - 모라이 제공 속도 및 가속도는 차량 CG 위치 기준
      - 차량 CG 위치 기준 속도/가속도 보정 필요
   - 각속도에 의한 회전 효과 보정 적용
   - 조향각은 타이어 각도와 스티어링 휠 각도 모두 제공

## 문제 해결

1. UDP 통신 문제
   - 포트 권한 확인
   - IP 주소 및 포트 번호 설정 확인
   - 방화벽 설정 확인
   - UDP 수신 타임아웃 설정 확인

## UDP 메시지 재발행 (Republish)

1. Vehicle Driver Node
   - udp_morai_ros_republish 옵션이 true일 경우:
     - UDP로 수신한 차량 상태 데이터를 MORAI 메시지 형식(/udp/Ego_topic)으로 ROS 토픽 발행
     - 원본 UDP 데이터의 형식을 유지하여 재발행
     - 디버깅 및 모니터링 목적으로 사용 가능

2. Object Driver Node
   - udp_morai_ros_republish 옵션이 true일 경우:
     - UDP로 수신한 객체 상태 데이터를 MORAI 메시지 형식(/udp/Object_topic)으로 ROS 토픽 발행
     - 원본 UDP 데이터의 형식을 유지하여 재발행
     - 객체 추적 성능 분석에 활용 가능

3. Vehicle Control Node
   - udp_morai_ros_multi_publish 옵션이 true일 경우:
     - UDP로 보내는 제어 명령을 ROS 토픽 (/ctl_cmd)로 발행
     - 제어 명령 실시간 확인 및 모니터링 목적, ROS 타입으로 활용 등으로 사용 가능

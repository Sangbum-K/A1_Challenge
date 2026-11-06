> 2026-11-06: 추적·대응·대상 선택이 교체됐다. 현재 동작은 [설계 문서](../../docs/superpowers/specs/2026-11-06-sensor-fusion-tracking-association-design.md)가 기준이고 이 문서는 이전 기록이다.

# LiDAR 주 대상 연속성 및 동일 ID P5 — 2026-11-04

## 이번 사용자 결정

- 가까운 유효 LiDAR 차량 후보 한 대만 P5에 사용. 먼 한 대는 대기 후보로 추적.
- 카메라는 존재/bin 확인 보조. 카메라 화면 이탈·미검출만으로 LiDAR 주 대상을 해제하지 않는다.
- 핵심은 대상의 순간 미검출 출력 단절과 재검출 ID/속도 초기화 해결.
- 기존 **raw P5 수식**은 유지. 별도의 속도 기반 제동거리식을 추가하지 않는다.
- 트랙 지도·주행라인·속도 프로파일 연동은 사용자의 요청으로 후속 작업으로 보류.

## 구현

### 추적과 수명

`cpp/src/association.cpp`의 CV Kalman 예측을 미검출 프레임의 박스 위치에도 반영한다.
마지막 관측 raw_box와 현재 예측 box/state를 구분한다. 미검출 때 예측 속도를 0으로 초기화하지 않는다.

| 설정 | 초기 시험값 | 의미 |
|---|---|---|
| track_coast_s | 0.3초 | 마지막 관측 이후 후보 출력 허용 시간 |
| track_memory_s | 0.8초 | 기존 트랙을 재연결용으로 내부 보존하는 시간 |
| track_max_position_variance_m2 | 1.0m² | XY 각 축 위치분산 상한. 초과 시 후보 사용 금지 |

3회 이상 관측된 트랙만 대상으로 한다. 확인된 기존 트랙과 대응 가능한 측정은,
대응이 모호하다는 이유만으로 즉시 새 ID를 만들지 않는다. 위치/크기/공분산 게이트를 만족하면
기존 필터 상태로 갱신한다. 대응이 불확실하면 예측 상태로 남기고, 기억시간이 지나면 삭제한다.
가까이 붙은 진짜 신규 차량의 ID 발급도 늦어질 수 있으므로 실차 교차/가림 시험이 필요하다.
장시간 소실 뒤에는 새 ID가 정상이며 무조건 과거 ID를 강제로 재사용하지 않는다.

`ClusterTrack` 출력에 measured, prediction_valid, observation_age_s,
position_covariance_xy를 추가했다. 예측 시 points=0, raw_center/raw_size는 마지막 실측값이다.
모든 시각은 Association.header 기준이며 실제 측정이 새로 들어온 것처럼 표시하지 않는다.

### 주 대상 / 대기 후보

Association에 primary_track_id, standby_track_id, primary_status를 추가했다.
가까운 실측 유효 후보가 주 대상이다. 기존 주 대상이 제한 시간 내 예측 중이면 유지하고,
카메라 대응 확률 또는 순간 미검출 때문에 먼 차량으로 즉시 바꾸지 않는다.
주 대상이 유효하지 않게 되면 가까운 실측 후보를 선택한다. 예측 트랙을 신규 주 대상으로 승격하지 않는다.
실측 상태에서 거리 순위가 바뀌면 가까운 대상을 선택하며 별도의 교체 지연은 이번에 추가하지 않았다.

신규 대상 탐색은 수평 중심거리10~50m다. 이미 선택한 주 대상은 10m 안으로 접근해도
추적을 끊지 않고 현재 처리 ROI 하한1m까지 유지한다. 이것은 P5 수식의 간격 항을 10m로
바꾸었다는 의미가 아니다. 근거리·경계 조건의 실측 검증은 미완료다.
현재 주행라인 연동 전이므로 관심영역은 기존 직선 corridor다. 지도/경로에 따른 선행차 확정은 미구현.

### P5 입력 연결

`shadow.yaml` 기본 mode를 `association`으로 변경했다.
`fusion_node`는 `/lead_fusion/association`의 **primary_track_id 하나만** 찾아 계산한다.
이 모드에서는 예전의 별도 LiDAR 군집/트래커 경로를 호출하지 않고 카메라 입력을 요구하지 않는다.
기존 camera/fusion 모드는 수동 선택용으로 남겨 두었다.

- 거리: 추적 박스의 전방 가까운 면 X − bumper_x_m. 군집 중심거리와 다르다.
- bumper_x_m는 이 모드에서 LiDAR 원점 기준 차량 전방 범퍼 위치이며 아직 실측값이 아니다.
- 상대속도: 겉보기 Vx − 자차 yaw_rate × 대상 Y. 평면 회전 항만 보정하며 scan deskew를 대신하지 않는다.
- raw P5: 기존 `p5(plan_speed, gap, lead_speed)` 그대로 사용.
- LiDAR 주 대상 모드의 gate는 1.0. 이전 카메라 logit gate/EMA가 매 프레임 제동 요청을 끊지 않도록 분리했다.
  따라서 raw 수식은 같지만 전체 요청의 gate 정책은 변경되었다. 요청 제한 −3~0m/s²는 유지한다.
- 후보 교체 시 다른 차량의 거리를 차분해서 속도를 구하지 않는다. 속도는 각 LiDAR 트랙 필터에서 온다.
- 최종 기존 제동값과의 min 결합 및 CAN 송신은 이번 구현에 포함하지 않는다. 출력은 shadow 추정이다.

P5 예측 유효기간은 `min(측정시각+max_age_s, 마지막관측시각+primary_coast_s)`다.
예측 메시지를 반복 발행해도 마지막 실측 기준 만료시각을 연장하지 않는다.
전송·계산 지연도 이 시간 안에 포함되므로 0.3초가 추가 통신 지연을 허용하는 값은 아니다.
현재 association에는 카메라 대응을 기다리는 약100ms 큐가 있어 실차 지연 측정 후 조정이 필요하다.

P5 소비자는 확인된 트랙, 관측 경과시간, 위치분산, 시각, 좌표계, ego 유효성 등을 다시 검사한다.
`primary_motion_verified: false`를 추가했고 미검증 운동 보상으로 유효 제동 요청을 만들지 않는다.
clock/extrinsics/IMU 검증도 계속 false/미완료다. 실물 유효 P5가 준비되었다는 뜻이 아니다.

### UI / 기록

- P = 주 대상, S = 대기 후보. `~`는 예측 상태.
- 빨강 = 현재 카메라+LiDAR 대응, 주황 = LiDAR 단독, 파랑 계열 = 단기 예측.
- P5 카드에는 현재 primary ID와 같은 lidar_primary 출력만 표시한다. 이전 대상의 P5와 새 박스를 섞지 않는다.
- CSV에 primary_id, standby_id, primary_status 추가. 진단 JSON에도 관측/예측 상태와 공분산 기록.
- 원커맨드 domain107 격리와 MP4 녹화는 유지한다. 이번에는 실제 센서를 재시작하지 않았다.

## 검증

- `cpp/evidence/continuity_core.txt`: C++ core_tests/association_tests 통과.
  움직이는 대상의 예측 위치·속도, 동일 ID 재연결, 만료, 모호 측정의 중복 ID 억제,
  주 대상 유지/대기 승격, raw P5 일치, 예측 유효기간 확인.
- `cpp/evidence/continuity_build.txt`: GPU 런타임 및 ROS 메시지 재빌드·설치 성공.
- `cpp/evidence/continuity_ros.txt`: 합성 점군의 연속 두 프레임 대상 누락 후 주 대상 유지/동일 ID 복귀,
  카메라 중단 중 LiDAR 주 대상 유지, 최대2대, 동일bin 모호성, stale UI/CSV 회귀 통과.
- `cpp/evidence/primary_p5_ros.txt`: 합성 INS + Association 입력으로 주 대상만 P5 계산,
  카메라 없음, 예측 만료, 재관측, ID 교체, 주 대상 없음/clock 미검증/입력 중단 거부 확인.
  검증 플래그를 켠 것은 합성 시험 프로세스뿐이며 실차 설정은 그대로 미검증이다.
- 초기 ROS 시험의 Python bytes→ROS 배열 설정이 느려 10Hz 대상 누락을 재현하지 못했다.
  array('B') 버퍼로 발행하도록 시험 코드를 수정했다. 이 결과를 실제 센서 처리량 측정으로 해석하지 않는다.

위는 코드·합성 검증이다. 실제 주행의 ID switch율, 검출률, 상대속도 오차, 제동 연속성은 미검증이다.
원거리 적응형 군집, 회전 박스, 정밀 시각/장착 보정, 실제 GNSS IMU 입력도 계속 후속 항목이다.

## 주행라인 활용: 설계 메모만, 연결 보류

로컬에서 발견한 후보 자료:

- `/home/ailab/git/Seunghwan/racingline/racingline_val2/yongin_raceline.csv`: s_m, x_m, y_m, yaw_rad, kappa_radpm, v_mps, ax_mps2.
- `/home/ailab/git/Yongin2_Map/raceline/raceline_metadata.json`: PlayerStart-origin ENU, 서로 다른 경로 버전/원점.
- racingline_val3와 main_pc_work의 여러 프로파일도 존재. 실제 차량의 선택된 버전/토픽은 확인하지 않았다.

추후 동일 map 좌표계의 localization pose와 경로가 확보되면:

1. LiDAR 트랙을 map으로 변환하고 경로에 투영해 진행거리 s와 횡방향 편차 d 계산.
2. 곡선 도로에서도 경로 앞쪽 대상/옆 차량/트랙 밖 물체를 구분. 가까운 대상 선택 시 직선거리와 경로 진행거리의 용도를 구분.
3. 트랙 경계는 배경 제거 근거로 사용하되 localization 오차 여유를 주고, 경로 밖 차량의 끼어들기를 무조건 제거하지 않음.
4. 속도 프로파일은 예상 도달시간·관심영역 길이·계획속도에 사용. 실제 자차속도나 상대 차량 속도의 대체값으로 사용하지 않음.
5. domain107에는 필요한 pose/path만 명시적으로 연결. 팀원 노드/TF 전체를 섞지 않음.

사용자 결정: 이 연동은 나중에 구현한다. 이번 변경은 지도·주행라인 노드를 추가하거나 외부 워크스페이스를 수정하지 않았다.

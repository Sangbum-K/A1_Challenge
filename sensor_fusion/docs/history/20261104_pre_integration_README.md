> **최신 변경 (2026-11-04): 카메라와 대응한 LiDAR 대상의 거리 추적.** 카메라와 일대일 대응을 3회 확인한 트랙만 거리 주 대상으로 선택한다. 가까운 LiDAR 단독 군집으로 주 대상을 바꾸지 않고 같은 ID를 유지한다. 대응 소실 시 최대 0.3초 유지/예측 표시 후 해제하며, 유지/예측·모호한 대응은 P5 유효값으로 사용하지 않는다. 얇은 관측 표면 지원과 트랙별 탈락/투영 진단 CSV 추가. 실제 12:04~12:10 점군 재처리와 보정 검증은 미완료이며, 현재 센서/UI는 중지 상태다. 아래 실행 및 LiDAR 독립 주 대상 설명은 이전 구현이다.

> **최신 상태 (2026-11-04):** 사용자 요청으로 카메라·GPU A₂·LiDAR bridge·fusion·association UI/녹화·domain107 ROS 데몬 종료 확인. 작업 공간 빌드 로그 747개(23,381,355바이트) 삭제. 아래 실행 중 표시는 과거 기록이다.

> **출력 정리 (2026-11-04):** 오늘 주행·녹화·진단 기록 보존. 이전 날짜 출력 26개 폴더 및 재생성 가능한 중간 파일/캐시 등 총 249개 파일(226,696,200바이트) 삭제. 과거 문서의 11월 3일 outputs 링크는 삭제된 기록일 수 있음. 원본 데이터 링크·모델·실행환경 유지. [삭제 목록](docs/cleanup/20261104_outputs.json).

> **최신 실행: 카메라+LiDAR 연결 복구 (2026-11-04).** `outputs/ui_20261104_120315`에서 UI/MP4 실행 중.
> 카메라21Hz·A₂21Hz, 이번10초 실측 공용 LiDAR 약5.8Hz/인지5.9Hz. 수신 간격 때문에 간헐 STALE 발생; 안정10Hz 검증 아님.
> 외부 localization/Hesai/GNSS 종료·재시작 없음. IP 경로 유지. bridge를 독립 source/sink 프로세스+로컬 socketpair로 분리해 domain0→107 전달 복구.
> [설정·검증 기록](docs/network/camera_restore_20261104/README.md). 아래 중복 입력 차단·종료 대기 기록은 과거 상태.

> **현재 연결 제한:** 외부 localization 두 실행의 `/lidar_points` 발행자2개를 감지해 우리 LiDAR 입력 전달 차단 중. 카메라/A₂는 수신 중. 외부 실행 종료는 사용자 확인 대기.

> **최신 실행 (2026-11-04): 카메라 복구 완료.** 카메라·GPU A₂·공용 LiDAR·UI·MP4 실행 중 (`outputs/ui_20261104_113405`).
> 기존 LiDAR/GNSS/공유/인터넷 프로필 보존. 카메라 전용 MACVLAN `a1cam0`만 추가. 자동 노출·게인 OFF, 노출3,000μs·게인0dB 설정.
> [네트워크 변경·보존 확인](docs/network/camera_restore_20261104/README.md). 아래 카메라 꺼짐 기록은 이전 상태.

> **카메라 수동 설정 (2026-11-04):** 사용자 요청으로 UI 실행기의 자동 노출·자동 게인 OFF. 노출3,000μs, 게인0dB. 다음 카메라 실행부터 적용하며 현재 카메라 미기동 상태는 유지.

> **현재 실행 (2026-11-04):** localization 공용 LiDAR → domain107 인지·UI·MP4 실행 중.
> 기존 LiDAR/GNSS 네트워크·설정·프로세스 유지. 카메라 포트가 바뀌어 **A1_CAMERA_ENABLED=0**으로 실행(카메라/A₂ 없음).
> [현재 설정·보존 확인·실행 명령](docs/network/current_before_shared_start/README.md). 출력: `outputs/ui_20261104_112823`.

> **최신 공용 LiDAR 연결:** [설계·명령어·검증](docs/SHARED_LIDAR_KR.md).
> 우리 원커맨드의 Hesai 드라이버 자동 기동 제거. 기본 domain0 `/lidar_points` → C++ 단방향 bridge → domain107 전용 입력.
> `./run_sensor_fusion.sh lidar-check`로 수신 검사. 입력 없으면 센서 자동 기동 없이 종료. 현재 공용 발행자 없음, UI 중지 유지.

> **최신 상태: 2026-11-04 종료 완료.** 사용자 요청으로 프로젝트 센서 드라이버·인지·UI·녹화 및 도메인107 조회 데몬 종료.
> 재생성 가능한 컴파일 객체·Python 캐시 2,297개(49,051,542바이트) 정리. 원시 데이터·MP4·모델·실행파일 보존.
> [정리 내역](docs/cleanup/20261104_shutdown.json). 아래 실행 중 표시는 이전 시점 기록이다.

> **최신 실차 기록 (2026-11-04):** [관측·수정·검증·미완료 항목 정리](docs/session_20261104/README.md).
> 거리 적응형 XY 군집 / 부분 관측 유지 / 처리·BEV100m / P5후보10~50m / 카메라진입0.4 적용.
> **옆모습 AABB 폭3.18m가 최대2.6m에 걸리는 문제 확인. OBB 수정은 아직 미완료.**
> 현재 `outputs/ui_20261104_101515`에서 실행·MP4 녹화 중. 아래 중지 상태 등은 이전 기록.

> **네트워크 최신 설정:** [센서 전용 영구 프로필](docs/NETWORK_KR.md). 2026-11-04 적용, 세 센서 ping 성공. 이후 인터넷 변경 시 센서 IP 재설정 불필요.

# 최신: LiDAR 연속 추적 / 동일 ID P5 — 2026-11-04

[추적 연속성 변경·검증](docs/TRACK_CONTINUITY_KR.md). 가까운 주 대상 1대만 P5, 먼 차량은 대기.
초기 coast0.3초/memory0.8초, 관측·예측 구분, 재연결 시 ID·속도 유지, 모호 측정의 중복 ID 억제.
기본 shadow mode=association으로 UI와 P5 대상 ID 연결. raw P5 수식 유지, 이 모드의 camera gate는1.
실차 IMU·보정·운동 검증은 미완료로 유효 제동값 미검증. CAN 송신 없음. 주행라인 연동은 사용자 요청으로 보류.
아래 ‘별도 P5 트래커’ 기록은 이전 구현이다. 현재 UI는 중지 상태.

# 최신 변경 — 2026-11-04

[작업 방향·코드 자체 점검](docs/SELF_REVIEW_20261104_KR.md): 후보 최대2대, 수평 중심거리 10~50m.
카메라 독립 bin 영역과 LiDAR 일대일 대응, 빨강=대응 후보/주황=LiDAR 단독.
P5 동일 ID 연결·실측 보정·원거리 군집 개선은 미완료. 아래는 이전 기록을 포함한다.

[현재 센서 설정·실행 환경 (2026-11-03)](SENSOR_ENVIRONMENT_KR.md) — 원커맨드, 네트워크, GPU, 보정 상태와 미완료 항목.

# C++ 카메라 · LiDAR · GNSS/INS → P5

[실험용 UI 실행 안내](docs/UI_KR.md) · 실행: `bash run_ui.sh`

**현재 우선 목표:** 카메라 bin과 LiDAR 군집이 같은 대상을 가리키는지 검증하는 것.
최종 거리/상대속도는 LiDAR만 사용한다. F2 연결은 대상 대응 검증 이후다.
GPU 실카메라 60초 시험: 1,260장 입력/출력, **20.997Hz**, history_ready **100%**.
[대상 대응·군집 안정화](docs/ASSOCIATION_KR.md) · [GPU 실행 안내](docs/GPU_RUNTIME_KR.md) · [최종 F2 경로](docs/F2_FUSION_DESIGN_KR.md)


2026-11-03 사용자 요청으로 만든 독립 프로젝트다. `cpp/`가 새 실행 코드이며,
기존 프로젝트 전체는 `reference/`, 원시 LiDAR rosbag은 `datasets/`로 복사했다.
2026-11-04 중복 대용량 데이터는 SHA256 확인 후 원본 링크로 정리했다. 폴더 이동 시 원본 데이터도 필요하다.
실시간 인지·항법 속도 처리·P5는 C++, Python은 모델 변환·자료 점검·시험에 사용한다.
차량 CAN 송신은 없고 유효성/만료시각을 포함한 shadow 브레이크 추정만 발행한다.

**최신 사용자 결정: GNSS 항법해 대신 내장 IMU만 사용.** 기본 `ego_source: imu`로 변경했다.
아래 GSOF 항법 속도 경로는 비교용 `ego_source: ins` 옵션으로 유지한다.
IMU 자체에는 속도값이 없으며, 현재 원시 IMU 토픽도 확인되지 않았다. `/gnss/imu`는
새 코드가 요구하는 입력 계약이지 실제 센서가 이미 발행 중인 토픽이 아니다.

## 두 모드

| 모드 | 카메라 | 거리 / 상대속도 | 자차속도 |
|---|---|---|---|
| `camera` | 원본 21-frame A₂, 40×7 | A₂ ch6×40 / ch3×20, 위치는 bin 방위 투영 | IMU 적분 기본 / INS 선택 |
| `fusion` | ch0 존재 logit와 연속 bin 영역만 사용 | LiDAR 지면 제거·군집·연속 좌표 추적 | IMU 적분 기본 / INS 선택 |

카메라 모드의 metric 출력은 과거 실험에서 큰 오차가 있었다. 기본
`camera_metric_validated: false`이며 거리 진단값은 내지만 유효한 P5는 내지 않는다.
검증된 보정과 명시적 실험 설정에서만 이를 켠다. 융합 모드는 카메라 metric 채널이
NaN이어도 계산하며, 거리·속도에는 해당 채널을 절대 사용하지 않는다.

## 실행

ROS 2 Humble, Eigen3, PCL, cv_bridge, LibTorch, CMake가 필요하다.
카메라 장치 드라이버는 시스템 `camera_aravis2`, GNSS/LiDAR 소스는 `drivers/`에 있다.
`build`는 복사된 GSOF 메시지를 먼저 빌드하므로 기존 localization workspace의
메시지 설치 경로를 참조하지 않는다. 런타임 코어는 C++17, 현재 호스트 LibTorch
요구에 따라 카메라 추론 실행파일만 C++20으로 빌드한다.

```bash
cd A1_Challenge-main/A1_SensorFusion_CPP
bash run.sh build
bash run.sh drivers    # 복사한 LiDAR/GNSS 드라이버를 이 폴더에 빌드
bash run.sh verify
bash run.sh ros-test   # 격리된 localhost ROS 도메인 87/88/89, 합성 입력
bash run.sh sensor-check --seconds 15  # 실제 입력 확인, 시작한 프로세스만 종료

# 센서 드라이버를 실행하고 시간/보정을 확인한 후, 각각 터미널에서:
bash run.sh camera
bash run.sh fusion -p csv_path:=/tmp/a1_fusion.csv
# 카메라 단독 비교:
bash run.sh fusion -p mode:=camera -p csv_path:=/tmp/a1_camera.csv
```

`cpp/config/shadow.yaml`의 clock/extrinsics/INS 장착 검증 플래그는 기본 false다.
값을 확인하지 않고 true로 바꾸지 않는다. 설정 미완료, stale, 소실, 모호한 대응,
트랙 초기화 중에는 `valid=false`, `request_mps2=NaN`, `reason`을 발행한다.
계산이 가능한 경우도 `reason=shadow_only`이며 실차 제동 성능 인증을 의미하지 않는다.
계획속도 기본은 실제 INS 속도다. `plan_speed_mps`를 0 이상으로 지정하면 별도 what-if이며,
앞차 절대속도 계산에는 항상 실제 자차속도를 사용한다.

입력 `/lead_fusion/a2`는 `A2Features`(header, bin-major float[280], history_ready),
출력 `/lead_fusion/estimate`는 `BrakeEstimate`다. 거리·상대속도·자차/앞차 속도,
계획속도, gate, raw P5, 제한 요청, 유효성, 출처, 만료시각을 함께 기록한다.

## IMU 전용 자차속도 추정

`sensor_msgs/Imu`의 specific force(m/s², 정지 시 중력 포함)와 각속도(rad/s)를
`imu_topic`에서 받는다. 초기 정지 확인(`imu_stationary_start_confirmed`)과 장착 보정
(`imu_mount_verified`, `base_from_imu`)이 모두 필요하다. 정지 2초 동안 중력 방향·가속도
잔차·자이로 바이어스를 추정하고, 자세 적분→중력 제거→속도/위치 적분을 수행한다.
이 경로는 절대속도 측정이 아닌 `imu_integrated_experimental`이며 휠/GNSS 속도 없이
장시간 정확도를 보장할 수 없다. 기본 10초 후 invalid, 50ms 초과 입력 단절이나 잘못된
시각 후에는 재시작 및 새 정지 확인이 필요하다. 이를 자동으로 속도 0으로 초기화하지 않는다.
정지 확인만으로 차량이 등속 주행 중인지 IMU가 판별할 수는 없으므로, 실제 정지 상태 확인이 필수다.

현재 GNSS 장치는 TCP 연결되지만 데이터가 나오지 않는다. 항법해를 사용하지 않더라도
내장 IMU raw 출력/드라이버가 있어야 한다. 이 입력 확보는 미완료이며 가상의 IMU 값을
실측처럼 넣지 않았다. LiDAR IMU로 바꿔 쓰거나 GNSS 속도에 조용히 fallback하지 않는다.

## 구현과 한계

- 카메라: Bayer/BGR → RGB 1280×250 ROI → ImageNet 정규화 → 원본 backbone →
  raw token → 현재+0.05~1.00초 과거 20개 → A₂. 초기/시간 누락 history는 ready=false.
  학습 시 0이던 모델 ego_motion 입력은 유지하고 INS 속도는 후단 계산에 쓴다.
- INS: GSOF 49의 NED 속도를 ENU로 바꾸고 장착 yaw·레버암을 적용한다.
  IMU 가속도를 속도로 취급하지 않는다. 현재 baseline은 roll/pitch 10° 미만의 평면 운동과
  `FULL_NAV=4`, GNSS 상태 1~6을 요구한다. RMS 공분산 기반 품질 모델은 후속 항목이다.
- LiDAR: 필드명/자료형/stride 검사, 점별 timestamp와 일정 INS twist로 scan-end deskew,
  전방 ROI, 제약된 지면 RANSAC, 16cm voxel, 65cm Euclidean clustering,
  세계 좌표 상호 최근접 추적, 3회 관측 확인. 군집은 **차량으로 학습 분류된 검출이 아니다**.
- 대응: 실제 카메라 intrinsics와 base→optical 6DoF로 군집 중심을 bin에 투영한다.
  직진 corridor 내 단 하나의 확정 대응만 선택한다. 다중 후보는 invalid다.
  곡선 경로 선택, full-6DoF motion, 공분산 Kalman/학습 detector, 가림 재식별은 미구현이다.
- 거리: 융합은 관측 표면의 최소 전방 x − 자차 bumper x. 과거 표의 radial range와 다르다.
  카메라 모드는 radial distance를 종방향으로 투영한 proxy이며 실제 범퍼 간격과 동일하지 않다.
- 속도: 군집 중심을 연속 ENU 좌표에서 추적한다. 차체 회전을 속도로 오인하지 않으며
  `v_rel=(v_lead_world-v_ego_world)·forward`, `v_lead=max(0,v_ego+v_rel)`이다.
  IMU 전용 모드에서는 절대 ENU가 아닌 초기 자세 기준의 연속 로컬 좌표다.
- P5: 기존 40개/0.1초 profile, 1초 평균, gate θ=.3 κ=5 β=.7, F1=0,
  F2 제동 요청 −3..0 m/s². 입력 부재는 제동 0으로 대체하지 않는다.

현재 보정은 미완료다. 임시 예제 변환이나 과거 bag의 −2초대 시간 오프셋을 live 값으로 쓰지 않는다.
포트/센서 실측과 시험 결과는 [인계 기록](PROJECT_CONTEXT.md), [센서 점검](docs/SENSOR_AUDIT_KR.md)에 기록한다.

이전 실물 CPU 시험에서 카메라 영상→C++ A₂까지는 동작했지만 A₂는 약 **1.55Hz**였고,
학습의 과거 20개 시각을 충족하지 못해 `history_ready=false`였다. 이후 CUDA 11.8 LibTorch 2.4.1과 대용량 DDS 설정으로 21Hz 처리량 검증을 통과했다. 원시 IMU 입력·외부 보정·시간 검증도 아직 필요하다.

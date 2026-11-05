> 최신 기본 실행은 LiDAR 주 대상 연속성 및 association P5 모드다. [변경 문서](docs/TRACK_CONTINUITY_KR.md)를 먼저 확인한다. 아래 별도 트래커 설명은 변경 전 상태다.

> 2026-11-04 갱신: 후보 최대2대, 수평 중심거리10~50m, 복수 bin 일대일 대응. [최신 자체 점검](docs/SELF_REVIEW_20261104_KR.md). 아래 하드웨어 확인일은 11월3일이다.

# 센서 설정 및 실행 환경 — 2026-11-03 확인

이 문서는 현재 원커맨드 UI 기준이다. 이전 날짜의 CPU 시험·네트워크 구성은 이 설정과 구분한다. 차량 CAN 명령은 송신하지 않는다.

## 실행 / 녹화

작업 루트 `/home/ailab/git/sangbum`에서:

```bash
./run_sensor_fusion.sh         # 카메라, LiDAR, GPU A₂, shadow P5, 대응 UI
./run_sensor_fusion.sh status
./run_sensor_fusion.sh stop    # 이 런처가 시작한 프로세스 종료 / MP4 마감
```

UI는 `A1_Challenge-main/A1_SensorFusion_CPP/outputs/ui_날짜_시간/`에 기록된다.
`outputs/ui_latest_path.txt`가 최근 경로다. `recordings/ui_*.mp4`는 1600×860,
10fps, 60초 단위 MP4V 영상이며 `frames.csv`에 시간이 기록된다.
현재 쓰는 MP4는 구간 종료 또는 정상 종료 후 재생할 수 있다. 자동 삭제는 없다.
[격리·녹화 세부 사항](docs/ISOLATION_RECORDING_KR.md).

## 센서 / 케이블 / IP

| 장치 | 센서 주소 | 호스트 NIC / 주소 | 현재 사용 |
|---|---|---|---|
| Hikrobot MV-CS023-10GC, DA8931780 | 169.254.222.118 | enp5s0 / 169.254.222.1/16 | GigE 카메라 |
| Hesai LiDAR (정확한 모델 미확정) | 192.168.1.201 | enp0s31f6 / 192.168.1.102/24 | UDP 2368, PTC 9347 |
| Trimble 계열 GNSS/INS (정확한 모델 미확정) | 192.168.53.100 | enp4s0 / 192.168.53.20/24 | TCP 5017 연결 이력 있으나 원시 IMU 미수신 |

2026-11-04부터 센서별 NetworkManager 영구 프로필을 사용한다.
enp4s0의 이전 192.168.1.100/24 중복 주소는 새 활성 프로필에서 제거됐다.
각 센서에 /32 경로(metric40)를 저장했으며 인터넷 기본 경로와 분리했다.
[프로필·복구 절차](docs/NETWORK_KR.md)를 따른다.
아래 명령으로 경로를 확인한다. 문서 작성 중 네트워크를 변경하지 않았다.

```bash
ip -br address
ip route get 169.254.222.118
ip route get 192.168.1.201
ip route get 192.168.53.100
```

아래는 이전 임시 설정의 기록이다. 현재는 사용하지 않고 NETWORK_KR.md의 영구 프로필을 활성화한다:

```bash
nmcli device modify enp5s0 +ipv4.addresses 169.254.222.1/16
nmcli device modify enp4s0 +ipv4.addresses 192.168.53.20/24
nmcli device modify enp0s31f6 +ipv4.routes '192.168.1.201/32 0.0.0.0 40'
```

## 카메라 / A₂

- `cpp/launch/hikrobot_gpu.launch.py`: 1280×720 ROI, offset (320,240), BayerRG8, 21Hz, packet 1500, packet delay 2000 ticks.
- `run_ui.sh`의 실제 요청은 자동 노출 Continuous 1000–20000µs, 자동 gain 최대 12dB, white balance Off. 개별 camera-driver-gpu 실행의 수동 노출과 구분한다.
- ROS parameter dump의 ExposureTime은 23841µs였다. 자동 노출 장치 readback인지 오래된 ROS 값인지 미확인으로, 요청 상한 준수를 실측했다고 해석하지 않는다.
- A₂ 입력: RGB y=235:485, 1280×250, 현재+과거20장, 40bin×7채널. 대응에는 존재 채널 ch0/bin만 사용한다.
- 기존 `a2_mf20_v2s_n20.pt` 및 `seg_pretrain_fx640_v3_backbone.pt` 사용. 재학습 없이 `cpp/models_gpu`로 CUDA TorchScript 변환.
- FP32, TF32 off. GPU 실카메라 60초 시험 1260장, 20.997Hz, history_ready 100%. 이는 인식 정확도 검증과 별개다.

## LiDAR / 카메라 대응

설정 원본: [association.yaml](cpp/config/association.yaml), [lidar_host_time.yaml](cpp/config/lidar_host_time.yaml).

- LiDAR −Y가 전방, +Z가 위. body X=−LiDAR Y, body Y=LiDAR X (yaw +π/2).
- 전방 180도 ROI 중심 보정 **0°**. 우측45° 변경은 사용자 요청 취소로 복원됨.
- 카메라는 LiDAR보다 전방 0.80m, 아래 0.30m. 좌우 0과 미세각 0은 가정.
- fx/fy=640, cx=640, cy=360, 센서 높이 1.8m는 초기값. 내부/외부 보정 완료 아님.
- 호스트 수신 시각 사용, 시간 offset 0, 페어 허용 ±80ms. clock/calibration verified=false.
- Patchwork++ 지면 제거, voxel 0.16m, 군집 간격 0.45m, 최소5점, 조각 병합 간격 0.45m.
- 후보 크기 상한 5.2×2.6×2.3m. 고정 축 정렬 크기 조건은 회전·부분 관측 차량에서 탈락 원인이 될 수 있음.
- CV Kalman/Hungarian 추적, 크기·높이 EMA. BEV 전방70m, 좌우 관심영역 ±12m.
- 카메라 존재 진입 0.5 / 유지 0.35 / bin 경계 0.25, 수평 IoU 0.15.
- 독립된 카메라 존재/bin 영역 각각에 LiDAR 투영 군집이 일대일 대응할 때 후보(최대2대). 카메라 거리·속도 채널은 이 대응에 쓰지 않는다.
- 사용자가 강조한 **10m 이상 원거리 적응형 군집 개선은 아직 미구현**. 주행 관측은 `outputs/driving_diagnosis*` 참조.

## IMU / P5 제한

[shadow.yaml](cpp/config/shadow.yaml)은 ego_source=imu, 입력 `/gnss/imu`다.
GNSS 내장 원시 IMU는 아직 수신하지 못했다. IMU 가속도 자체가 속도가 아니며,
정지 바이어스·중력 보정 후 제한 시간 적분이 필요하다. 장착 검증·초기 정지 확인도 미완료다.
LiDAR의 `/lidar_imu`는 GNSS IMU 대체 입력으로 연결하지 않았다.

현재 UI 붉은 대응 트랙과 기존 P5 fusion_node의 트래커는 별개이며 동일 대상 연결 미완료다.
P5 표시 단위 m/s², 제한 요청 −3~0. 결측·보정 미완료·만료는 invalid/NaN/`--` 처리한다.
현재 유효한 실차 제동값이 검증됐다고 해석하면 안 된다.

## ROS 격리 / 주요 토픽

원커맨드는 ROS_DOMAIN_ID=107, ROS_LOCALHOST_ONLY=1, rmw_fastrtps_cpp 및
`cpp/config/fastdds_isolated_ui.xml`을 사용한다. 상속된 팀원 workspace 경로는 초기화한다.
기존 절대 토픽명을 유지하되 도메인으로 격리한다. 동일 물리 센서 동시 점유는 피해야 한다.

| 역할 | 토픽 |
|---|---|
| 카메라 | /camera_driver_gv/vis/image_raw, /camera_driver_gv/vis/camera_info |
| LiDAR | /lidar_points |
| A₂ | /lead_fusion/a2 |
| 대응 / 전방 점군 | /lead_fusion/association, /lead_fusion/front_points |
| 자차 / shadow P5 | /lead_fusion/ego_motion, /lead_fusion/estimate |
| 미수신 GNSS IMU 입력 계약 | /gnss/imu |

우리 노드는 /tf, /tf_static, 차량 control 명령을 발행하지 않는다.
GNSS 입력을 나중에 연결할 때 도메인107로 명시적으로 연결해야 한다.

## 빌드 / 실행 의존성

| 항목 | 확인된 환경 |
|---|---|
| OS / kernel | Ubuntu 22.04.5 / 6.8.1-1059-realtime |
| CPU / GPU | i7-12700, 20 logical CPUs / RTX A5000 약24GB |
| NVIDIA driver | 470.256.02 |
| CUDA toolkit | `/home/ailab/git/sangbum/cuda-11.8`, 11.8.89 |
| LibTorch | `vendor/libtorch`, 2.4.1+cu118 cxx11 ABI |
| ROS | Humble, `/opt/ros/humble` |
| compiler / cmake | g++11.4.0 / CMake3.22.1 |
| OpenCV / PCL / Eigen | 4.5.4 / 1.12.1 / 3.4.0 |
| C++ | core C++17 / camera C++20 |

GPU 빌드: 프로젝트 폴더에서 `bash run.sh build-gpu`. 드라이버 빌드: `bash run.sh drivers`.
모델 변환은 `../A1_FSDS_F2_Validation/.venv`의 torch2.4.1+cu118 / timm1.0.20 환경을 사용했다.
CPU용 system Python torch와 혼동하지 않는다. 현재 install, install_gpu, vendor/libtorch,
cpp/models_gpu, drivers, CUDA 경로는 실행·재빌드 의존성이므로 유지한다.

실측 ROS 파라미터와 설정 사본: [20261103 snapshot](docs/snapshots/20261103/).
구성값 요청과 실제 장치 적용 여부는 구분한다.

# 센서 설정 및 실행 환경

차량 CAN 명령은 보내지 않는다. 이 문서는 센서 배선, 네트워크, 카메라·LiDAR 설정, 빌드 환경을 모은다.
실행 방법은 [README](README.md)에, 동작 설명은 [추적·대응 설계 문서](../docs/superpowers/specs/2026-11-06-sensor-fusion-tracking-association-design.md)에 있다.

> **확인일이 섞여 있다.** 센서 하드웨어와 네트워크는 2026-11-03~04에 확인했고, 설정 값은 2026-11-06 코드 기준이다.
> 센서 구성이 바뀌었을 수 있으므로 현장에서 [센서 확인 명령](#현장-확인-명령)으로 다시 확인한다.
> 설정과 실제 장치 적용 여부는 구분한다.

## 목차

1. [센서와 네트워크](#센서와-네트워크)
2. [카메라와 A₂](#카메라와-a₂)
3. [LiDAR와 군집](#lidar와-군집)
4. [카메라·LiDAR 대응](#카메라lidar-대응)
5. [자차 입력과 P5](#자차-입력과-p5)
6. [ROS 구성과 토픽](#ros-구성과-토픽)
7. [빌드 환경](#빌드-환경)
8. [현장 확인 명령](#현장-확인-명령)

## 센서와 네트워크

| 장치 | 센서 주소 | 호스트 포트 / 주소 | 용도 |
|---|---|---|---|
| Hikrobot MV-CS023-10GC (DA8931780) | 169.254.222.118 | `a1cam0` (enp5s0 위 MACVLAN) / 169.254.222.1/16 | GigE 카메라 |
| Hesai LiDAR (모델 미확정) | 192.168.1.201 | enp0s31f6 / 192.168.1.102/24 | UDP 2368, PTC 9347 |
| Trimble 계열 GNSS/INS (모델 미확정) | 192.168.53.100 | enp4s0 / 192.168.53.20/24 | TCP 5017 (GSOF) |

링크 속도는 카메라 1 Gb/s, LiDAR와 GNSS 100 Mb/s다.

### 네트워크 구성

- 센서별 NetworkManager 영구 프로필(`a1-sensor-camera`, `a1-sensor-lidar`, `a1-sensor-gnss`)을 쓴다. autoconnect, 기본 경로 없음, IPv6 끔, 센서 IP마다 `/32` 경로(metric 40)를 저장해서 인터넷 기본 경로와 분리했다.
- **카메라는 2026-11-04 이후 `a1cam0`로 연결한다.** enp5s0의 기존 공유 프로필(`Wired connection 1`, `ipv4.method=shared`)이 보조 주소를 유지하지 못해, 그 프로필은 두고 MACVLAN 프로필 `a1-camera-isolated`를 추가했다. [복구 기록](docs/network/camera_restore_20261104/README.md)
- LiDAR는 `255.255.255.255:2368`로 **브로드캐스트**한다. 호스트 주소가 바뀌어도 수신은 막히지 않지만, 유니캐스트로 바꾸면 목적지 주소를 확인해야 한다.
- 센서 IP나 케이블 포트를 바꾸면 위 매핑을 함께 수정한다. 프로필 이름은 [네트워크 문서](docs/NETWORK_KR.md)를 따른다.

### 주의할 점

| 항목 | 내용 |
|---|---|
| Hesai 드라이버 | 호스트에서 한 번만 실행한다. ROS 도메인이 달라도 UDP 2368은 공유되므로 두 번 실행하면 점유 실패나 수신 분배 문제가 생긴다 |
| `setup_lidar_host.sh` | `A1_localization`의 이 스크립트는 호스트 주소를 192.168.1.100/24로 바꾼다. 현재 GNSS 포트(enp4s0)에 실행하면 GNSS 주소를 깨뜨린다 |
| 프로필 우선순위 | 센서 프로필(priority 100)이 원래 프로필보다 앞선다. 예전 용도의 케이블을 같은 포트에 꽂으면 매핑을 다시 확인한다 |
| GNSS 수신 | 2026-11-04 점검 시점에는 TCP 5017 연결만 확인됐고 GSOF 항법·원시 IMU는 수신되지 않았다. 이후 수신 상태는 이 문서에서 확인하지 못했다 |
| 카메라 교체 | 과거 카메라(DA8931779, 192.168.1.51)와 serial이 다르다. 저장된 장착 보정이 맞는다고 가정하지 않는다 |

재부팅 후 프로필이 유지되는지는 실제로 시험하지 않았다. 점검 상세: [배선·수신](docs/SENSOR_AUDIT_KR.md), [localization·GNSS 호환성](docs/NETWORK_COMPATIBILITY_20261104_KR.md), [공용 LiDAR](docs/SHARED_LIDAR_KR.md).

배선 사본은 [current_cabling.yaml](cpp/config/current_cabling.yaml)이다 (2026-11-03 관찰값, 기하 보정이 아니라 네트워크 주소만).

## 카메라와 A₂

### 카메라 설정

설정 원본: [hikrobot_gpu.launch.py](cpp/launch/hikrobot_gpu.launch.py). 모든 값이 launch 인자다.

| 항목 | 값 |
|---|---|
| 전송 영상 | 1280×720, 센서 ROI offset (320, 240), BayerRG8 |
| 프레임 속도 | 21 Hz |
| 패킷 | 크기 1500, 지연 2000 ticks |
| 노출 | launch 기본 수동 15,000 µs. **UI 실행은 3,000 µs**로 덮어쓴다 (`A1_CAMERA_EXPOSURE_US`) |
| 게인 | 수동 0 dB. 자동 게인과 자동 화이트밸런스는 Off |

실행 경로에 따라 노출이 다르다. UI가 카메라를 켜면 3,000 µs, `run.sh camera-driver-gpu`를 직접 실행하면 launch 기본값(15,000 µs)이다.
2026-11-04 실행에서 드라이버 로그로 두 Auto가 Off이고 노출 3,000 µs가 읽히는 것을 확인했다. 게인 0 dB는 실행 인자로만 확인했고 장치에서 숫자로 읽지는 않았다.

노출을 올리면 노면이 포화된다. 2026-11-05 영상 분석에서 모델 입력 band의 평균 36.9%가 포화됐다. 현장 최종 노출은 [보정 절차](docs/CALIBRATION_KR.md)에서 정한다.

### A₂ 모델

| 항목 | 값 |
|---|---|
| 입력 | RGB, y=235:485 (1280×250), 현재 + 과거 20장 |
| 출력 | 40 bin × 7 채널. 대응에는 존재 채널 ch0만 쓰고 거리·속도 채널은 읽지 않는다 |
| 가중치 | `a2_mf20_v2s_n20.pt`, `seg_pretrain_fx640_v3_backbone.pt`. 재학습 없이 변환 |
| 변환본 | `cpp/models_gpu/` (CUDA TorchScript). 변환 환경 torch 2.4.1+cu118, timm 1.0.20 |
| 정밀도 | FP32, TF32 끔 |

변환 검증은 `cpp/models_gpu/export.json`에 있다. 가중치 키가 모두 일치했고 head 오차는 1.4e-6 이하다.
GPU 실카메라 60초 시험에서 1,260장, 20.997 Hz, history_ready 100%였다. 이는 처리 속도이며 인식 정확도의 검증이 아니다.

## LiDAR와 군집

설정 원본: [association.yaml](cpp/config/association.yaml), [lidar_host_time.yaml](cpp/config/lidar_host_time.yaml).

### 좌표계와 시각

| 항목 | 값 |
|---|---|
| 방향 | LiDAR −Y가 전방, +Z가 위. 차체 X = −LiDAR Y, 차체 Y = LiDAR X (yaw +π/2) |
| 위치 | 차체 기준 LiDAR `[1.16, 0, 1.20]` m. `vehicle_extrinsics.yaml`의 값 |
| 전방 ROI 중심 | 0° (우측 45° 변경은 사용자 요청으로 취소됨) |
| 시각 | 드라이버 `use_timestamp_type: 1` (호스트 수신 시각), 시간 offset 0, 페어 허용 ±80 ms |
| 검증 | `clock_verified=false` |

호스트 수신 시각을 쓰면 한 scan 안의 점 시각이 뭉칠 수 있다. 시각 폭이 5 ms 미만이면 scan 시각 하나로 처리한다.

### 처리 순서

| 단계 | 설정 |
|---|---|
| 처리 거리 | 120 m까지 (주 대상은 50 m까지) |
| 전방 ROI | 폭 `max(12 m, 0.58·x)`, 높이 경사 여유 10% |
| 지면 제거 | Patchwork++, 센서 높이 1.2 m |
| 맵 필터 | 켜짐 (`map_filter_enabled`). 맵이 정상일 때 바닥·벽 후보 제거 |
| voxel | 0.16 m |
| 군집 간격 | 0.45 m. 20 m부터 거리 1 m당 0.01 m씩 넓어져 최대 0.85 m |
| 최소 점 수 | 5점. 40 m 이상은 3점 (링 하나로 보이는 차량 허용) |
| 조각 병합 | 간격 0.45 m, 높이 간격 1.2 m |
| 지면 위 최소 높이 | 0.15 m |
| 차량 크기 상한 | 길이 5.2 m, 폭 2.6 m (최소면적 사각형 기준) |

### 원거리에서 점이 적은 이유

32채널이 31°를 고르게 나눈다고 가정한 계산이며, 실제 값은 `ring` 필드로 확인해야 한다.

| 거리 | 링 간격 | 차에 맞는 링 | 링당 점 |
|---|---|---|---|
| 30 m | 0.52 m | 약 2개 | 약 20개 |
| 65 m | 1.13 m | 약 1개 | 약 9개 |
| 100 m | 1.75 m | 0~1개 | 약 6개 |
| 120 m | 2.09 m | 0~1개 | 약 5개 |

관련 모델(XT32M2X) 매뉴얼은 10% 반사율 기준 80 m를 말한다. 공식 데이터시트를 찾지 못해 이 수치들은 계산이나 간접 자료다.
원거리 판정은 이 한계를 고려해 [점수 합산 방식](../docs/superpowers/specs/2026-11-06-sensor-fusion-tracking-association-design.md)으로 설계했다.

## 카메라·LiDAR 대응

설정 원본: [calibration.yaml](cpp/config/calibration.yaml). 카메라와 LiDAR 기하 값은 이 파일 한 곳에만 둔다.

| 항목 | 값 | 상태 |
|---|---|---|
| 카메라 위치 | LiDAR보다 전방 0.80 m, 아래 0.30 m, 좌우 0 | 좌우와 미세각은 가정 |
| 내부 파라미터 | fx=fy=640, cx=640, cy=360 | **초기값. 측정하지 않았다** |
| 왜곡 | `k1, k2, p1, p2` 모두 0 | 측정 전 |
| 검증 | `calibration_verified=false` | |

fx=640은 실제와 크게 다른 것으로 보인다. 2026-11-05 영상에서 일부 구간의 추정값이 약 1030~1170이었으나 구간마다 흩어져 yaw나 시간 오차도 섞여 있다. 추정값을 넣지 않고 [보정 절차](docs/CALIBRATION_KR.md)로 측정한다.
1920 px 센서를 1280 px로 잘라 쓰므로 수평 화각이 학습 카메라와 같다는 근거도 아직 없다.

### 대응 방식

- 트랙이 영상에 투영된 bin의 A₂ 확률을 읽어 트랙별 근거를 누적한다. 다른 트랙이 차지한 bin은 읽지 않는다.
- 한 bin에 여러 트랙이 겹치면 가까운 트랙이 차지하고, 뒤 트랙은 가림 상태로 근거를 유지한다.
- A₂ 존재 확률의 표시 기준은 진입 0.4, 유지 0.35, 경계 0.25다. 판단에는 쓰지 않고 화면과 CSV용이다.
- 주 대상은 근거 L이 3.0 이상일 때 진입하고 1.0 미만에서 해제한다. A₂가 2초 멈추면 원거리 대상 외에는 해제한다.
- 카메라 거리·속도 채널은 대응에 쓰지 않는다.

## 자차 입력과 P5

설정 원본: [shadow.yaml](cpp/config/shadow.yaml).

| 항목 | 값 |
|---|---|
| 기본 입력 | `ego_source: gnss_odom` — `/a1_gnss/odom` + `/gsof_client/gsof/ins_solution_49` |
| 위치 | 속도를 적분해서 만든다. localization의 map 보정 위치나 정지 유지는 쓰지 않는다 |
| 거부 조건 | 정렬 불량, 발행자가 하나가 아님, 상태가 오래됨, 프레임 불일치, 기울기 10° 초과 |
| 검증 플래그 | `gnss_odom_verified`, `primary_motion_verified`, `ins_mount_verified`, `imu_mount_verified` 모두 false |
| `bumper_x_m` | 0.0 (실측 전. 간격이 실제보다 크게 계산될 수 있다) |

**검증되지 않은 값의 처리:** 속도와 P5는 계산해서 `speed_unverified_mps`, `request_unverified_mps2`로 CSV와 UI에 기록한다. `valid=false`와 `request=NaN`은 그대로다.
UI는 이 값을 회색 `UNVERIFIED`로 보여 준다. 유효한 제동값이 검증됐다고 해석하면 안 된다.

**IMU 적분 모드:** `ego_source: imu`는 `/gnss/imu`를 쓰는 실험 모드다. 가속도 자체는 속도가 아니므로 정지 바이어스·중력 보정 후 제한 시간(10초) 적분이 필요하다. 장착 검증과 초기 정지 확인도 끝나지 않았다. LiDAR의 `/lidar_imu`는 GNSS IMU 대체 입력으로 연결하지 않았다.

P5 단위는 m/s², 제한 요청은 −3~0이다. 결측·보정 미완료·만료는 invalid, NaN, `--`로 처리한다.

## ROS 구성과 토픽

### 기본 구성

- 공용 domain0의 `/lidar_points`를 직접 구독한다 (`A1_FUSION_TRANSPORT=direct`). 도메인은 A1_Total의 `config/ros.env`가 정한다.
- `bridge` 모드는 domain107, `ROS_LOCALHOST_ONLY=1`로 격리하고 LiDAR만 전달한다. 맵과 TF를 받지 못해 맵 필터가 꺼진다.
- DDS는 `rmw_fastrtps_cpp`이고 설정은 [fastdds_large_images.xml](cpp/config/fastdds_large_images.xml)(direct), [fastdds_isolated_ui.xml](cpp/config/fastdds_isolated_ui.xml)(bridge)다.
- 동일한 물리 센서를 동시에 점유하지 않는다. DDS 격리로는 센서 점유를 막지 못한다.

### 토픽

| 역할 | 토픽 |
|---|---|
| 카메라 | `/camera_driver_gv/vis/image_raw`, `/camera_driver_gv/vis/camera_info` |
| LiDAR | `/lidar_points` |
| 자차 | `/a1_gnss/odom`, `/gsof_client/gsof/ins_solution_49` |
| 맵 | `/localization/map_surfaces`, `/localization/map_surfaces_epoch`, `/tf`, `/tf_static` |
| A₂ | `/lead_fusion/a2` |
| 대응 | `/lead_fusion/association`, `/lead_fusion/association/image` |
| 점군 | `/lead_fusion/front_points`, `ground`, `nonground`, `clusters`, `map_removed` |
| 자차 / P5 | `/lead_fusion/ego_motion`, `/lead_fusion/estimate` |

우리 노드는 `/tf`, `/tf_static`, 차량 제어 명령을 **발행하지 않는다.**

## 빌드 환경

2026-11-03에 확인한 차량 PC 환경이다.

| 항목 | 환경 |
|---|---|
| OS / 커널 | Ubuntu 22.04.5 / 6.8.1-1059-realtime |
| CPU / GPU | i7-12700 (논리 20) / RTX A5000 약 24 GB |
| NVIDIA 드라이버 | 470.256.02 |
| CUDA | 11.8.89 (`third_party/cuda-11.8`) |
| LibTorch | 2.4.1+cu118, cxx11 ABI (`third_party/libtorch`) |
| ROS | Humble |
| 컴파일러 / CMake | g++ 11.4.0 / 3.22.1 |
| OpenCV / PCL / Eigen | 4.5.4 / 1.12.1 / 3.4.0 |
| C++ 표준 | 17 (카메라 노드는 20) |

경로는 A1_Total 기준이다. 이 저장소에는 `install/`, LibTorch, CUDA가 없다.

- 빌드: `bash build.sh --packages-up-to a1_fusion_cpp` (A1_Total 루트)
- 드라이버 빌드: `bash run.sh drivers`
- Patchwork++는 KAIST 원본을 `third_party/`에서 그대로 컴파일한다. 출처와 라이선스는 [SOURCES.md](third_party/SOURCES.md)에 있다.
- 모델 변환은 별도 venv의 torch 2.4.1+cu118 / timm 1.0.20을 쓴다. CPU용 시스템 Python torch와 섞지 않는다.

실측 ROS 파라미터와 설정 사본: [20261103 스냅샷](docs/snapshots/20261103/).

## 현장 확인 명령

```bash
ip -br address
ip route get 169.254.222.118   # 카메라 → a1cam0
ip route get 192.168.1.201     # LiDAR  → enp0s31f6
ip route get 192.168.53.100    # GNSS   → enp4s0
nmcli -f NAME,DEVICE connection show --active
```

자동 연결이 선택되지 않았을 때:

```bash
nmcli connection up a1-sensor-camera   # 카메라는 a1-camera-isolated일 수 있다
nmcli connection up a1-sensor-lidar
nmcli connection up a1-sensor-gnss
```

수신 점검은 `run_sensor_fusion.sh lidar-check`, `topic-audit`를 쓴다. 이전 임시 `nmcli device modify` 설정은 현재 쓰지 않는다. 필요하면 [네트워크 문서](docs/NETWORK_KR.md)의 복구 절차를 따른다.

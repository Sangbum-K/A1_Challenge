# A1_Total 센서 퓨전

카메라 모델 A₂가 앞차의 방향을 고르고, LiDAR가 그 차의 거리와 상대속도를 재서 제동 요구값(P5)을 **shadow로 계산**하는 C++ ROS 패키지(`a1_fusion_cpp`)다.
차량 CAN 명령은 보내지 않는다. 센서·장착·시각·운동 검증이 끝나지 않아 요청값은 `valid=false`, `NaN`이다.

> **상태:** 합성 시험만 통과했다. 실제 빌드와 실차 성능은 검증되지 않았다.
> 진행 경과는 [개발 기록](../docs/devlog/README.md), 현재 동작의 기준은 [추적·대응 설계 문서](../docs/superpowers/specs/2026-11-06-sensor-fusion-tracking-association-design.md)다.

## 목차

1. [한눈에 보기](#한눈에-보기)
2. [실행](#실행)
3. [환경 변수](#환경-변수)
4. [녹화·재생·평가](#녹화재생평가)
5. [주요 동작](#주요-동작)
6. [아직 꺼져 있는 기능](#아직-꺼져-있는-기능)
7. [폴더 구성](#폴더-구성)
8. [문서 안내](#문서-안내)
9. [변경 이력](#변경-이력)

## 한눈에 보기

```text
카메라 21 Hz ─ A₂ (GPU) ───────────────┐  트랙별 bin 확률
                                        ▼
LiDAR 10 Hz ─ 지면 제거 ─ 군집 ─ 추적 ─ 카메라 근거 누적 ─ 대상 선택 ─ shadow P5
                 ▲                ▲
         맵 표면 필터        자차 운동 (GNSS odom)
```

| 항목 | 내용 |
|---|---|
| 런타임 | `cpp/` C++ ROS 패키지. Python은 진단·시험·모델 변환용 |
| LiDAR 입력 | 공용 domain0 `/lidar_points` 직접 구독 (`config/ros.env`에서 도메인 지정) |
| 자차 입력 | `/a1_gnss/odom` + GSOF #49 정렬 상태 (`ego_source: gnss_odom`) |
| 센서 드라이버 | 통합 workspace에서 한 번만 실행한다. UI가 LiDAR·GNSS를 기동하지 않는다 |
| 카메라 | UI가 소유·기동하는 것이 기본값 (`A1_CAMERA_MODE`) |
| 출력 | `/lead_fusion/*` 토픽, `association.csv`, `tracks.csv`, UI 영상 |

센서 토픽과 도메인은 팀의 `A1_localization`과 같다. [계약 표](SENSOR_ENVIRONMENT_KR.md#localizationa1_localization과-맞춘-계약)
자차 입력의 검증 플래그(`gnss_odom_verified` 등)는 모두 false다. `/gnss/imu` 적분은 직접 선택하는 실험 모드이며 자동 fallback하지 않는다.

## 실행

차량 PC 기준이다. 이 저장소에서는 `sensor_fusion/` 아래에 있으며, 스크립트는 두 배치를 모두 인식한다.

```bash
cd /home/ailab/git/A1_Total
source env.sh
bash build.sh --packages-up-to a1_fusion_cpp
```

| 목적 | 명령 |
|---|---|
| UI 실행 (카메라, GPU A₂, 대응, shadow P5) | `bash src/perception/sensor_fusion/run_sensor_fusion.sh` |
| 상태 확인 | `… run_sensor_fusion.sh status` |
| 종료 (MP4 마감) | `… run_sensor_fusion.sh stop` |
| LiDAR 수신 점검 | `… run_sensor_fusion.sh lidar-check` |
| LiDAR 발행 Hz 측정 (5초, 4 Hz 미만이면 실패) | `… run_sensor_fusion.sh lidar-rate` |
| 토픽 감사 | `… run_sensor_fusion.sh topic-audit --seconds 20` |

시험은 `run.sh`에서 한다.

| 목적 | 명령 |
|---|---|
| ROS 없는 라이브러리 시험 | `bash run.sh core` |
| 기준 시험 전체 | `bash run.sh verify` |
| ROS 시험 / 대응 시험 | `bash run.sh ros-test` / `bash run.sh target-test` |

## 환경 변수

| 변수 | 값 | 설명 |
|---|---|---|
| `A1_CAMERA_MODE` | `start`(기본) / `shared` / `off` | `shared`는 같은 도메인의 기존 `image_raw`를 쓰고 드라이버를 시작하지 않는다. `off`는 LiDAR-only 진단용 |
| `A1_CAMERA_ENABLED` | `0` | 예전 방식의 `off` |
| `A1_CAMERA_EXPOSURE_US` | 기본 `3000` | UI가 카메라를 켤 때의 수동 노출 |
| `A1_FUSION_TRANSPORT` | `direct`(기본) / `bridge` | `bridge`는 이전 domain107 격리 경로이며 LiDAR만 전달한다. 맵과 TF를 받지 못해 맵 필터가 꺼진다 |
| `A1_RECORD_BAG` | `1` | 주행 원본 녹화 |
| `A1_BAG_MIN_FREE_GB` | 기본 `64` | 녹화 전 필요한 빈 공간 |
| `A1_MAP` | localization `map:=`과 같은 값 (접두 경로 또는 `.mm`) | UI 시작 때 그 맵의 표면 캐시를 확인하고 맵 서버를 함께 시작한다. 없으면 맵 필터는 우회된다 |
| `A1_MAP_SURFACES` | `.surfaces.pcd` 경로 | 재생 때 맵 서버가 읽을 맵 |

## 녹화·재생·평가

```bash
# 1. 주행 원본 녹화 (분당 약 3.2 GB)
A1_RECORD_BAG=1 bash src/perception/sensor_fusion/run_sensor_fusion.sh

# 2. 같은 입력으로 다시 실행 (센서 없음, 맵 서버는 같은 PCD로 함께 시작)
A1_MAP_SURFACES=maps/track_merged/map.surfaces.pcd \
  bash src/perception/sensor_fusion/run.sh replay <bag_dir>

# 3. 결과 평가 (labels.csv: start,end,lead)
python3 src/perception/sensor_fusion/cpp/tools/track_eval.py <out>/association.csv --labels labels.csv
```

- bag에는 A₂ 결과(`/lead_fusion/a2`)도 들어가서, 재생할 때 GPU 없이 같은 입력을 쓴다. `--rerun-a2`를 주면 A₂를 다시 계산한다.
- 맵 토픽은 bag에 없다. 재생 때 같은 PCD로 맵 서버를 띄운다.
- 맵 품질 검사가 실시간 시계를 쓰므로 재생 배속은 1.0으로 고정이다.
- UI 영상은 `outputs/ui_날짜_시간/recordings/`에 1600×860, 10 fps, 60초 단위 MP4로 남고 `frames.csv`에 시각이 기록된다.

## 주요 동작

### 추적과 카메라 대응

- 트랙은 연속 odom 좌표계에서 CTRV 필터로 차량 **후면 기준점**을 추적한다.
- 카메라는 트랙이 투영된 bin의 확률을 트랙별 근거로 누적한다. 카메라가 잠시 못 봐도 대상을 유지한다.
- 더 가까운 차가 조건을 갖춰 0.3초 유지되면 주 대상을 교체한다.
- 카메라 근거 없는 근거리 배경은 주 대상이 되지 못한다.

### LiDAR 속도 감시

트랙 확정과 카메라 근거는 시간이 아니라 **스캔 수**로 쌓이고 10 Hz에 맞춰 정했다. 최근 11개 스캔 간격의 중앙값이 7 Hz 미만이면 경고를 내고,
4 Hz 미만이면 주 대상을 비우고 상태를 `lidar_rate_low`로 낸다(근거 상한이 확정 기준 3.0에 못 미치기 시작하는 지점은 약 3.2 Hz).
`association.csv`의 `scan_dt_s`에 스캔 간격이 남고, `track_eval.py` 요약의 `scan_rate`에 처리된 스캔의 중앙값 Hz가 나온다. UI 시작 때도 같은 측정을 한다.

### 거리 범위

| 항목 | 값 |
|---|---|
| 추적·기록 | 120 m까지 |
| 주 대상 | 3~50 m. 상한은 bag 평가를 통과한 뒤에만 올린다 |
| 원거리 판정 | `map_status=ok`일 때만 수행 |

### 차량 형상 필터

실측 군집 점의 XY 최소면적 사각형으로 최대 길이 5.2 m, 폭 2.6 m를 판정한다. 회전해도 축 정렬 폭이 커지지 않는다.
`tracks.csv`에 `footprint_valid`, `length_m`, `width_m`, `yaw_rad`가 남는다. 이 yaw는 관측 표면의 축이며 차량 진행각이 아니다.

### 맵 기반 필터

localization이 선택한 같은 맵의 바닥·수직 평면 후보를 해시 검증 후 연결한다. 최근 ICP 품질이나 LiDAR 시각의 TF가 없으면 기존 필터로 우회한다.
팀의 `A1_localization`은 맵 표면을 발행하지 않으므로 `A1_MAP`을 주어야 이 필터가 켜진다.
세 맵 비교, 토픽, 재생성·검증 방법은 [맵 필터 안내](docs/map_filter/README.md)에 있다.

### 노출 조절 (UI)

연결 창의 Exposure (µs) 칸을 클릭하거나 `E`를 눌러 값을 입력하고 Enter 또는 Apply를 누른다. `±1ms` 버튼은 1000 µs씩 바꾼다.

- Actual은 카메라 readback이다. `unavailable`/`stale`이면 최근 확인이 없다는 뜻이다.
- 지원 범위는 15~1,000,000 µs다. 노출이 길면 카메라 FPS가 떨어질 수 있다.
- 수동 노출 모드에서만 동작하며 `/camera_driver_gv`의 `camera_control_node` 파라미터를 바꾼다.
- 입력한 값은 현재 카메라 세션에만 적용된다. `shared` 모드는 기존 설정을 유지한다.

## 아직 꺼져 있는 기능

| 기능 | 이유 | 켜는 조건 |
|---|---|---|
| 가드레일 거부, 원거리 경로 | 맵 캐시에 구조물 라벨이 없어 `map_status=no_structure` | 키프레임 접근 확인 후 라벨 연결, 캐시 재생성 |
| 레이싱라인 판단 | `raceline_path`가 비어 있고 `raceline_verified=false` | CSV 경로 입력, 맵과 좌표 정렬 확인 |
| 카메라 대응의 정확도 | fx=640은 초기값이라 틀린 bin을 읽을 수 있다 | [카메라·LiDAR 보정](docs/CALIBRATION_KR.md) |
| 유효한 제동 요청 | 검증 플래그가 모두 false | 장착·시각·운동 실측 |

## 폴더 구성

```text
sensor_fusion/
├── cpp/            ROS 패키지 본체
│   ├── src/ include/   노드와 라이브러리 (tracker, camera_evidence, raceline …)
│   ├── config/         association, shadow, calibration, DDS, LiDAR 설정
│   ├── models_gpu/     A₂ CUDA TorchScript
│   ├── tests/ tools/   시험, 평가·감사 도구
│   └── launch/ msg/
├── docs/           설계·검증 기록
├── tools/          맵 표면 캐시 생성
├── third_party/    Patchwork++
├── reference/      이전 프로젝트 보존본 (디스크에만 있고 git 제외)
├── outputs/        실행 결과 (git 제외)
└── run*.sh         런처
```

`docs/snapshots/`는 증거 자료다. 모델과 녹화 원본의 바이트를 보존한다.
이전 프로젝트 보존본 `reference/`는 디스크에만 두고 git은 추적하지 않는다. 코드가 읽는 카메라 설정, P5 기준 구현, 재생 입력은 `cpp/config/`, `cpp/tests/p5_reference.py`, `cpp/evidence/`로 복사해 `reference/` 없이 동작한다.

## 문서 안내

| 문서 | 내용 |
|---|---|
| [SENSOR_ENVIRONMENT_KR.md](SENSOR_ENVIRONMENT_KR.md) | 센서, 네트워크, 카메라·LiDAR 설정, 빌드 환경 |
| [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) | 누적 결정과 인계 사항 |
| [CALIBRATION_KR.md](docs/CALIBRATION_KR.md) | 카메라·LiDAR 보정 절차 |
| [맵 필터](docs/map_filter/README.md) | 맵 비교, 실행·재생성 |
| [UI](docs/UI_KR.md), [네트워크](docs/NETWORK_KR.md), [공용 LiDAR](docs/SHARED_LIDAR_KR.md), [GPU](docs/GPU_RUNTIME_KR.md) | 항목별 상세 |
| [추적 연속성](docs/TRACK_CONTINUITY_KR.md), [대응](docs/ASSOCIATION_KR.md) | 11/06 교체 이전의 기록 |
| [이전 README](docs/history/20261104_pre_integration_README.md), [이전 인계](docs/history/20261104_pre_integration_PROJECT_CONTEXT.md) | 원문 보존. 상대 링크는 원래 sensor_fusion 루트 기준 |

토픽 계약과 2 Hz 진단의 기준은 A1_Total의 `docs/SENSOR_PIPELINE_KR.md`다. 이 저장소에는 없다.

## 변경 이력

| 날짜 | 변경 | 검증 수준 |
|---|---|---|
| 11/06 | 추적·대응·대상 선택 교체. CTRV, 후면 기준점, 카메라 근거 누적, bag 녹화·재생·평가 | 합성 시험 |
| 11/05 | 후면→측면 전환 연결 개선. 자차 입력을 GNSS odom으로 전환 | 합성 시험 |
| 11/04 | 차량 형상 필터 (영상 00001–00004 검토 후). 맵 기반 필터 | 합성 시험. 해당 주행의 원시 점군이 없어 재처리 검증 불가 |

실차 코너 주행의 추적률과 상대속도 정확도는 어느 항목도 검증되지 않았다. 투영 오차는 별도 실측 보정이 필요하다.

# reference 추적 해제와 git 추적 범위 정리

- 시각: 2026-11-06 오후
- 커밋: 없음. 변경은 스테이지되어 있다
- 근거: Claude Code 세션 기록, 의존성 검색, 이동 전후 파일 비교
- 앞 기록: [폴더 정리](2026-11-06-repo-restructure.md)

## 요약

작업 환경을 차량 통합 워크스페이스(A1_Total)로 정했다. 이전 프로젝트 보존본 `sensor_fusion/reference/`는 **git에서 추적하지 않기로** 했다.
코드가 `reference/`를 읽는 곳이 있어서, 필요한 파일은 `cpp/` 안으로 복사해 `reference/` 없이 동작하게 했다.

> **정정:** 처음에는 "reference는 필요 없다"는 말을 폴더를 치워도 된다는 뜻으로 오해해 휴지통으로 옮겼다. 요청은 추적에서 빼라는 것이었으므로 같은 날 디스크에 원상 복구했다(1.5 GB, 바이트 단위로 이동). 아래 "휴지통으로 옮겼다가 되돌린 것" 절은 그 과정의 기록이다.

## 먼저 한 결정: git 추적 범위

추적 파일을 약 1,500개(수백 MB)에서 286개로 줄였다. 용량은 소스와 모델 `models_gpu`(162 MB) 중심이다.

| 추적 | 제외 |
|---|---|
| `sensor_fusion/cpp/`, `docs/`, 런처, 문서 | `outputs/` (단 `COLCON_IGNORE`는 유지) |
| `third_party/` 소스 | `docs/presentations/` (77 MB), `cpp/models/*.pt` (CPU용 사본), Patchwork++ 그림 |
| `docs/devlog/`, 설계 문서 | `build/`, `install/`, `log/`, `*.db3`, `*.mcap`, `*.bag` |

`.git`은 689 MB다. 과거 커밋에 들어간 파일은 줄지 않는다.

## reference 의존성과 처리

| 사용처 | 처리 |
|---|---|
| 카메라 launch (`hikrobot_gpu.launch.py`)가 설정 2개를 읽음 | `camera_info_1280x720_uncalibrated.yaml`, `camera_dynamic_parameters.yaml`을 `cpp/config/`로 복사하고 경로 변경 |
| P5 기준 시험 (`test_p5_reference.py`)이 Python P5 구현을 import | `p5.py`를 `cpp/tests/p5_reference.py`로 복사 (numpy만 사용) |
| `run.sh verify`의 재생 입력 CSV | `cpp/evidence/lidar_p5_20261008_10hz.csv`로 복사 (0.9 MB) |
| 모델 변환 `export_a2.py`와 `run.sh export` | 삭제. 원본 `senior_code` 체크포인트가 있어야 동작한다 |
| `verify_copies.py`, `COPY_MANIFEST.json` | 삭제. reference 복사본 검증용이었다 |
| `run.sh verify`의 보존본 시험과 번들 검증 | 삭제 |

복사한 4개는 원본과 바이트가 같다(`cmp`).

## 휴지통으로 옮겼다가 되돌린 것 (복구 완료)

처음에 `~/.Trash/A1_Challenge_cleanup_20261106/reference/` (1.5 GB)로 옮겼다가 `sensor_fusion/reference/`로 되돌렸다.

| 폴더 | 크기 | 비고 |
|---|---|---|
| `scc_controller_imitator-main` | 1.0 GB | 체크포인트 942 MB는 git에 없던 파일이라 **이 Mac에만 있었을 수 있다** |
| `A1_RealCamera_A2_21FPS` | 209 MB | |
| `A1_LeadVehicle_LidarFusion` | 185 MB | |
| `A1_RealCar_A2_P5Brake` | 160 MB | |

git 이력에는 `reference/` 파일 대부분이 남아 있다. scc의 `data/`는 이력에 없고 이 Mac의 `reference/`에만 있다.

## 검증

| 항목 | 결과 |
|---|---|
| 코드에서 `reference/` 참조 | 없음 (py, sh, yaml, cpp, hpp, xml 검색) |
| `run.sh` 문법 | 통과 |
| launch 설정 경로 | 두 파일 모두 `cpp/config/`에서 확인 |
| `p5_reference.p5_accel` | import와 호출 정상 |
| `test_track_eval.py` | 2건 통과 |

**실행하지 못한 것:** `run.sh verify` 전체와 `test_p5_reference.py`는 cmake와 Eigen이 없어 돌리지 못했다. 변경 후 첫 실행은 Nuvo에서 한다.
카메라 launch는 경로만 확인했고 실제 ROS 실행은 하지 않았다.

## 주의할 점

- 차량 PC가 이 저장소를 `git pull`하면 새 커밋이 삭제하는 `reference/` 파일이 그쪽 작업 폴더에서도 지워진다.
- `export_a2.py`와 `verify_copies.py`는 이미 지웠다. `reference/`가 있으므로 필요하면 git 이력에서 되살릴 수 있다.
- 이 저장소를 받는 쪽에는 `reference/` 파일이 없다. 새 커밋이 추적 파일을 지우기 때문이다.

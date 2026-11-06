# 저장소 폴더 구조 정리

- 시각: 2026-11-06 13:48 ~ 13:57
- 커밋: 없음. 이동과 삭제는 스테이지만 됐고, 내용 수정은 스테이지되지 않았다
- 근거: Claude Code 세션 기록, 이동 전후 전체 파일 해시 비교

## 요약

사용자 요청은 세 가지였다. sensor_fusion을 저장소 루트 바로 밑으로 옮기고, 21FPS와 scc_controller를 reference 안으로 넣고, 불필요한 파일을 정리한다.

## 구조 변화

| 이전 | 이후 |
|---|---|
| `src/perception/sensor_fusion/` | `sensor_fusion/` |
| `A1_RealCamera_A2_21FPS/` | `sensor_fusion/reference/A1_RealCamera_A2_21FPS/` |
| `scc_controller_imitator-main/` | `sensor_fusion/reference/scc_controller_imitator-main/` |
| `무제 폴더/` | `docs/presentations/` |
| `A1 Architecture Analysis.html` | `docs/A1 Architecture Analysis.html` |

```text
A1_Challenge/
├── .gitignore
├── docs/
│   ├── A1 Architecture Analysis.html
│   ├── devlog/
│   ├── presentations/
│   └── superpowers/specs/
└── sensor_fusion/
    ├── cpp/  docs/  outputs/  third_party/  tools/  run*.sh
    └── reference/
        ├── A1_LeadVehicle_LidarFusion/
        ├── A1_RealCamera_A2_21FPS/
        ├── A1_RealCar_A2_P5Brake/
        └── scc_controller_imitator-main/
```

## 결정과 이유

### reference 위치

새 폴더를 만들지 않고 기존 `sensor_fusion/reference/`에 합쳤다.

- P5Brake의 `activate_nuvo_env.sh`와 `preflight.sh`가 `../A1_RealCamera_A2_21FPS/.venv`를 형제 경로로 찾는다.
- `COLCON_IGNORE`가 있어서 scc 안의 ROS 패키지가 빌드에 잡히지 않는다.

### 차량 PC 경로 유지

차량 PC는 계속 `A1_Total/src/perception/sensor_fusion` 구조다. 경로를 단순히 바꾸면 차량 PC가 깨지고, 그대로 두면 이 저장소에서 `run.sh core`가 저장소 밖에 빌드 폴더를 만든다.
그래서 경로가 `src/perception/sensor_fusion`으로 끝나면 기존처럼, 아니면 상위 폴더를 워크스페이스로 쓰도록 판별을 넣었다.

적용 위치는 `run.sh`, `run_ui.sh`, `cpp/tools/association_audit.py`, `gpu_camera_benchmark.py`, `live_sensor_audit.py`다.

README의 `src/perception/sensor_fusion` 명령은 차량 PC 기준이라 그대로 두고, 맨 위에 안내를 한 줄 추가했다.

## 정리한 파일 (132개)

| 대상 | 개수 | 처리 | 이유 |
|---|---|---|---|
| `.codex-build/` | 115 | 휴지통 | 9월 발표 자료 빌드 잔재 |
| `outputs/perception.mp4` | 1 | 휴지통 | `docs/presentations` 사본과 해시 동일 |
| `.DS_Store` | 7 | 삭제 | 시스템 파일 |
| `__pycache__` 안의 파일 | 6 | 삭제 | 재생성됨. 추적되던 `.pyc` 1개 포함 |
| `outputs/ui.lock`, `ui.pid`, `ui_latest_path.txt` | 3 | `git rm` | 실행 중 생기는 파일 |

휴지통 위치는 `~/.Trash/A1_Challenge_cleanup_20261106/`이다. `.codex-build`의 `user-v6-p5-edit/candidate-v2.pptx`는 다른 곳에 사본이 없을 수 있다.

## 함께 수정한 내용

- 설계 문서 링크 4곳: `README.md`, `PROJECT_CONTEXT.md`, `docs/ASSOCIATION_KR.md`, `docs/TRACK_CONTINUITY_KR.md`
- 설계 문서의 경로 2곳: 대상 코드 위치, `bag_qos_override.yaml` 출처

## 일부러 남긴 것

| 대상 | 이유 |
|---|---|
| `reference/` 내부 | `sensor_fusion/AGENTS.md`가 원본 보존을 요구한다 |
| 같은 체크포인트의 `senior_code` 3벌 (각 160 MB) | 위와 같음. 지우려면 사용자 결정이 필요 |
| LeadVehicle의 `.build/` (236 KB) | 위와 같음 |
| 깨진 심볼릭 링크 5개 | 이동 전부터 깨져 있었다 |
| `../../../docs/SENSOR_PIPELINE_KR.md` 링크 2개 | 대상 파일이 A1_Total에만 있다 |
| `COPY_MANIFEST.json` | 777개 항목 중 631개가 이동 전부터 없던 오래된 목록이지만 기록으로 둔다 |

## 검증

| 항목 | 결과 |
|---|---|
| 파일 해시 대조 | 이동 전 1,499개 중 1,357개 동일, 수정 10개, 정리 132개. 예상 밖 누락과 추가 없음 |
| 심볼릭 링크 | 5개 보존 |
| 번들 검증 | P5Brake, 21FPS 모두 통과 (이동 전과 동일) |
| `test_track_eval.py` | 2건 통과 (이동 전과 동일) |
| 워크스페이스 판별 | 두 배치 모두 쉘 스크립트 실행과 Python 식 평가로 확인 |
| 설계 문서 링크 | 4곳 모두 실제 파일로 연결됨 |

C++ 빌드, ROS 실행, `test_p5_reference.py`, `run.sh verify` 전체는 이 Mac에 cmake, colcon, ros2가 없어 실행하지 못했다.

## 주의할 점

- 차량 PC에 넣을 때는 `A1_Total/src/perception/sensor_fusion` 경로를 유지해야 기존 명령이 맞다.
- `sensor_fusion/`을 통째로 복사하면 1.9 GB다. git에 없는 scc 체크포인트 942 MB가 포함된다.
- `docs/presentations`(77 MB)는 git에 없는 파일이라 `git add -A`를 하면 함께 올라간다.

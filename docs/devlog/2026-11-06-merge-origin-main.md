# 원격 병합

- 시각: 2026-11-05 23:58 ~ 2026-11-06 00:18
- 커밋: `8d5f7bc` 로컬 변경 커밋, `9bd0355` 병합, `1a42852` 루트 AGENTS.md 삭제
- 근거: Claude Code 세션 기록, git 이력

## 요약

GitHub의 `A1_Challenge`를 이 Mac으로 가져왔다. 로컬에 커밋하지 않은 카메라 실험 변경이 있어서 먼저 커밋한 뒤 병합했다.
사용자 지시는 "하나씩 보고 대충 해결해, fusion 폴더만 그대로 가져오고"였다. sensor_fusion은 원격 그대로 받고, 나머지 충돌은 로컬 쪽을 택했다.
병합된 코드는 실행하지 않았고 충돌 마커가 없는지만 확인했다. push하지 않아서 로컬이 `origin/main`보다 3커밋 앞서 있다.

## 진행 순서

1. 로컬의 `A1_RealCamera_A2_21FPS` 변경을 커밋했다 (`8d5f7bc`).
2. 원격에서 새 커밋 3개를 받았다: `3fd2fa3`, `8192773`, `a353f97`.
3. 원격이 추가한 파일과 겹치는 untracked 파일 15개를 임시 폴더로 옮겼다.
4. 충돌을 해결하고 병합 커밋을 만들었다 (`9bd0355`).
5. 사용자 확인을 받아 루트 `AGENTS.md`를 지웠다 (`1a42852`).

## 충돌 해결

| 파일 | 처리 |
|---|---|
| sensor_fusion 전체 | 원격 그대로. 충돌 없음 |
| `A1_RealCamera_A2_21FPS/README.md` | 로컬 |
| `A1_RealCamera_A2_21FPS/PROJECT_CONTEXT.md` | 로컬 |
| `A1_RealCamera_A2_21FPS/scripts/live_camera_a2_21fps.py` | 로컬 |
| `launch/hikrobot_a2_camera_1280x720.launch.py` | pull로 원격 버전이 됐던 것을 로컬 백업으로 복원 |
| `scripts/run_live_visualization.sh` | 위와 같음 |
| 루트 `AGENTS.md` | 로컬은 삭제, 원격은 수정. 한때 원격 버전을 남겼다가 사용자 확인 후 삭제 |

## 로컬 커밋 내용 (`8d5f7bc`)

2026-09 실내외 실험 내용이 커밋되지 않은 채 남아 있던 것이다. 카메라 입력을 UVC 시제품에서 실제 ROS2 경로로 바꾼 내용이 중심이다.

| 파일 | 변경 |
|---|---|
| `configs/real_camera_mf20.yaml` | 입력을 `camera_aravis2` BayerRG8 ROS2로 변경. 1280×720, offset (320, 240), 노출 15,000 µs |
| `requirements.txt` | 버전 고정: torch 2.4.1(cu118), torchvision 0.19.1, timm 1.0.20, numpy 1.26.4, opencv 4.10.0.84 |
| `scripts/live_camera_a2_21fps.py` | +375/−58. ROS 이미지 구독(`LatestROSFrame`), overlay, band 포화 측정, ego 입력 처리 추가 |
| `README.md`, `PROJECT_CONTEXT.md`, 실험 문서 | 9/14 실내, 9/18 실외 결과 반영 |

이 커밋에는 루트 `AGENTS.md` 삭제도 들어 있다.

## 정정 (2026-11-06)

당시 기록에는 `RESULTS_2026-09-14.md`의 로컬 원본을 복구하지 못했고, 로컬에서 고친 내용이 있었다면 사라졌다고 적었다.
**사실이 아니다.** 백업 파일이 `RESULTS_2026-09-14.md/` 폴더 안에 한 단계 더 들어가 있었을 뿐이다.
백업과 현재 파일의 SHA-256이 같으므로 잃은 내용은 없다. 나머지 백업 14개도 현재 파일과 같다.

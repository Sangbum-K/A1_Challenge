# A1 실차 A₂ → P5 룰베이스 브레이크

실물 Hikrobot 카메라 한 대로 21-frame A₂를 21 FPS로 돌리고, A₂의 앞차 거리·종방향 상대속도로
선배 P5 규칙의 브레이크 요구량을 **m/s²**로 계산·표시·기록하는 통합 프로젝트다.
차량으로 보내지는 않는다. F1과의 결합은 나중에 정한다.

```text
camera_aravis2 → ROS2 BayerRG8 → backbone/i₁ → 21-frame A₂ (40,7)
  → 식 0 lead(d, v_rel) → 식 1 rule gate → 식 2 P5 → 식 3 결합 → ACC_Cmd [−3, +1] m/s²
```

- 설계와 식: [docs/DESIGN_KR.md](docs/DESIGN_KR.md) (원본 설계안 [docs/F2_RuleBased_Design.html](docs/F2_RuleBased_Design.html))
- 내일 현장 절차: [docs/FIELD_TEST_2026-10-08_KR.md](docs/FIELD_TEST_2026-10-08_KR.md)
- 세션 인수인계: [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)

## 실행

```bash
cd /home/ailab/git/sangbum/A1_Challenge-main/A1_RealCar_A2_P5Brake
./run.sh check          # 출발 전 점검
./run.sh live d20       # 실행 (이름만 바꿔가며). q/Esc로 종료
./run.sh stop           # 비정상 종료 뒤 정리
```

`live` 기본값: 노출 3000 µs 고정, 자차 50 km/h 기준 브레이크, 원본 bag 녹화, **화면 녹화
`a2_overlay.mp4`**, 화면 3프레임마다 갱신. 카메라는 `enp0s31f6`(프로필 `camlink-onboard`). 결과는 `outputs/<날짜_시각>_<이름>/`이고, 종료하면 `brake_replay/`에 브레이크·트래킹
그래프(`brake_replay.png`, `tracks_bev.png`)와 CSV가 자동으로 생긴다.

화면 오른쪽에는 앞차 거리, 자차 50 km/h 기준 **종방향** 상대속도, 브레이크(m/s²), 앞차 bin의 A₂ 값,
위에서 본 **차량 트래킹 지도**가 나온다. 왼쪽 카메라 화면에는 트래킹된 차량이 삼각형으로 표시된다.

그 밖의 모드: `verify`(해시+단위시험), `aim`(조준), `bench`(처리량만), `replay <bag>`, `brake-log <frames.jsonl>`.
옵션은 `./run.sh help`. 자주 쓸 만한 것:

| 옵션 | 의미 |
|---|---|
| `--plan-kph K` | 브레이크 계산용 자차속도 (기본 50). `actual`이면 실제 자차속도(정지면 P5 항상 0) |
| `--exposure V` | 노출 µs (기본 3000 고정) |
| `--display-every N` | 화면 갱신 간격 (live 기본 3) |
| `--no-bag` / `--no-video` | 원본 bag 녹화 끄기 / 화면 녹화 끄기 (둘 다 기본 켜짐) |
| `--vrel-mode track\|min\|a2\|ddot` | 상대속도: 트래킹 앞차 속도(기본) / `min(A₂, ḋ)` / A₂만 / 거리 기울기만 |
| `--ego-source can` | 실제 자차속도를 CAN 0x201에서 (**미검증**) |

## 환경

Nuvo의 ROS2 Humble + `A1_RealCamera_A2_21FPS/.venv`(torch 2.4.1 cu118)를 그대로 쓴다. 이 폴더에
`.venv`를 만들면 그쪽이 우선이다. `src/a1_brake`와 `brake-log`는 numpy·PyYAML만 필요하고,
단위시험과 PNG는 시스템 `python3`(pytest, matplotlib)로 돈다.

## 지켜야 할 것

- `senior_code/`는 byte-identical 복사본이다. 수정하지 않는다. `./run.sh verify`로 확인한다.
- `src/a1_brake/p5.py`의 `replan_profile`을 바꾸면 `tests/test_p5_matches_senior.py`가 깨져야 정상이다.
- 처리량(21 FPS), A₂ 인식 정확도, 브레이크 규칙의 동작, 브레이크 값의 실차 타당성은 따로 판정한다.

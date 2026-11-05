# A1 실차 A₂ → P5 룰베이스 브레이크 통합 설계

작성 2026-10-07 · 상태: 구현·단위시험·합성영상 end-to-end 시험 완료, **실물 카메라·실차 미검증**

## 1. 목표와 범위

실물 Hikrobot 카메라 한 대로 A₂(21-frame)를 21 FPS로 돌리면서, A₂의 **앞차 거리**와
**종방향 상대속도**만 사용해 선배 P5 규칙이 요구하는 브레이크 감속을 **m/s²**로 낸다.

```text
Hikrobot → camera_aravis2 → ROS2 BayerRG8 → debayer → ROI y=235:485 → ImageNet norm
 → seg-pretrained EfficientNetV2-S → i₁ (40,256) → current + 20 past (0.05–1.00 s)
 → A2ModelMultiFrame → A₂ (40,7)                         ← A1_RealCamera_A2_21FPS (검증된 경로)
 → 식 0 lead (k*, d, v_rel) → 식 1 rule gate g → 식 2 P5 a_P5 → 식 3 결합 → ACC_Cmd m/s²
                                                          ← 이 프로젝트에서 추가
```

범위 안: 브레이크 요구량 **계산·표시·기록**.
범위 밖: 차량 송신(CAN 0x157 등), F1 결합 확정, 레이트 리밋, 학습 모델(F2/gate) 복원.
F1과 어떻게 합칠지는 나중에 정한다. 지금은 `a_F1 = 0`(현재 속도 유지)으로 둔다.

## 2. 프로젝트 구성

```text
A1_RealCar_A2_P5Brake/
├── run.sh                      단일 진입점 (check/verify/aim/live/bench/replay/brake-log/stop)
├── PROJECT_CONTEXT.md          세션 인수인계 (먼저 읽기)
├── configs/brake_p5.yaml       P5·gate·결합 파라미터 (값마다 [선배]/[실차]/[미정] 표시)
├── configs/…                   카메라 설정 (A1_RealCamera_A2_21FPS에서 그대로 복사)
├── launch/                     camera_aravis2 launch (그대로 복사)
├── src/a1_brake/               브레이크 로직, numpy만 사용
│   ├── a2_lead.py              식 0  A₂ → k*, d, v_rel
│   ├── closing_rate.py         ḋ    거리 기울기 (1 s 최소제곱)
│   ├── gate.py                 식 1  선배 규칙 gate
│   ├── p5.py                   식 2  선배 replan_profile + 1 s 평균 읽기
│   ├── brake.py                식 0–3 묶음 BrakeController, what-if
│   ├── tracking.py             차량 위치 트래킹 (bird's-eye Kalman, 표시·기록 전용)
│   └── ego_speed.py            실제 자차속도: 상수(기본 0) / CAN 0x201(미검증)
├── scripts/live_a2_brake.py    실시간 러너 (A₂ 러너 + 브레이크 단계 + 브레이크 패널)
├── scripts/replay_brake_log.py 기록된 frames.jsonl → 브레이크 CSV/PNG/요약
├── tests/                      선배 P5 수치 일치 + 식 0–3 동작 시험
├── senior_code/                byte-identical 선배 파일 4개 + COPY_MANIFEST.sha256
└── docs/                       이 문서, 현장 절차, F2_RuleBased_Design.html 원본
```

### 출처

| 부분 | 출처 | 변경 |
|---|---|---|
| 카메라 launch/config, aim, preflight, camera_params | `A1_RealCamera_A2_21FPS` | preflight의 .venv/러너 이름만 |
| A₂ 런타임 (`live_a2_brake.py`의 앞부분) | `A1_RealCamera_A2_21FPS/scripts/live_camera_a2_21fps.py` | 모델·전처리·history 경로는 그대로, 브레이크 단계·패널·요약만 추가 |
| `replan_profile` | `A1_FSDS_F2_Validation/src/fsds_f2_validation/p5_expert.py` | 없음 (선배 원본과 atol 1e-12 일치 시험) |
| 선배 P5 원본 | `scc_controller_imitator-main/scripts/gen_profile_pedal_labels_recalc.py` | `senior_code/scripts/`에 복사, 수정 금지 |
| 식 0–3 | `F2_RuleBased_Design.html` (2026-09-29) | 그대로 구현 |
| 실행 환경 | `A1_RealCamera_A2_21FPS/.venv` | 이 폴더에 `.venv`가 없으면 그것을 사용 |

## 3. 계산식 (frame마다, 21 Hz)

**식 0 · lead 추출** — A₂ 채널 0 logit의 bins 16..25 argmax.

```text
k* = argmax_{k∈16..25} A[k,0],  d = 40·A[k*,6],  v_rel = 20·A[k*,3]   (음수 = 접근)
ḋ  = d(t)의 최근 1 s 최소제곱 기울기 (카메라 timestamp 기준, 8샘플·0.5 s 이상일 때만)
v_rel,used = 트래킹 앞차 vy         (vrel_mode=track, 기본; 없으면 ḋ → A₂)
             min(v_rel, ḋ)          (vrel_mode=min, 설계안 원래 방식; a2 / ddot 선택 가능)
v_lead = max(0, v_ego,actual + v_rel,used)
```

d ≤ 0 또는 NaN이면 lead 없음 → a_P5 = 0.

**식 1 · 규칙 gate** — 선배 `MODEL_AUDIT_2026-05-26.md` L33–37.

```text
s_t = max_{k∈16..25} A_t[k,0]
g_raw = σ((s_t − 0.3)·5),   g_t = 0.7·g_{t−1} + 0.3·g_raw,   g_0 = 0
```

**식 2 · P5** — 선배 라벨 생성기 그대로, 실차용으로 바꾼 곳은 세 가지.

```text
s_min,eff = 7 + 6·ln(1 + v_plan/5)
t_i = 0.1·i (i<40),  v_ref,i = v_plan,  x_i = Σ v_ref·Δt
L_i = d + v_lead·t_i                      ← 실차: 등속 가정 (선배는 시뮬 앞차 궤적)
u_i = min(v_ref,i, max(0, (L_i − x_i − s_min,eff)/2.0))
backward: u_i ≤ u_{i+1} + 20·Δt ;  forward: u_0 = v_plan, clip(±20/5·Δt), ≥0
a_P5 = (u_10 − u_0) / 1.0 s               ← 실차: 1 s 평균 (첫 칸은 −20 고정 bang-bang)
```

**식 3 · 결합**

```text
δ = min(0, a_P5 − a_F1),  a_final = a_F1 + g·δ,  ACC_Cmd = clip(a_final, −3, +1)
```

`ACC_Cmd` 범위는 차량 DBC `EAIT_Control_02 (0x157) ACC_Cmd 24|16@1+ (0.01,−10.23) [−3|1] m/s^2`.
로그에는 클립 전 `a_final`도 남긴다.

## 4. 실제 자차속도와 계획 자차속도 (정지 시험에서 중요)

두 속도를 분리했다. `run.sh` 기본은 **계획 50 km/h 고정, 노출 3000 µs 고정**(2026-10-07 결정)이다.

- **실제 자차속도** `v_ego,actual`: 카메라가 실제로 움직이는 속도. A₂의 상대속도는 이것 기준이므로
  `v_lead` 계산에는 항상 이 값을 쓴다. 기본 상수 0, `--ego-source can`이면 CAN 0x201 휠속 평균.
- **계획 자차속도** `v_plan`: P5가 계획하는 자차속도. 기본은 실제 자차속도와 같다.

**자차가 정지(0)이면 P5는 구조적으로 항상 0을 낸다.** 계획이 u_0 = 0에서 시작하고 0 아래로
내려갈 수 없기 때문이다(`tests/test_brake.py::test_parked_ego_never_requests_deceleration`).
그래서 정지 시험에서는 같은 관측(d, v_lead, g)에 대해 가상 주행속도 10/20/30/40/50 km/h에서
P5가 무엇을 요구했을지를 **what-if** 행으로 함께 계산·표시·기록한다. `--plan-kph`(기본 50)를 주면
화면 큰 숫자(main)가 그 속도 기준으로 계산된다.

참고 수치 (gate = 1, 앞차 정지, `a_P5`, m/s²):

| d | 10 km/h | 30 km/h | 50 km/h |
|---:|---:|---:|---:|
| 10 m | −2.78 | −8.33 | −13.89 |
| 20 m | 0 | −8.33 | −13.89 |
| 30 m | 0 | −3.94 | −13.32 |
| 40 m | 0 | 0 | −8.32 |
| 60 m | 0 | 0 | 0 |

1 s 창에서는 `a_P5 ≥ −v_plan/1 s`이다. ACC_Cmd에서는 −3으로 잘린다.
선배 식은 x_i를 **원래 계획(v_ref)대로 갔을 때**의 위치로 두므로 감속 요구가 큰 편이다.

## 5. 출력

`live` / `bench` / `replay`는 `--out` 폴더에 다음을 남긴다.

| 파일 | 내용 |
|---|---|
| `frames.jsonl` | frame마다 기존 A₂ 기록 + `brake` {lead, gate, main, what_if, ego_speed_valid} + `brake_ms`. `live`는 `a2_bins.raw_40x7` 전체도 저장 |
| `summary.json` | 기존 21 FPS 판정 + `brake` {config, ego 출처, plan, gate on 비율·flip/분, gate on 동안 d·v_rel·ḋ 분포, ACC_Cmd 분포(main, what-if), brake_ms} |
| `a2_overlay.mp4` | 1920×910: 왼쪽 카메라 + A₂ 진단 + 트래킹 삼각형, 오른쪽 640 px에 거리·자차 50 km/h 기준 종방향 상대속도·브레이크(m/s²), 앞차 bin A₂ 값, 트래킹 지도, 10 s 그래프 (`live`는 기본 꺼짐, `--video`로 켬) |
| `run_config.json` | 카메라·노출·브레이크 옵션 |

`./run.sh brake-log <frames.jsonl>`은 모델 없이 기록된 A₂만으로 식 0–3을 다시 계산해
`brake_replay/brake_replay.csv`, `brake_replay.png`, `brake_replay_summary.json`을 만든다.
파라미터를 바꿔 같은 run을 몇 번이든 다시 볼 수 있다.

## 6. 검증 현황 (2026-10-07)

| 항목 | 상태 | 근거 |
|---|---|---|
| 선배 파일 무결성 | PASS | `verified 4 bundled senior files` |
| P5 = 선배 원본 | PASS | `test_p5_matches_senior.py` 14건, atol 1e-12 |
| 식 0–3 동작 | PASS | `test_brake.py` 16건 (중간값 일치, 계획속도 기준 상대속도 포함) |
| 트래킹 | PASS | `test_tracking.py` 5건 (기하, 그룹→측정, 접근 추적·속도, 옆 차로 제외, 삭제) |
| 합성 BayerRG8 → 모델 → 브레이크, 화면/영상 없음 | 21.0 FPS, 유실 0, compute p95 46.5 ms, brake p50 0.68 ms | `outputs/synthetic_bench_*` (RT 커널) |
| 같은 경로 + MP4 + 전체 로그 | 18.5 FPS, FAIL | `outputs/synthetic_smoke_*`; 시각화 p50 10.7 ms가 원인 |
| 9/27 실측 로그 재생 (90 cm, 20 m 정지차) | 동작함, 아래 문제 확인 | `outputs/replay_20260927_h90_d20_static` |
| 실물 카메라로 이 러너 실행 | **미검증** | 카메라 미연결 |
| 거리·상대속도 정확도 | **미검증** (9/27: 20 m 정지차를 약 29 m, +10 m/s로 읽음) | `RESULTS_2026-09-27.md` |
| CAN 자차속도 | **미검증** | 코드만 있음, 기본 꺼짐 |

9/27 재생에서 확인된 점 (20 m 정지차, 카메라 정지):

- gate ≥ 0.5 비율 26 %, **분당 90회 on/off**. 선배 θ=0.3이 실카메라 logit 분포에서 문턱 근처에 걸린다.
- gate on 동안 A₂ v_rel 중앙값 **+9.95 m/s** (실제 0). ḋ는 중앙값 −1.97이지만 p10–p90이
  −8.7~+3.2 m/s로 흔들린다. 거리 자체가 24–46 m로 움직이기 때문이다.
- min 규칙 때문에 97.6 % 프레임에서 ḋ가 선택됐다. 결과 v_lead 중앙값 0(정지차와 맞음)이지만
  이건 우연에 가깝고, ḋ 잡음이 그대로 브레이크 쪽 잡음이 된다.
- what-if 30 km/h에서 ACC_Cmd ≤ −0.5 비율 40 %. 실제 20 m라면 −3에 가까워야 하지만 거리가
  29 m로 읽혀 약하게 나온다.

결론: **브레이크 계산 경로는 동작하지만, 입력인 A₂ 거리·속도가 아직 정확하지 않다.**
브레이크 숫자의 타당성은 A₂ 정확도 검증 뒤에만 말할 수 있다.

## 6-1. 차량 위치 트래킹

브레이크와 별개로, A₂ 40 bin 전체에서 σ(logit) ≥ 0.5인 인접 bin 그룹을 한 대의 차량 검출로 보고
위에서 본 좌표로 추적한다.

```text
θ_k = atan(((k+0.5)·32 − 640) / 640)          학습 기하 fx = cx = 640 (수평 90°)
x = d·sin θ (횡, + 오른쪽),  y = d·cos θ (종, 앞)
상태 [x, y, vx, vy] 등속 Kalman, 측정 잡음 σ_range 6 m · σ_bearing 0.05 rad, 가속 잡음 3 m/s²
같은 차 중복 병합: 횡 2.5 m · 종 25 % 이내 트랙은 hit 많은 쪽만 유지
χ² 9.21 게이트 nearest-neighbour 연결, 3 hit 확정, 1 s 미검출 삭제
리드 트랙 = 방위가 bins 16..25 안인 확정 트랙 중 가장 가까운 것
```

`frames.jsonl`의 `tracking`에 측정·확정 트랙·리드 트랙이 남고, `brake-log`가
`tracks_bev.png`와 CSV 열(`track_lead_*`)을 만든다. 실제 카메라는 보정 전이므로 위치의 m 단위는 참고용이다.

## 7. 남은 결정

- F1 결합: `a_F1`, `v_ref,i`를 F1 계획에서 받기 (현재 0, v_plan 상수).
- 레이트 리밋(급제동·급해제 방지)과 gate 히스테리시스.
- gate θ/κ/β 실카메라 재조정 (9/27 재생에서 flip 90/분).
- ḋ 창 길이 / bin 전환 처리 (d 잡음 → 브레이크 잡음).
- 차량 송신 방식: m/s²는 0x157 `ACC_Cmd`, A1 기준 DBC 0x210 `break_command`는 % 단위.
- 규정: 손으로 짠 gate/P5는 "단일 신경망" 해석에 걸릴 수 있음 (설계안 10장).

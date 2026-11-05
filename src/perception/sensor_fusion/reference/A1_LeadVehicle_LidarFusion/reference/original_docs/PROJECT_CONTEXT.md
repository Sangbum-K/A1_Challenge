# A1 실차 A₂ → P5 브레이크 — 세션 인수인계

마지막 갱신: 2026-10-07 (Asia/Seoul). 새 세션은 작업 전에 이 파일과 `docs/DESIGN_KR.md`를 읽는다.

## 1. 사용자 목표 (2026-10-07)

A1_RealCamera_A2_21FPS에서 카메라로 외부 인식 시험을 했던 21-frame A₂를 실차에 적용해,
A₂ 출력 중 **앞차 거리(ch6×40)**와 **종방향 상대속도(ch3×20)**로 **P5 룰베이스 브레이크**를
**m/s²**로 본다. 카메라 경로, 화면 진단, F2 계산식, F2 validation 코드를 하나로 합친 것이 이 프로젝트다.
제어(F1)와의 결합은 나중에 정한다. 차량 송신은 범위 밖이다.

2026-10-08 트랙 시험: 자차 정지(0 km/h), 앞차 거리를 바꿔가며 A₂와 브레이크 출력을 확인한다.
절차는 `docs/FIELD_TEST_2026-10-08_KR.md`.

## 2. 기존 프로젝트와의 관계

| 폴더 | 역할 | 지금 |
|---|---|---|
| `A1_RealCamera_A2_21FPS` | camera → A₂ 21 FPS 처리량 검증 (범위 a₂까지) | 기록 보존. `.venv`를 이 프로젝트가 빌려 씀 |
| `A1_FSDS_F2_Validation` | FSDS 시뮬 F1/F2/P5 검증 | `p5_expert.replan_profile`과 senior 대조 시험을 가져옴 |
| `A1_F2_Laptop_Transfer` | FSDS validation의 노트북용 축소본 (`src/` 동일) | 사용 안 함 |
| `F2_RuleBased_Design.html` | 식 0–3 설계안 (2026-09-29) | `docs/`에 복사, 그대로 구현 |
| **`A1_RealCar_A2_P5Brake`** | **통합 프로젝트** | 앞으로 실차 작업은 여기서 |

A₂ 경로 코드(`scripts/live_a2_brake.py`)는 `live_camera_a2_21fps.py`를 복사해 브레이크 단계만 더했다.
모델 로드, 전처리, ROI, timestamp history, ROS2 QoS는 바꾸지 않았다.

## 3. 결정사항

- 식 0: lead bin = bins 16..25의 채널 0 logit argmax. 임계값 없음(신뢰도는 gate가 담당).
- 식 1: 선배 규칙 gate θ=0.3, κ=5, β=0.7, g₀=0 (MODEL_AUDIT_2026-05-26.md L33–37에서 확인).
- 식 2: 선배 `replan_profile` 그대로 + 등속 앞차 + v_ref=v_plan 상수 + 1 s 평균 읽기.
- 식 3: `a_final = a_F1 + g·min(0, a_P5 − a_F1)`, `a_F1 = 0`(F1 미정), `ACC_Cmd = clip(−3, +1)`.
- 상대속도 (2026-10-07 변경): 기본 `vrel_mode=track` = 트래킹 앞차 Kalman vy(없으면 ḋ, 그다음 A₂).
  근거: A₂ ch3는 학습·배포 모두 ego_motion=0인데도(`train_a2_multiframe.py:178`) 실카메라 정지 장면에서
  +10.2/+13.4/+14.8 m/s(장면마다 다름)로 읽혀 상수 보정 불가. 같은 정지 장면에서 ḋ p10–p90 ±8, track vy ±2–5.
  10/07 live 마지막 60 s 재계산: v_rel_used ±8 → ±2 m/s, 브레이크 on/off 33 → 15회/분.
  ch3가 실제 움직임에 반응하는지는 이동 앞차 GT run으로만 판단 가능(정지 데이터뿐이라 미확인).
  이전 방식은 `--vrel-mode min`.
- **실제 자차속도와 계획 자차속도를 분리했다.** v_lead는 실제 자차속도로 계산한다. 자차가 정지면
  P5는 구조적으로 0이므로 what-if 10–50 km/h를 항상 같이 계산·표시·기록한다.
- 2026-10-07 사용자 결정: **노출 3000 µs 고정**(9/27 A₂ run 값, `live`의 `[`/`]` 조절 끔),
  **P5 계획 자차속도 50 km/h 고정**(`run.sh` 기본, `--plan-kph actual`로 해제). 실제 자차속도는 0 유지.
  화면(1920×910) 오른쪽 640 px 열에는 **계산식 없이** 거리, 자차 50 km/h 기준 앞차 상대속도
  (`lead_rel_speed_at_plan_mps = v_lead − v_plan`), 브레이크 ACC_Cmd m/s², 선택 bin의 A₂ 7채널
  raw/decoded, 10 s 그래프만 표시한다(사용자 요청). 식 중간값은 frames.jsonl에만 남긴다.
- 2026-10-07 추가 결정: 화면 속도는 **종방향 상대속도만** 표시(횡방향·절대속도 표시 제거, 로그에는 유지).
  실행은 `./run.sh live <이름>` 하나로 단순화(bag 기본 켬, MP4 기본 끔, 화면 3프레임 간격,
  종료 후 `brake_replay/` 그래프 자동 생성; 그래프는 matplotlib가 있는 `/usr/bin/python3`로 실행).
- **차량 위치 트래킹** `src/a1_brake/tracking.py` (2026-10-07 첫 실물 run 뒤 σ_range 6 m·σ_bearing 0.05 rad, 같은 차 중복 트랙 병합 추가): 40 bin 전체의 object 그룹(σ ≥ 0.5)을 학습 기하
  (fx=cx=640, 90°)로 bird's-eye (x 횡, y 종) 점으로 바꾸고, 등속 Kalman + gated NN으로 추적
  (3 hit 확정, 1 s 미검출 삭제). 리드 = bins 16..25 방위 안의 가장 가까운 확정 트랙.
  표시·기록 전용이며 브레이크에는 쓰지 않는다. 9/27 20 m 정지차 로그에서 리드 트랙 존재 84 %,
  30 s 동안 리드 ID 15개, y 중앙값 31.3 m(p10–p90 24.5–38.9), vy 중앙값 −1.5 m/s
  (A₂ ch3는 +10). 파라미터 스윕에서 기본값보다 나은 조합은 없었다 — 흔들림은 A₂ 측정 자체.
- 실제 자차속도 기본은 상수 0. CAN 0x201 `USER_right_wheel_info`(A1 기준 DBC
  `/home/ailab/git/A1-BSW/DBC/A1_dbc_fixed.dbc`, 12bit ×0.1 km/h) 리더는 stdlib SocketCAN으로
  구현했지만 **미검증·기본 꺼짐**. `.venv`에는 python-can이 없다.
- `EAIT_CAN(AVANTE_CN7).dbc`는 2026-10-07 A1-BSW 커밋 496b32a에서 삭제됐다(git 이력에 있음).
  m/s² 명령은 0x157 `ACC_Cmd`(−3..+1), A1 0x210 `break_command`는 % 단위다.

## 4. 2026-10-07 검증

- `./run.sh verify`: senior 파일 4개 해시 OK, pytest 35개 통과(트래킹 5개 포함)(선배 P5 원본 대조 14개, 화면 중간값 일치 1개 포함).
- 합성 BayerRG8 → 실제 GPU 모델 → 브레이크, 화면/영상 없음, RT 커널: a₂ 21.0 FPS, 유실 0,
  compute p95 46.5 ms, brake p50 0.68 ms / p95 1.23 ms. `outputs/synthetic_bench_*`.
- 같은 경로 + MP4 + 전체 로그: 18.5 FPS FAIL (시각화 p50 10.7 ms). `outputs/synthetic_smoke_*`.
- 9/27 90 cm·20 m 정지차 로그 재계산: gate on 26 %, flip 90회/분, A₂ v_rel 중앙값 +9.95 m/s,
  ḋ p10–p90 −8.7~+3.2 m/s. `outputs/replay_20260927_h90_d20_static`.
- 실물 카메라 실행: 2026-10-07 `first_live`(250 s), `second_live`(125 s), 2026-10-08 `video_check` 완료. CAN은 미검증.

## 5. 다음 세션이 지킬 것

1. A₂ 정확도, 브레이크 규칙 동작, 처리량, 실차 브레이크 타당성을 따로 보고한다.
2. 정지 자차 run에서 main ACC_Cmd = 0은 정상이다. what-if 행으로 규칙 반응을 본다.
3. `senior_code/`는 수정 금지. `replan_profile` 수정 시 senior 대조 시험이 깨져야 한다.
4. 파라미터는 `configs/brake_p5.yaml`에서 바꾸고 `./run.sh brake-log`로 같은 입력에 재계산해 비교한다.
5. 처리량 숫자는 `./run.sh bench`로 잰다. 화면·영상 run의 FPS를 기준값으로 쓰지 않는다.
6. 남은 결정: F1 결합(a_F1, v_ref), 레이트 리밋, gate 히스테리시스·재조정, ḋ 잡음/bin 전환 처리,
   차량 송신 경로, 규정 해석(손으로 짠 gate/P5).

## 2026-10-08 카메라 포트 변경·화면 녹화 기본화

- 카메라가 온보드 포트 **enp0s31f6**(1 Gb/s)로 옮겨졌다. enp4s0에는 라이다(192.168.1.201, `a1-lidar`
  192.168.1.100/24, 약 5.6 MB/s), enp5s0에는 `Wired connection 1`(192.168.53.20 공유)이 있다.
  NM 프로필 **`camlink-onboard`**를 만들었다: enp0s31f6 192.168.1.102/24, route-metric 500,
  호스트 경로 192.168.1.51/32 metric 50, never-default, IPv6 off, MTU 1500, autoconnect.
  카메라만 enp0s31f6로 가고 나머지 192.168.1.x(라이다)는 enp4s0 그대로다.
  `preflight.sh`는 카메라 NIC을 커널 경로(`ip route get 192.168.1.51`)로 찾는다.
  확인: ping 0 % 손실, aravis `Hikrobot-MV-CS023-10GC-DA8931779`, `bench --secs 15` 20.99 FPS PASS
  (compute p95 45.2 ms, skip 0), 라이다 수신 유지.
- `live`/`replay`는 **화면 녹화 `a2_overlay.mp4`를 기본으로 남긴다**(`--no-video`로 끔). 화면과 같은
  프레임(3출력마다)을 재사용하고 인코딩은 별도 스레드(`AsyncVideoWriter`, 큐 8, 넘치면 버리고 집계)라
  모델 루프를 막지 않는다. 7 FPS 실시간 재생. 20 s 실물 확인: 132 프레임, drop 0.
- 같은 20 s live(화면+녹화+bag)는 20.27 FPS, compute p95 51.4 ms로 21 FPS FAIL. 화면 없는 bench는 PASS라
  비용은 화면 표시(같은 GPU) 쪽이며 녹화 추가분은 거의 없다(10/07 화면만 run 20.21 FPS).
- 실외 주간에 노출 3000 µs는 모델 band 포화 45.9 %(`OVEREXPOSED`)였다. 9/27 실외는 600–1300 µs.

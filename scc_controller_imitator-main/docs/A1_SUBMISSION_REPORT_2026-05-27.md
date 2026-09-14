# A1 CHALLENGE 제출 준비 작업 보고서 (2026-05-27)

팀 Hanyang AutoHYU · controller_imitator (camera + ego + raceline → steer/pedal).
본 문서는 2026-05-27 세션의 엔지니어링 히스토리·결정·검증 상태를 기록한다.
표기: [확정] 검증됨 · [진행] 미완/부분 · [미검증] 코드·수치는 맞으나 sim 라이브 미확인.

---

## 1. 개요

| 항목 | 내용 |
|---|---|
| 목적 | 본선(scc) + 예선(qualifier) 제출물 + 운영 스크립트 정리, 단일 .pt 배포, 레이턴시 절감 |
| 본선 브랜치 | `minimal` (HEAD `6c6f5c5`, origin 동기화) |
| 통합 브랜치 | `minimal-b` (HEAD `89a9418`) — scc 4종 + 예선(prof635-beat) 통합 |
| 예선 원본 | `qualifier-beat-minimal` (검증된 기존 배포) |
| 배포 단위 | A1E2EModel 통합 단일 `.pt` (규정 §1.1 state_dict 하나) |

---

## 2. 본선 제출 구조 — 단일 .pt (E2E)

규정 §1.1(단일 state_dict) / §1.3(단일 미분 그래프)에 맞춰, 분리 모듈을 통합한
`A1E2EModel` 한 그래프로 배포.

- 그래프: seg backbone(EfficientNetV2-S tokenizer) → a_2(mf20 21f, in-lane 앞차 vL_rel·dist·존재) → f_1(main #654 frozen, waypoint/steer) → f_2(side, pedal residual delta) → learned temporal gate → `final pedal = clamp(base + g·delta)`.
- [확정] `build_and_load_scc` 가 분리 ckpt(seg/a2/main/side/gate) → 단일 `A1E2EModel.state_dict()` 로 묶음. reload missing=0 unexpected=0.
- [확정] `closed_loop_node.py` 에 `E2E_CKPT` 로더 추가: 단일 .pt → A1E2EModel 인스턴스 → `backbone/a2/side/gate` 서브모듈을 closed_loop 의 `iy_backbone/a2_v6_model/model/_gate_net` 로 참조 교체. 동일 클래스·동일 가중치라 분리 로드와 forward **수치 동일** (maxdiff steer 0, pedal 1.8e-7).
- [확정] 모듈 ckpt 전면 제거 — 배포물은 단일 .pt 4개뿐. 단일 .pt 만으로 노드 init(모델 로드·raceline·pedal map) → ready 확인.
- [제약] 단일 .pt 176MB/개 → GitHub 100MB 한도 초과 → **git 미추적, 별도 제출물**. 실행 전 `submit/` 에 배치 필요.

---

## 3. 모델 옵션 (multicar prefix 5종)

| prefix | 모델 | side loss | 비고 |
|---|---|---|---|
| `scc_p3_dbspd2` | a_2 db_spd + side p3(vx-log) | 0.0344 | 본선 최종(중간 프로파일) |
| `scc_p5_dbspd2` | a_2 db_spd + side p5(vx-log) | 0.0322 | 보수(s_min7/headway2.0), 일찍·세게 제동 |
| `scc_p3_dbspd` | a_2 db_spd + side p3(preVxlog) | 0.0427 | vx-log 전 비교 |
| `scc_p3_db` | a_2 db + side db | — | speedway 인지 전 비교 |
| `qualifier` | prof635 distbrake + a_2 2f + beat gate | — | 예선 (minimal-b: QBEAT / minimal: 구 iy) |

공통(scc): NO_GATE raw delta + distill gate(db) + mf20 a_2(21f) + main #654, 모두 단일 .pt 내장.

---

## 4. SCC profile imitation

- Expert: HPIPM 기반 SCC(adaptive cruise) MPC. 시뮬 합성 데이터에서 (t_headway, s_min) profile 별 pedal label 생성. p1=공격(0.4/2.0) ~ p5=보수(2.0/7.0).
- 활성거리 vx-종속 log: `s_min_eff = s_min + α·ln(1 + v_ego/v0)` (α=6, v0=5). 고속에서 일찍 제동, 저속 완만.
- f_2(side) 학습: `--no-gate --delta-mode raw --pedal-suffix p{N}`, db_spd i2_mem, warm-start prior best, 25 epoch.
- [확정] p5(vx-log) 신규 학습: sangam_lead 2종 라벨 vx-log 재생성 → ep25 best **val_pedal 0.0322** (p3 0.0344 대비 낮음). DriverB(lead 7%, ego v0=0)·pertO/pertX(no-lead)는 vx-log 무의미 → 기존 유지.

---

## 5. 예선 qualifier 정정 ([확정] 긴급 수정)

- 문제: minimal 의 `qualifier` 가 **iy v25k-B4**(IyModel)로 잘못 번들됨. 어제 실제 예선은 `qualifier-beat-minimal` 의 **prof635-beat**. 모델 자체가 달라 거동 상이.
- 원인: dev(acc-test) multicar 주석 "qualifier = iy v25k-B4" 를 따랐으나 실배포와 불일치.
- 수정(minimal-b): qualifier-beat-minimal 구성을 **verbatim 포팅** — Prof635SideModel(distbrake_v1, brake_only) + A2Model **2-frame**(a2_v7b) + learned **beat** gate(`learned_temporal_gate_mlp_beat`) + main654 + seg. `QBEAT=1` 플래그로 mf20→2f 전환 + 2-frame forward(encoder.proj 융합) + 외부 beat gate.
- [확정] 노드 init: 2-frame a_2(v7b) miss0 unex0 + beat gate K16 + distbrake side 로드 → ready.
- [확정] minimal 의 잘못된 iy qualifier 는 사용자 지시로 미변경(취소).

---

## 6. 레이턴시 최적화 ([확정])

- 발견: prof635 추론이 backbone 을 2회(`fc`=현재, `fp`=과거) 돌렸으나 mf20 deque 는 raw `fc` 만 사용 — `fp`/`fp_rot` 는 **dead compute**.
- 조치: prev-frame backbone 제거(2→1회) + `cudnn.benchmark=True`. 둘 다 무손실(거동·수치 불변).
- 마이크로벤치(250×1280, GPU): backbone 11.0ms(×2) → 4.9ms(×1), **2.3×**.
- 실측(run_multicar.sh, post-opt): dt_ms median 22ms / mean 24ms (n=11). ~45Hz, 카메라 20Hz budget 대비 여유.
- 남은 ~17ms(비-backbone): raceline sample, mf20 deque Python 루프, gate GPU↔CPU `.item()` 동기화, ROS 오버헤드 — 추가 절감 후보.
- fp16 autocast 는 i2_mem 수치 드리프트 위험 + 이득 미미(5.5→4.8ms)로 **제외**.

---

## 7. 운영 — multicar + 커맨드라인 override ([확정])

- `run_multicar.sh`: ini 의 [CarN] 마다 moraibridge + closed_loop 1세트 기동. moraibridge devel 자동 source.
- ini 안 건들고 env override(우선순위 `<CarN>_X` > `X_ALL` > ini):
  - 모델: `Car1_PREFIX=scc_p5_dbspd2` / `PREFIX_ALL=qualifier`
  - 트랙: `Car1_TRACK=speedway` / `TRACK_ALL=speedway` (sangam|speedway)
  - 난이도: `Car1_MAP=high` / `MAP_ALL=high` (mild|high)

---

## 8. 검증 상태

| 항목 | 상태 |
|---|---|
| 단일 .pt == 분리모듈 forward 등가 | [확정] maxdiff 1.8e-7 |
| 단일 .pt 만으로 노드 init + ready (scc/qualifier) | [확정] |
| QBEAT 예선 = qualifier-beat-minimal 모듈·코드 동일 | [확정] 로드/코드 verbatim |
| 레이턴시 절감 | [확정] 마이크로벤치 + 실측 dt_ms |
| 각 옵션 sim 라이브 주행 거동(before/after) | [미검증] 세션 중 sim 토픽 미스트림 |

---

## 9. git 상태

| 브랜치 | HEAD | 핵심 |
|---|---|---|
| `minimal` | `6c6f5c5` | scc 4종 단일 .pt + E2E 로더 + 레이턴시 opt + cmdline override (qualifier=구 iy) |
| `minimal-b` | `89a9418` | minimal + 예선 prof635-beat 통합(QBEAT) |
| `qualifier-beat-minimal` | (기존) | 검증된 예선 원본 |
| `acc-test` | (dev) | 학습/합성 인프라 |

---

## 10. 남은 작업 / 리스크

- [필수] 본선 전: `submit/*.pt` 4개를 테스트 머신 `submit/` 에 배치(미추적). moraibridge catkin_make 1회.
- [권장] 각 prefix(scc 4 + qualifier) sim 1회 정성 주행 확인 — 라이브 거동 미검증분 해소.
- [후보] 비-backbone 레이턴시: deque 벡터화 + `.item()` 동기화 묶기 (수 ms).
- [정리] minimal 의 iy qualifier 잔존(취소 상태) — 본선용으로 정리 여부 추후.

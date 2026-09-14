# Paper proposal — lab naturalistic driving dataset 활용 갈래

**기록 시점**: 2026-05-31 (A1 CHALLENGE 본선 종료 후, paper 작업 시작 전)

이 문서는 본선 작업 (`scc_controller_imitator`) 끝난 후 **새 cycle 의 paper 작업 시작점**.
페이퍼 작업 agent 가 *"이 dataset 으로 어떤 paper 갈래 가능 / 어디부터 시작"* 의 starting context 로
쓰도록 작성.

본선 직전의 *"논문화·연구 발전 방안 냉정 평가"* (commit `a5fe044`, `docs/PAPER_PROPOSAL_2026-05-27.md`)
는 본선 작업 (시뮬 + profile-conditioned IL) 기반의 평가였음. 이 문서는 **lab 자체 보유 naturalistic
dataset 갈래** 의 새 proposal.

참고: 9 일 본선 회고는 [BLOG_READABLE_2026-05-19_to_28.md](BLOG_READABLE_2026-05-19_to_28.md).
데이터 재생성 가이드는 [DATA_REGENERATION.md](DATA_REGENERATION.md).

---

## 1. 요약 한 단락

연구실이 자체 보유한 **N=10 driver × 6 hour/driver = 60 hour 총** 의 서울 도심 (광진 / 강남) 일상 주행
데이터셋이 paper-able. 같은 시간대 (점심) / 같은 차량 (2023 아이오닉 5) / 같은 navigation route /
동일 sensor stack (mobileye aftermarket + Hyundai 순정 ADAS + NovAtel PwrPak7-E2 + 추정 DBC) 로
**controlled naturalistic study** 의 unique design point. *4 perception source* (mobileye-radar
raw + mobileye-camera raw + Hyundai-fusion + PwrPak7+lane marker) 동시 보유는 literature 에 흔치
않은 setup. 단 *N=10* / *lab member bias* / *점심시간만* / *DBC parsing 정확성 미검증* 의 한계.

**main paper 후보** — `Controlled Naturalistic Driver Phenotyping with Multi-Source Perception
(paper C)`. 본선 profile paper (paper A) 의 sister paper + thesis 의 main chapter.

**현재 status** — 데이터셋 *capability inventory* 완료, *DBC parsing sanity check* (priority 1)
다음 step. 그 결과 paper 진짜 가능성 확정.

---

## 2. 데이터셋 spec — 사용자 확인 결과 종합

### 2.1 확정 (사용자 답변)

| 항목 | 값 |
|------|------|
| **driver 수 N** | **10 (lab 인원)** |
| **per-driver 시간** | **6 hour** (1 인당 합) |
| **총 시간** | **60 hour** |
| **수집 기간** | 약 2 개월 |
| **시간대** | **점심시간** (대략 11~14 시 추정) — 시간대 confound 제거 의도 |
| **차량** | **2023 아이오닉 5, 모든 driver 동일 차량** |
| **지역** | **서울 광진 / 강남 일대** 도심 일상 주행 |
| **route** | **navigation 단위 동일** (차선까지는 driver 자유) |
| **차량 CAN** | Hyundai 순정 — SCC / LFA / FCA / HDA state 등 (DBC 확보, *어둠의 경로*) |
| **mobileye** | **aftermarket 장착** — Radar Object / Camera Object / Camera Lane 신호 |
| **GNSS / IMU** | **NovAtel PwrPak7-E2** (RTK cm-level + tactical-grade IMU) |
| **HD map** | **NGII** 22,000 km (사용자 working tracks 에 명시) |

### 2.2 확인 필요 (다음 sprint 의 inventory)

| 항목 | 우선순위 | 1 trip 안에 확정 가능? |
|------|---------|---------------------|
| DBC parsing 정확성 (mobileye radar / camera / lane 의 plausibility) | **P1** | ✓ |
| PwrPak7 RTK fix rate + 후처리 정확도 | **P2** | ✓ |
| Hyundai 순정 ADAS DBC 의 SCC / LFA / FCA state 신호 list 확정 | P3 | ✓ |
| mobileye 세대 / EyeQ 버전 | P3 | ✓ |
| NGII HD map matching pipeline 준비도 (광진/강남) | P4 | △ (1 주) |
| 같은 route 의 정확한 정의 (1 경로? candidate set?) + 각 trip 의 conformity ratio | P5 | ✓ |
| mobileye + Hyundai + PwrPak7 의 3 source timestamp 동기화 | P6 | ✓ |
| trip metadata (날씨 / 출발지-목적지 sequence) | P7 | ✓ |

### 2.3 핵심 capability — *"why this dataset is unique"*

- **controlled naturalistic** — 시간대 + 차량 + route + sensor 다 control, *driver 만 random*. NDS literature 의 SHRP2 (*"all random, huge N"*) 와 정반대 design point
- **dual ADAS vendor** — mobileye aftermarket + Hyundai 순정. 같은 frame 의 두 vendor perception 직접 비교
- **mobileye 의 raw modality 분리** — Radar Object / Camera Object 별도 신호. *modality-level disagreement* 분석 가능
- **cm-level localization** — PwrPak7-E2 RTK + post-process. 도심 multipath 환경에도 lane-level
- **HD map 매칭** — NGII 22,000 km. *"같은 도로 segment"* 분석 가능
- **lane polynomial** — mobileye Camera Lane → lane-relative trajectory 직접 계산

이 5 가지 capability 동시 보유 = literature 에서 흔치 않음.

---

## 3. paper 갈래 — 3 슬롯

### 3.1 paper A — *profile-conditioned longitudinal IL* (본선 직계)

본선의 SCC profile p1~p5 (`(t_headway, s_min)`) 를 main contribution 으로 한 paper.
- **본선 작업의 자연 paper** — 이미 시뮬 데이터 + 학습 ckpt + closed_loop 검증 다 있음
- 이전 평가 (`a5fe044` commit) 에서 *"5 후보 기여 중 (ii) profile-conditioned longitudinal IL 만 잠재 novelty
  (ITSC/IV 1차, 보강 후 T-IV)"* 라고 짚었음
- contribution: profile parameterization + IL framework + closed_loop 성능
- venue: **ITSC / IV 1차 → 보강 후 T-IV 도전**

lab dataset 의 **angle 7** (raceline 검증 / external validation) 으로 extension 가능.

### 3.2 paper B — *sensor fusion + vendor disagreement* 평가

mobileye / Hyundai / radar / camera 의 multi-source perception 의 *agreement / disagreement* 평가
+ driver action 의 mapping.

- contribution: 4-source 의 frame-by-frame agreement matrix, *modality* + *vendor* 두 갈래
- **단독 paper vs paper C 의 contribution 2** — 분량 / venue 결정
- venue 후보: **IV / ITSC 1차 → T-VT / T-ITS sensor fusion track**

### 3.3 **paper C** — *Controlled Naturalistic Driver Phenotyping with Multi-Source Perception* (**main paper 후보**)

가장 강한 갈래. lab dataset 의 모든 capability 통합:
- methodological — controlled NDS design point (vs SHRP2 의 random-everything)
- empirical — driver phenotyping with multi-source perception
- applied — *어느 ADAS 신호가 individual style 의 predictor 인가* (downstream — personalized ADAS)

분량 따라 **paper C-1 (perception 분석) + paper C-2 (driver phenotyping)** split 가능.
- venue 후보: **T-IV / T-ITS 도전 가능** (한 번에 안 되면 IV / ITSC 1차 → 보강 후 도전)

---

## 4. angle 1~15 표 — paper 별 contribution 후보

paper-able 각도. 각 paper 의 contribution 으로 어떤 angle 들을 묶을지 결정 필요.

| # | angle | data 가용성 | novelty | venue 후보 | paper 배치 |
|---|-------|-----------|---------|----------|----------|
| 1 | driver × ADAS interaction (SCC ON/OFF 시 차이) | ✓ | 중~강 | ITSC / IV | paper C contribution |
| 2 | Hyundai SCC vs 본선 profile 비교 | ✓ | 중 | paper A 의 validation | paper A extension |
| 3 | takeover behavior (LFA / HDA override) | ✓ event-rich | 강 | ITSC / IV | paper C contribution |
| **4** | **dual perception disagreement** (vendor 간) | ✓ event-rich | **매우 강** | T-IV / T-ITS 도전 | paper B / C |
| **5** | **takeover ~ disagreement explanation** (vendor disagreement → override 예측) | ✓ event-rich | **매우 강** | IV / ITSC, T-IV 도전 | paper B / C |
| **6** | **route-controlled segment-conditional driver style** (같은 segment 의 driver 별 행동) | ✓ (route + HD map) | **강 (route fix 후 격상)** | T-IV / T-ITS | paper C contribution |
| 7 | raceline 검증 (본선 profile p1~p5 ↔ 실제 driver 분포) | ✓ | 중 | paper A validation | paper A extension |
| 8 | takeover spatial map (도시 도로의 override 분포) | ✓ | 강 | Transportation Research Part C / IV | paper C contribution |
| **9** | **controlled inter-driver comparison** (시간 / 차 / route / sensor 다 fix, driver 만 random) | ✓ | **강 — methodological 단독 contribution** | T-IV / T-ITS | paper C 의 main framing |
| 10 | (cognitive load / 점심 직전 직후) | ✓ but niche | 약 | Human Factors / 무시 | skip |
| 11 | whole-trip replication (같은 route 의 multiple replication 의 mixed-effect model) | ✓ | 강 | IV / ITSC | paper C contribution |
| 12 | driver-navigation interaction (navigation 지시 vs 실제) | ✓ | 중 | niche | paper C side observation |
| **13** | **mobileye 안 modality disagreement** (radar ↔ camera within mobileye) | ✓ if raw signals OK | **강** | T-VT sensor fusion / T-ITS | paper B / C |
| **14** | **4-source cross-validation** (mobileye-radar + mobileye-camera + Hyundai + PwrPak7+lane) | ✓ if all 4 OK | **매우 강** | T-IV / T-ITS | paper B / C 의 main perception contribution |
| **15** | **lateral fingerprint** (PwrPak7 cm-level + camera lane → lane-relative trajectory) | ✓ if cm-level OK | **강** | T-IV / driver behavior venue | paper C contribution |

**가장 강한 angle 5 개** = **4 / 5 / 6 / 9 / 14**. 이 5 개 묶음이 paper C 의 골격.

intra-driver variability angle (이전 brainstorm 의 angle C) 은 **시간대 fix 로 사용 불가**.

---

## 5. paper C 의 framing — *current draft*

### 5.1 title 후보

```
Controlled Naturalistic Driver Phenotyping with Multi-Source Perception:
  Dual Vendor ADAS, Raw Modality Disagreement, and cm-level Lane-Relative
  Trajectory on Identical Urban Routes (N=10, 60 hours)
```

또는 short version:

```
Driver Phenotyping under Controlled Naturalistic Conditions:
  Four Perception Sources, Identical Routes (N=10, urban Seoul)
```

### 5.2 abstract first paragraph (draft)

```
Naturalistic driving studies (NDS) typically maximize sample size at the cost
of confound control. We invert this trade-off: time-of-day, vehicle, route,
and sensor stack are controlled, while only the driver varies. Our sensor
stack provides — to our knowledge — the most complete naturalistic perception
multi-source dataset to date: four perception sources (mobileye-radar raw,
mobileye-camera raw, Hyundai-OEM ADAS fusion, NovAtel PwrPak7-E2 cm-level
with camera lane markers) collected on identical urban routes (Seoul
Gwangjin/Gangnam, lunchtime, 2023 IONIQ 5, N=10 drivers, 60 hours total).
We use this controlled setup to (1) characterize cross-source perception
agreement, (2) phenotype driver style with lane-relative fingerprint, and
(3) map takeover behavior onto spatial road features.
```

### 5.3 contribution claim — 3 개

1. **methodological** — *controlled NDS design point*. NDS literature 의 SHRP2 trade-off 의 정반대.
   *"all variables controlled + small N + clean individual difference"*
2. **perception multi-source dataset** — 4-source frame-by-frame agreement matrix + driver action
   linkage. literature 에 흔치 않은 raw modality + vendor + lane-relative truth 의 동시 setup
3. **driver phenotyping** — lane-relative trajectory fingerprint (lateral fingerprint) + segment-conditional
   style (route-controlled) + takeover spatial map. *"individual difference 가 main 분석 대상이라
   N=10 limitation 의 일부 해소"*

### 5.4 paper structure (draft)

```
1. Introduction
   - NDS literature: SHRP2 / 100-car / Toyota MPM 의 design trade-off
   - controlled NDS 의 design point gap
   - 4-source perception 의 gap
   - 본 paper 의 3 contribution

2. Related Work
   - NDS taxonomy
   - dual ADAS / multi-vendor perception evaluation
   - driver style / phenotyping (SHRP2, Doshi & Trivedi, etc)
   - takeover behavior literature
   - lane-relative trajectory analysis

3. Dataset
   - acquisition (사용자 답변 그대로)
   - sensor stack 상세 (mobileye + Hyundai + PwrPak7-E2)
   - controlled variables 와 random variable 의 명시
   - data processing pipeline (DBC parsing, HD map matching, 4-source align)

4. Methodology
   4.1 4-source agreement matrix
   4.2 lateral fingerprint extraction (PwrPak7 + lane polynomial)
   4.3 segment-conditional driver phenotyping
   4.4 takeover event detection + spatial map

5. Results
   5.1 cross-source perception agreement / disagreement
   5.2 driver phenotyping clusters
   5.3 takeover spatial map
   5.4 individual difference statistical analysis (mixed-effect model)

6. Discussion
   - controlled design 의 strengths / limitations
   - N=10 + lab bias 의 generalization
   - lunchtime fix 의 ecological validity
   - lane choice 자유의 의미 (random variable)

7. Conclusion + Future Work
   - 데이터셋 확장 (drivers, time-of-day, weather)
   - dataset release 가능성 (privacy)
```

---

## 6. 한계 분석 — *honest discussion 부분*

| 한계 | 영향 | 대응 |
|------|------|------|
| **N=10 lab member bias** | generalization claim 약. reviewer first-round critique 거의 확실 | *individual difference 가 main 분석 대상이라 design choice* 로 framing + future work 에 N 확장 |
| **시간대 점심으로 fix** | external validity ↓ (출퇴근 / 야간 모름). intra-driver variability angle 포기 | *deliberate confound control* 로 framing. discussion 에 명시 |
| **route navigation level 동일, lane 자유** | lane choice 가 nuisance vs driver style 의 한 dimension 인지 분리 어려움 | lane choice 자체를 *"lateral fingerprint 의 dimension"* 로 분석 |
| **차 1 대 동일** | vehicle dynamics confound 제거 (positive). 단 *driver × vehicle interaction* 측면 모름 | strength 로 framing |
| **DBC parsing 정확성 미확정 (어둠의 경로)** | 모든 분석의 신뢰도 단일 결정 인자 | **inventory P1 sanity check 필수**. plausibility 검증 안 거치면 paper 전체 fragile |
| **mobileye 의 radar / camera timestamp** | 다른 cycle 이면 frame-by-frame matching mis-align → angle 13/14 fake disagreement | **inventory P6 sanity check** + interpolation 또는 nearest-neighbor align |
| **mobileye internal fusion vs raw modality** | 우리가 받은 게 fusion 전 raw 인지 후 인지 confirm 필요 | datasheet / EyeQ documentation 확인 |
| **camera lane polynomial 의 coordinate frame** | PwrPak7 자차 위치와 align 위해 정확한 frame transformation 필수 | 검증 필요 |
| **PwrPak7 후처리의 시간 비용** | 60 hour Inertial Explorer 후처리 며칠 | RTK real-time vs 후처리 정확도 비교, 후처리 안 해도 충분한지 평가 |
| **HD map matching** | NGII 광진/강남 부분 정확도 변동 가능 | inventory P4 |
| **점심시간 dataset 의 sample bias** | light traffic + 식사 경로 + 비교적 여유 — *"worst case driving"* 안 포함 | discussion 명시 |
| **driver action override 의 GT 정의** | wheel torque threshold / brake pedal angle 의 *override* quantitative criterion 정해야 | methodology 에 명확히 정의 |
| **광진/강남 vs sangam (본선 시뮬) 의 domain gap** | paper A 와의 narrative 연결 약함 | paper A 의 *external validation* 으로만 활용, 본선 결과의 raw transfer 안 시도 |

---

## 7. 다음 sprint — inventory priority + 1 일 sanity check

paper 진행 결정 인자 = **DBC parsing 의 정확성 (P1)**. 다른 priority 는 그 후.

### 7.1 inventory priority 재정렬 (route fix + mobileye raw 신호 확정 후)

```
P1 (CRITICAL, 1 일) — DBC parsing sanity (lab member 1 명 + 30 분 segment)
  - 1 trip 의 30 초 segment 추출
  - raw camera image (저장돼 있으면) overlay
  - mobileye Camera Object 박스 시각화
  - mobileye Camera Lane polynomial overlay
  - mobileye Radar Object top-down plot
  - Hyundai SCC lead 거리 비교
  - PwrPak7 자차 위치 + IMU + 속도 시각화
  → 모든 신호 plausible 한지 시각으로 확정

P2 — PwrPak7 RTK fix rate + 후처리 결과 정확도 (RTK fix rate 비율, 후처리 vs real-time 차이)

P3 — Hyundai 순정 ADAS DBC 의 신호 list 확정 (SCC state / LFA active / FCA warning / HDA target)

P4 — NGII HD map matching pipeline (광진/강남) 준비도 + 정확도

P5 — 같은 route 의 정확한 정의 + 각 trip 의 conformity ratio (route 일치 비율)

P6 — 3 source timestamp 동기화 (mobileye + Hyundai + PwrPak7)

P7 — trip metadata (날씨 / 출발지-목적지 / driver demographic)
```

### 7.2 P1 sanity check 의 구체 plan

```
input:
  - 1 trip 의 30 초 segment (점심시간 임의 trip)
  - raw CAN log (.blf or .asc or .mat)
  - PwrPak7 binary log
  - (있으면) raw camera image

output:
  - 1 페이지 시각화 PDF:
    * top: raw camera image + mobileye Camera Object box overlay
    * middle: top-down plot (PwrPak7 자차 위치 + mobileye Radar Object + Hyundai SCC lead 위치 + 비교)
    * bottom: time series (속도, accel, brake, steering, FCW, SCC state, LFA active)
  - sanity assertion:
    * mobileye Camera Object box 가 raw image 위에 plausible 위치
    * Radar Object 거리가 0~100 m 정상 범위
    * Camera Lane polynomial 이 차선 위치 정확히 fit
    * Hyundai SCC lead 와 mobileye lead 의 cross-check 거리 차이 < 5 m (같은 lead 면)
    * PwrPak7 RTK fix rate > 90% in 도심
    * timestamp align 확인

소요 시간: 1 일 (lab member 1 명)
```

### 7.3 P1 결과별 paper 가능성 분기

| sanity 결과 | paper 가능성 | 다음 action |
|------------|------------|------------|
| 모든 신호 OK | **paper C 의 main 4 angle (4/5/6/14) 다 가능. paper C 진행 확정** | paper C outline draft, advisor pre-review |
| mobileye Camera Object 만 가짜 | angle 4/13/14 fragile. paper B 분량 ↓ | angle 1/3/6/8/15 만 가능 → paper C-light |
| Radar Object 만 가짜 | angle 13 (modality disagreement) 못 함. angle 4/14 의 quality 일부만 | paper C 의 perception contribution 축소 |
| Camera Lane 만 가짜 | angle 15 (lateral fingerprint) 못 함. paper C 의 phenotyping contribution 큰 손실 | paper 가능성 재평가 |
| PwrPak7 fix rate 낮음 (50% 미만) | lane-level analysis 의 신뢰도 의문. angle 6/8/15 영향 | RTK 후처리 또는 보조 sensor 검토 |
| 다 가짜 | DBC reverse engineering 필요 (수 주~수 개월) | paper 시작 전 데이터 신뢰성 확보 우선 |

---

## 8. publication 갈래 + venue

### 8.1 venue 분류 + 적정도

| venue | tier | 적정 paper | 1 차 round 도전 |
|-------|------|---------|--------------|
| **ITSC** (IEEE Intelligent Transportation Systems Conference) | conference | paper A / C / B | ✓ — 안전 1 차 |
| **IV** (IEEE Intelligent Vehicles Symposium) | conference | paper A / C / B | ✓ — 안전 1 차 |
| **T-IV** (IEEE Transactions on Intelligent Vehicles) | journal | paper C / A 보강 | △ — paper C 도전, paper A 보강 후 |
| **T-ITS** (IEEE Transactions on Intelligent Transportation Systems) | journal | paper C / B | △ — paper C 가능성, paper B 분량 따라 |
| **T-VT** (IEEE Transactions on Vehicular Technology) | journal | paper B (sensor fusion track) | △ |
| **Transportation Research Part C** | journal | paper C 의 angle 8 (takeover spatial) | △ — interdisciplinary |
| **WACV / CVPR workshop** | workshop | paper B 의 perception focus | ✓ — secondary outlet |
| **Applied Ergonomics / Human Factors** | journal | angle 12 (driver-navigation interaction) — niche | ✗ — main paper 아님 |

### 8.2 publication sequence draft

```
2026-06 ~ 07  — inventory sprint (P1 sanity + P2~P5 1 주 sprint) + paper outline draft 3 개
2026-07 ~ 09  — paper A (본선 profile) IV/ITSC 1 차 submission
2026-08 ~ 10  — paper C outline → advisor pre-review → draft writing
2026-10 ~ 12  — paper C IV/ITSC 1 차 submission, paper B subset 결정
2027-01 ~ 06  — paper C revision → T-IV / T-ITS 도전, paper B IV/ITSC
2027-06+     — thesis writing (paper A / B / C 의 chapter 통합)
```

### 8.3 thesis 와의 연결

사용자 thesis novelty axis:
> *"Augmented-state UKF → covariance P → Stochastic MPC (HPIPM) chance constraint.
>  Three target papers (IEEE T-VT / T-IV / T-ITS). Message = slip-tolerant path tracking."*

lab dataset paper 갈래는 thesis 의 *slip-tolerant path tracking* 과 **직접 연결 약함**.
가능한 연결:
- paper C 의 *individual difference* → *"chance constraint 의 bound 가 driver 별로 다를 수 있다"*
  의 한 다리. 그러나 약함
- paper A (profile-conditioned IL) → 본선 *"profile 별 chance bound"* 의 직접 적용. 더 자연 연결

→ thesis 의 main message 는 **UKF/SMPC** 그대로, lab dataset paper 는 *"perception-control 연계의
empirical 측면"* 의 chapter 1~2 개로. paper A 가 thesis 의 main, paper C 가 supporting.

---

## 9. 본선 작업과의 narrative 연결

본선의 SCC profile p1~p5 (`scc_p3_dbspd2`, `scc_p5_dbspd2`) 는 **manual 로 정한 prototype**.
lab dataset 으로:

| 연결 | 의미 |
|------|------|
| **paper A** | 본선 작업 직계, profile-conditioned IL methodology |
| **paper C → paper A** | lab 의 driver phenotyping clusters 가 manual profile p1~p5 와 fit / mismatch 분석 → paper A 의 *external validation* |
| **angle 2** | Hyundai SCC 의 control trajectory vs 우리 profile 의 control trajectory 비교 — paper A 의 *vendor baseline comparison* |

본선 작업의 회고는 [BLOG_READABLE_2026-05-19_to_28.md](BLOG_READABLE_2026-05-19_to_28.md), 특히
**부록 H (a_2 + a_2_aux 의 9 일 진화)** 와 **§ 0.5.8 (SCC profile p1~p5)** 가 paper A 의 detail source.

---

## 10. 참고 자료 (paper 작성 시 reference)

### 10.1 본 repo 안

- [BLOG_READABLE_2026-05-19_to_28.md](BLOG_READABLE_2026-05-19_to_28.md) — 9 일 본선 회고 (paper A
  의 detail + thesis chapter 의 supporting)
- [DATA_INVENTORY.md](DATA_INVENTORY.md) / [DATA_PIPELINE.md](DATA_PIPELINE.md) — 본선 데이터 spec
- [DATA_REGENERATION.md](DATA_REGENERATION.md) — 본선 ckpt cold-start 가이드
- [PAPER_PROPOSAL_2026-05-27.md](PAPER_PROPOSAL_2026-05-27.md) — 본선 직전 논문화 평가 (이 문서 기록
  시점 기준, *profile-conditioned IL 만 잠재 novelty* 라 짚었음. lab dataset 갈래는 그 후 발견)
- [A1_SUBMISSION_REPORT_2026-05-27.md](A1_SUBMISSION_REPORT_2026-05-27.md) — 본선 제출 정리

### 10.2 NDS literature key references (paper 작성 전 review 필요)

- SHRP2 NDS (Naturalistic Driving Study) — Hankey, Antin et al. — *N=3000+, multi-year, all variables random*
- 100-car NDS — Dingus et al. 2006
- Toyota MPM (Toyota Naturalistic Driving Study) — Toyota R&D
- Doshi & Trivedi 2010s — driver attention / intent prediction
- Wang+ T-ITS 2017 — driver style classification
- Lefèvre+ 2014 — personalized ACC
- Kuderer+ 2015 — IRL for driving style
- Naumann+ — counterfactual driving style
- mobileye EyeQ datasheet — Camera Object / Radar Object signal format
- NovAtel PwrPak7-E2 datasheet + Inertial Explorer post-processing manual

paper 작성 agent 는 위 references 의 직접 access 검색 + 인용 비교 필요.

### 10.3 외부 reference 검색 keyword (paper 작성 agent 의 search 시작점)

```
"naturalistic driving study" + "controlled" + "small N"
"dual ADAS" OR "multi-vendor perception" + "evaluation"
"mobileye" + "OEM" + "comparison" + naturalistic
"driver phenotyping" + ADAS + "individual difference"
"takeover behavior" + spatial + naturalistic
"sensor fusion" + radar + camera + "naturalistic data" + driver action
"lane-relative trajectory" + driver + fingerprint
NovAtel + PwrPak7 + naturalistic driving + paper
```

---

## 11. paper 작업 agent 를 위한 즉시 action items

paper 작업 시작 시 다음 순서:

### 11.1 첫 1 일 — sanity check

1. **inventory P1 의 sanity check** — `7.2 P1 sanity check 의 구체 plan` 의 1 페이지 시각화 PDF 만들기
2. 결과 → `7.3` 표의 분기 결정

### 11.2 첫 1 주 — outline draft

3. paper C 의 outline draft (`5.4` structure 기반)
4. NDS literature 의 key references 5~7 편 정독 + paper C 의 *novelty claim* 의 differentiation 1
   문단 작성
5. paper A 와 paper C 의 *external validation* connection 설계
6. paper B 의 단독 vs paper C 의 contribution 2 결정 (분량 따라)

### 11.3 첫 1 개월 — methodology + preliminary result

7. 4-source agreement matrix 계산 (mobileye-radar + mobileye-camera + Hyundai + PwrPak7+lane)
8. lateral fingerprint 추출 pipeline 구현 (PwrPak7 + Camera Lane polynomial)
9. segment-conditional driver phenotyping preliminary clustering (HD map matching 결과 위)
10. takeover event detection + spatial map preliminary
11. preliminary result 의 *"이 paper 진짜 가능"* 의 sanity 결정 → advisor pre-review

### 11.4 그 후

- paper C 진행 확정 시 — 5.4 structure 따라 draft writing
- paper A 와 묶어 IV / ITSC 1 차 submission target

---

## 12. 마지막 — paper 작업 agent 에게의 한 줄

이 dataset 의 *"진짜 unique"* 한 부분 = **mobileye raw radar / camera + Hyundai 순정 fusion + PwrPak7
cm-level + camera lane + identical route + lunchtime + same vehicle** 의 *combination*. 단일 항목은
다 literature 에 있음 — *combination* 이 흔치 않음. paper C 의 first paragraph 의 *"to our knowledge"*
claim 의 근거 = 이 combination 의 unique 성. 그것이 진짜인지 inventory P1 sanity check 로 먼저 확정.

**inventory P1 결과** 가 모든 paper 갈래의 진짜 가능성 결정. 다음 step 은 그것부터.

---

*문서 작성: 본선 + 정리 마지막 conversation 직후 (2026-05-31). 다음 cycle = paper 작업 cycle.*

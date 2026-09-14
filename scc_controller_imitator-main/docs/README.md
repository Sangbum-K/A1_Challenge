# SCC Controller Imitator — 리포지토리 오리엔테이션 (에이전트 진입점)

이 문서는 **이 repo 를 처음 분석하는 AI 에이전트(또는 사람)**가 먼저 읽는 진입점이다.
"어느 모듈이 어떻게 학습되고, 테스트는 뭘 하면 되는지"를 여기서 출발해 파악한다.

> 작성 2026-05-25 (배포 갱신 2026-05-26). 현행 배포 모델 = **prof635 + distbrake_v1** (`prof635_side_distbrake_v1.pt`, delta_mode=**brake_only**, dt0p1s raceline). v5(signed)는 legacy. 과거 버전 문서는 `docs/archive/` 에 보존.

---

## 0. 한 줄 요약

자율주행 레이싱(트랙: sangam=상암, speedway)에서 **카메라 + 자차 상태 + raceline → 조향(steer) + 페달(accel/brake)**
을 내는 imitation 제어기. ROS closed-loop 으로 실차/시뮬 주행. SCC = Smart Cruise Control (앞차 추종/제동 포함).

---

## 1. 모델 스택 (perception → control, 4 모듈)

데이터 흐름:
```
카메라(curr,prev) → [seg backbone] → i1 토큰(40,256) → [a_2 AuxHead] → i2_mem(40,128) + a2_logits(40,7)
raceline + ego pose → traj6(6,3)                                                          │
ego state → ego6(6)                                                                       │
                         ┌────────────────────────────────────────────────────────────────┘
                         ▼
            [f_1 main #654, frozen]  (traj6, ego6) → steer, base_pedal
            [f_2 side prof635]       (ego6, base_pedal, i2_mem) → pedal delta
            gate g = sigmoid((a2 in-lane obj − 0.3)·5)   (앞차 인지 시만)
            final_pedal = clamp(base_pedal + g·delta, −1, 1)
            steer(deg) = steer·40 ;  pedal → accel=max(p,0), brake=max(−p,0)
```

| 모듈 | 코드 | 역할 | 학습 상태 | ckpt (data/ckpts/) |
|---|---|---|---|---|
| seg backbone | `seg_pretrain_model.py` | EfficientNetV2-S, 카메라→feature. 4-class seg 로 pretrain 후 frozen feature extractor | **frozen** (pretrain 완료) | `seg_pretrain_fx640_v3_backbone.pt` (mIoU 0.921) |
| a_2 (AuxHead) | `a2_model.py` | curr/prev 토큰 fusion → per-column 객체/상대속도 perception. i2_mem + a2_logits(7ch) 산출 | **학습됨** (frozen at f_2 학습) | `a2_v7b_d128_alpha075_best_loss.pt` |
| f_1 (main, #654) | `prof635_side_model.py` (Prof635Main) | controller_imitator #654. traj+ego → steer + base pedal. 이미지 안 봄 | **frozen** | `main_prof654.pth` |
| f_2 (side) | `prof635_side_model.py` (Prof635SideModel) | image+ego+f_1pedal → pedal **delta**(보정, brake_only). 앞차 gate 로 override | **학습 대상** | **배포=`prof635_side_distbrake_v1.pt`** (legacy v5/v6) |

상세: [ARCHITECTURE.md](ARCHITECTURE.md)

---

## 2. 리포지토리 지도

```
src/model/image_models/   seg_pretrain_model · a2_model · prof635_side_model (현행) · iy_model(legacy) · image_scc_models_v2(legacy)
src/datamodule/           prof635_dataset(현행) · seg_pretrain_dataset · a2_pair_dataset · scc_synth_*(prebake) · controller_imitator_aug
src/utils/                prof635_infer · raceline_search · pedal_label · pedal_map
scripts/                  train_* · 캐시빌드(compute_i1_tok_cache, build_i2_cache, build_inlane_gt_cache, gen_ttc_pedal_labels, generate_image_seg_masks)
                          closed_loop_node.py + run_closed_loop_prof635.sh · test_/verify_/ablate_/viz_ (검증·진단) · data_pipeline.py(스캔)
data/ckpts/               배포 weight (git 추적, force-add). main_prof654 · a2_v7b · seg_fx640_v3 · prof635_side_distbrake_v1
data/control/             dt0p1s 통합맵 {track}_{map}_{P}_{M}.csv (sangam/speedway × mild/high) · pit_polygon_*.json · pedal_map_a1_f4.csv (git 추적)
data/local/               (타 머신) 로컬 캐시 — gitignored. 이 머신은 AILabDatasetLocal 사용
outputs_multi/            학습 출력(ckpt/log) — gitignored
docs/                     이 문서들 + archive/(과거 버전)
external/                 autohyu_msgs (ROS 메시지, 내제화)
mpc/                      HPIPM MPC fork (SCC 제어 GT 합성용)
```

상세 모듈/스크립트 인벤토리: [ARCHITECTURE.md](ARCHITECTURE.md), 파이프라인: [DATA_PIPELINE.md](DATA_PIPELINE.md)

---

## 3. 데이터 파이프라인 (raw → 학습 캐시)

```
raw 이미지/EGO/OBJECT  ──(SAM)──>  seg masks ──> [seg pretrain, 글로벌]
        │                                              │ (frozen backbone)
        ├──> compute_i1_tok_cache  ──> i1_tok_cache(40,256)  ──┐
        │                                                       ├──> build_i2_cache ──> i2_mem_cache(40,128)+a2_logits
        ├──> prebake_iy_dataset    ──> prebake(traj/ego/target) ┘            (a_2 forward, ego_mot=0)
        ├──> build_inlane_gt_cache ──> inlane_gt(dist/closing)
        └──> gen_ttc_pedal_labels  ──> pedal_label_*_ttc       (BRAKE_MODE=ttc: closing 기반 ACC)
                                   └─> pedal_label_*_distbrake  (BRAKE_MODE=distbrake: 거리-onset 곡선, 현행 실험)
```
- "데이터셋" = (track, policy) 쌍. raw: `AILabSSD/02_Data/57_A1_E2E/00_raw/{track}/{policy}/{run}/{CAMERA_1, EGO_INFO, OBJECT_INFO}`.
- 처리 순서·캐시 명명·경로 portability(SCC_DATA_DIR): [DATA_PIPELINE.md](DATA_PIPELINE.md)

---

## 4. 학습 (어느 모듈을 어떻게)

순서: **seg pretrain → a_2 → f_2** (f_1=#654 는 frozen, 재학습 안 함).
- seg pretrain, a_2 학습은 글로벌(1회). f_2 만 신규 데이터로 반복 학습.
- f_2 학습: `scripts/train_prof635_side.py` (현행). 5 datasets, L1(final_pedal, pedal_label), main frozen.
- 현재 best ckpt 레지스트리: `data/ckpts/CURRENT_BEST.json`.

상세 명령·하이퍼파라미터·현재 best: [TRAINING.md](TRAINING.md)

---

## 5. 테스트 / 주행

**closed-loop (ROS, 실차/시뮬)**:
```
bash scripts/run_closed_loop_prof635.sh <sangam|speedway> [mild|high] [--viz-all]
# 예: ... speedway high   /   ... sangam mild --viz-all   (map 생략=mild)
# 비상: SIDE_OFF=1 (f_2 끔), gate 튜닝: GATE_THRESH=0.3 GATE_SCALE=5.0
```
roscore(또는 시뮬 master) 필요. 카메라 `/app/perc/camera_1` + ego `/app/loc/vehicle_state` 구독, steer/pedal publish.
python 은 런처가 `ml_con` env(torch+rospy) 자동 선택(또는 `PYTHON=` 지정). raceline = `data/control/{track}_{map}_{P}_{M}.csv`(dt0p1s+polygon).

**오프라인 검증** (ROS 없이):
- `test_prof635_infer.py` / `test_prof635_closedloop.py` — 모델 I/O·raceline 파이프라인 정상성.
- `verify_i2_live_vs_cache.py` — live i2_mem 이 학습 캐시와 일치하는지 (cosine).
- `ablate_traj_speed.py` — base=traj-speed artifact 진단.
- `viz_prof635_gradcam.py` — brake attribution grad-cam.

상세: [TESTING.md](TESTING.md)

---

## 6. 현재 상태 / 반드시 알아야 할 함정

- **배포 = distbrake_v1**(`prof635_side_distbrake_v1.pt`, brake_only). legacy: v5(signed/ttc), v6(DriverB 포함). **gate FP brake 버그 fix 됨**(learned_bias +3.02 → fixed_thresh).
- f_2 현행: **gate=fixed_thresh, delta=brake_only**(가속 보정 불가, 제동만). 상세 [REPORT_prof635_v5.md §9].
- **closed-loop i2_mem 은 학습 캐시와 동일하게 계산해야 함**: seg=fx640_v3 + ego_mot=0. 안 맞으면 cos 0.80 으로 어긋나 제동 붕괴.
- **f_1(base)의 brake 는 traj-speed artifact** — inference(raceline nominal)에선 base 가속 → f_2 가 제동 담당.
- **ACC 약함 → 거리 기반 제동으로 선회**: a_2 vL_rel corr 0.40(약함) + f_2 가 vL_rel 무시. closing 대신 잘 추정되는 **거리(dist)** 로 제동하는 `_distbrake` 라벨(distance-onset 곡선) 도입 → distbrake_v1 으로 배포 ([REPORT §9]).

전체 상태·교훈·incident: [STATE_AND_LESSONS.md](STATE_AND_LESSONS.md)

---

## 7. 향후 도구 (계획)

데이터 현황 "마법사" 도구 — 모델 구조 / 학습된 모듈 / 데이터셋·캐시 현황을 한눈에. 설계: [DATA_AUTOMATION_PLAN.md](DATA_AUTOMATION_PLAN.md). (현재 `data_pipeline.py scan` 으로 raw 인벤토리+처리상태 일부 제공.)

---

## 부속 문서 색인
- [ARCHITECTURE.md](ARCHITECTURE.md) — 4 모듈 구조·I/O·dim 상세
- [DATA_PIPELINE.md](DATA_PIPELINE.md) — raw→캐시 DAG, 명명, 경로 portability
- [TRAINING.md](TRAINING.md) — 모듈별 학습법·하이퍼파라미터·현재 best
- [TESTING.md](TESTING.md) — closed-loop + 오프라인 검증
- [STATE_AND_LESSONS.md](STATE_AND_LESSONS.md) — 현재 상태·함정·incident
- [DATA_AUTOMATION_PLAN.md](DATA_AUTOMATION_PLAN.md) — 데이터 현황 도구 설계
- [REPORT_prof635_v5.md](REPORT_prof635_v5.md) — f_2 v5 재현 리포트 (모델 spec 원본)
- DATA_INVENTORY.md, DATA_PIPELINE_GLOSSARY.md — 참고(데이터 목록/용어)
- docs/archive/ — 과거 버전 리포트(v1~v6)·figure (히스토리)

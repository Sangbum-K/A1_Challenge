# SCC Controller Imitator — Architecture & Training Report (v6)

작성일 2026-05-25 · branch `main` · 모델 `src/model/image_models/iy_model.py`

REPORT_v5 (2026-05-24 21:20, v25d 까지) 이후. v25e ~ **v26 (min-ACC)** + a_2 v7~v9.
핵심: **f_2 (Side SCC) 의 결합 방식을 additive residual → lead-gated ACC override 로 변경.**

> 실수/디버깅 타임라인은 `mistakes_report.md`, 데이터 검증은 `data_verification_report.md` 참조.

---

## 0. 모델 명명 규약 (v5 계승 + 갱신)

| 기호 | 모듈 | 역할 | v6 변경 |
|---|---|---|---|
| **f_1** | main transformer (Professor #654 base, `main_no_image=True`) | traj+ego → steer/ax | v0301 단독 추가 finetune |
| **f_2** | Side SCC (cross-attn) | image-grounded brake | **additive → lead-gated ACC override** |
| i_1 | seg backbone token cache (C_b=256) | LateralStripEncoder | — |
| i_2 | a_2 fusion → **fc_proj(256→128)** → image_mem(128d) | spatial fusion + 차원 정합 | **fc_proj 추가 (a_2 v5)** |
| a_2 | per-bin (40,7) [obj, vL_abs, vY_abs, vL_rel, vY_rel, prox, dist] | image → lead 측정 | v7b production |
| B1 / B4 | f_2 cross-attn **layer 수** (B1=1L, B4=4L) | side capacity | 명명 규약 변경 (2026-05-25) |

---

## 1. 변경 요약 (v5 → v6)

| 항목 | v5 (v25d) | v6 (v26) |
|---|---|---|
| **f_2 결합** | `ax = main_ax + α·side_ax` (additive residual) | **`ax = main_ax + g·clamp(b_side − main_ax, max=0)`** (lead-gated ACC override) |
| **f_2 출력 의미** | ax delta (보정량) | **절대 brake ceiling b_side** (init bias +20 → cruise 보존) |
| **f_2 query** | Q = traj_text (traj+state) | **Q = ego state token만 (traj 제거)** — in-lane 은 gate 담당 |
| **lead gate** | 없음 (always-on residual) | **g = sigmoid(a_2 in-lane bin16-25 obj max + gate_bias)** — 전방차 인지 시에만 |
| **f_2 학습 신호** | 최종 ax loss 만 | + **b_side 직접 supervision** (→ target ax step0, dead-start 회피) |
| **i_2 표현** | iy 내부 img_proj(Conv1d 256→128) — 고립 layer | **a_2 fc_proj(256→128)** 로 이동 (i_2 cache 가 128d) |
| **f_2 capacity** | 1L cross-attn | **4L TransformerDecoder (B4)** |
| **a_2 reg loss** | (v5/v6) normalize+mask 누락 → vL_rel std=0 | **v7: fix** (DriverB vL_rel corr +0.40) |
| **TTC augment** | i2-cache path 에서 NO-OP (inlane 미계산) | **GT inlane cache (`--inlane-gt-cache`)** → 실제 fire |
| **TTC 형태** | closing²/(2d) floor | **ACC-style** (kinematic+headway+emergency, TTC-gated, 긴급 -20) |

---

## 2. f_2 결합 재설계 — additive → lead-gated ACC override

### 2.1 additive residual 의 한계 (사용자 지적)
> "side 는 main 이 어느 정도의 ax 를 낼지 모르는데, 목표를 맞추기 위한 변화량을 제시하라니 억지 아니야?"

- `ax = main_ax + α·side_ax` 에서 f_2 는 main_ax 를 입력으로 못 봄.
- f_2 가 동일 traj+ego 를 보긴 하나, main 의 ax 매핑을 **내부 재구성해야** 보정량 산출 → 낭비 + 난이도↑.
- 치명적: **main 을 finetune 하면 main_ax 가 바뀌어 side residual 이 miscalibrate** (calibration 결합).

### 2.2 lead-gated ACC override (v26)
```
ax_step0 = main_ax + g · clamp(b_side − main_ax, max=0)
  b_side : f_2 의 절대 brake ceiling (cross-attn Q=ego, KV=i_2_mem)
  g      : sigmoid(a_2 in-lane(16-25) obj max + gate_bias)  — 전방차 인지 게이트
  clamp(·, max=0) : downward only (f_2 가 절대 가속 안 함)
```
- **decouple**: b_side 는 절대값 → main finetune 과 독립 (min 은 항상 보수적 선택).
- **lead-gate**: 전방차 인지 시에만 ACC 작동 (FP 안전, raw min 보다 강함).
- **identity start**: b_side init bias +20 > cruise ax → 초기 clamp=0 → main 보존. b_side 직접 supervision 으로 brake target 학습.
- **traj 제거 근거**: in-lane 판정이 a_2 obj bin(16-25) gate 로 이뤄지므로, f_2 의 traj query 는 redundant → ego token 만 query (b_side = ego속도 + image closing 기반).

`iy_model.py` flag: `--side-min-arb` (+ `--side-min-arb-w 0.5`).

---

## 3. f_1 (main) — v0301 추가 finetune

- v25c (Professor #654 → ci_aug v0237+v0301+v0302 finetune) 가 B 시리즈 main 의 frozen base.
  - 검증: B4 의 main params 가 v25c 와 **byte-identical** (side 학습 동안 100% frozen).
- **v26 용 main**: v25c 에서 **v0301 단독 추가 finetune** (`v25c2_main_v0301only`).
  - traj_zero_after 20 (v26 정합), lr 1e-4.
  - ⚠️ 데이터 total 5,355 (train 5,088 / val 267) — 단독 분포라 **장 epoch 시 overfit**. val=267 (전부 far) 은 same-distribution → 일반화 지표 아님. best_loss(val) 기준 저장하나 held-out 아님에 주의.

---

## 4. i_2 / a_2 차원 정합 (a_2 v5~v7b)

### 4.1 근본 결함 (REPORT_v5 이후 발견)
- a_2 fusion 출력 256d ↔ transformer 128d 불일치를 잇느라 iy_model 에 `img_proj`(Conv1d 256→128) 끼워둠 → **a_2 pretrain 에도 Professor pretrain 에도 안 속하는 고립·학습불가 layer**.

### 4.2 fix (a_2 v5)
- 256→128 projection 을 **a_2 내부 `fc_proj`** 로 이동 → i_2 cache 가 이미 128d → iy 의 img_proj/img_fusion 우회.
- AuxHead 도 128d 로 재학습.

### 4.3 a_2 v7b (production)
- v5/v6 reg loss 버그(normalize+obj_mask 누락 → vL_rel pred std=0) **fix** (v7).
- focal α=0.75 (pos rate ~5% → 1-pos_rate 근처), sangam upweight.
- DriverB: obj rec 63% / prec 66%, vL_rel corr 0.40, closing rec 49%.
- v8(Gaussian NLL 13ch interleaved) / v9(LDS) 둘 다 개선 실패 → **perception 은 2-frame+perspective 정보 한계, loss tricks 로 개선 불가** 결론. v7b 동결.

---

## 5. TTC augment — NO-OP fix + ACC-style

### 5.1 NO-OP (최대 brake 미학습 원인)
- in-lane 계산이 `if a2_cache_path:` 블록 안에만 존재 → B(i2-cache only)는 `inlane_dist=inf` → ttc 한 번도 fire 안 함 → raw accel target 생존 → "감속 안 함".

### 5.2 fix: GT inlane cache
- `build_inlane_gt_cache.py`: GT aux 에서 `inlane_dist`(bin16-25 closest) + `inlane_closing`(−GT vL_rel) precompute.
- `--inlane-gt-cache` → cache 종류 무관 적용. (predicted dist 과대평가 회피 — 근거리 정확)

### 5.3 ACC-style target
```
safe_decel = max( kinematic(closing²/2d), headway_gap(d0+T·v − d), emergency(TTC) )
clo_gate   = clamp((decay − ttc)/(decay − ttc_emerg), 0, 1)   # 원거리 과제동 완화
aug_ax     = min(raw_ax, −safe_decel),  max_brake 20 m/s²
```
voxel 검증: <5m+접근 −13.7 (std 0.2), outlane +1.79 cruise, 40m@11m/s −6.6→−3.0 (완화).

---

## 6. closed_loop_node 업데이트

| flag | 값 | 비고 |
|---|---|---|
| MODEL_CKPT_IY (B) | `iy_v25k_B4_gttc_accgate_best_loss.pt` | B4 (additive). v26 학습 후 교체 예정 |
| MODEL_TRAJ_ZERO_AFTER (B) | **20** | training 정합 (15→20 수정) |
| MODEL_A2_V6_CKPT | `a2_v7b_d128_alpha075` | i_2 cache 실시간 생성 (d_out=128, 7ch) |
| debug overlay a_2 | **제어 경로 v7b head(i2_mem)** | 구 v4 standalone 대체 — overlay=control 일치 |
| debug viz | **SCC ACTIVE/idle + EGO 속도** overlay 추가 | in-lane lead 인지 + 감속 여부 |
| `side_min_arb` (B) | **미적용 (pending)** | v26 ckpt 테스트 시 True + traj-removed 경로 wiring 필요 |

---

## 7. grad_cam 진단 (이미지 + traj 동시)

`scripts/viz_iy_gradcam_b4.py` — lead-TTC 샘플(inlane_gt dist<20 & closing>1):
- **brake → image** grad-cam: `−ax0` backward → seg backbone conv feat (image→a_2 v7b→i2_mem→side→ax 전체 grad).
- **steer → traj** attribution (신규): `steer0` backward → `|∂steer/∂traj_40|` per-waypoint → BEV 하이라이트.
- B4 관찰: TTC 에서 ax −0.6~−8.3 (감속 함), 단 image attribution 은 diffuse (a_2 흐릿한 128d 표현 + bin기반 target 의 path-localization 미학습과 일관).

---

## 8. 학습 시리즈 history (v25e → v26)

| 시리즈 | 의도 | 결과 |
|---|---|---|
| v25e/f/g | f_2 ACC multi-data + main long freeze + DriverB fix | image_mem 학습 가능 phase 정의 |
| v25h A/B | i_2 cache 사용 (A=self-attn, B=Professor+side) | B steer 정통, 감속 약함 |
| v25i | A + obj_gate (a_2 v6 logits) | — |
| a_2 v7/v7b/v7c | reg loss fix + focal α 튜닝 | v7b production |
| a_2 v8/v9 | NLL+LDS / LDS | 개선 실패 (perception 한계) |
| v25k-B1/B4 | GT inlane + ACC ttc (NO-OP fix) | B4 best 0.5691 (additive baseline) |
| **v26 (B-minACC)** | **lead-gated ACC override + traj 제거 + b_side supervision** | **진행 예정** (v0301 main resume) |

---

## 9. 최종 구조 요약

```
main (f_1):  Professor #654 → v25c(0237/0301/0302) → v0301 단독 finetune
             traj+ego (NO image) → steer, main_ax        [frozen during f_2]

side (f_2):  ego_token (Q) × i_2_mem(KV, a_2 v7b 128d, 4L cross-attn) → b_side
             g = sigmoid(a_2 in-lane obj max + bias)
             ax = main_ax + g·clamp(b_side − main_ax, max=0)   ← downward, lead-gated

i_2 = seg backbone → a_2 (encoder.proj → fusion → fc_proj 256→128) → image_mem + a2_logits(7ch)
```
steer 는 Professor 전담(보존), brake 는 lead 인지 시 f_2 가 보수적 override. main finetune 과 decouple.

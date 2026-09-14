# SCC Controller Imitator — Data & Training Report (v5)

작성일 2026-05-24 (21:20) · branch `main` · 모델 `src/model/image_models/iy_model.py`

REPORT_v4 (2026-05-23 02:30) 이후 약 **42 시간** 분량. v15 ~ v25d 시리즈.

---

## 0. 모델 명명 규약 (이번 보고서부터 통일)

| 기호 | 모듈 | 역할 |
|---|---|---|
| **f_1** | main transformer (Professor RAW base, `main_no_image=True`) | waypoint tracking — traj+ego 만으로 steer/ax 결정 |
| **f_2** | **Side SCC module** (cross-attn Q=traj, KV=image_mem + a_2 v4 feat + TTC label aug) | image-grounded safety overlay — lead car 인지/제동 |
| f_1 + f_2 | residual gated sum (α·side fc) | 최종 (steer, ax) — f_2 zero-init 으로 f_1 backward-compatible |
| i_1 | seg backbone token cache (per-image, 한 번 계산) | LateralStripEncoder |
| i_2 | image fusion block (curr + prev rotated) | spatial fusion |
| a_2 | per-bin (40, 7) features: obj / vL_abs / vY_abs / vL_rel / vY_rel / prox / dist | image → lead 측정 |

이전 보고서에서 "ControlNet side branch" 로 부르던 모듈은 v18-v22 까지의 self-attn 변종. **v23 부터 cross-attn 으로 전환** 후 **f_2 (Side SCC)** 로 통일.

---

## 1. 변경 요약 (v4 → v5)

| 항목 | v4 (5-23 02:30) | v5 (5-24 21:20) |
|---|---|---|
| **모델 분해** | 단일 IyModel (image-encoder concat) | **f_1 (main) + f_2 (Side SCC) 분리 학습** |
| **f_1 base** | spd ckpt | **Professor #654 RAW** (controller_imitator v1, val 0.0355) + finetune |
| **f_2 (side) 형태** | (v18) self-attn 6L parallel | (v23+) **cross-attn 1L** (Q=traj, KV=img_mem, zero-init α gate) |
| **a_2 통합** | 보조 head (학습 신호) 만 | **(v20+) feature 주입** (40×7-ch early inject + late decoder option) |
| **label augment** | brake_aug (정적 floor) | **TTC augment (동적 floor)** — closing²/(2·dist) 기반 안전 감속도 |
| **traj horizon** | full 40 steps (4s) | **traj_zero_after=15** (1.5s 만, 이후 zero-mask) |
| **learning rate** | lr 3e-5 (좁은 범위) | **3e-4 + decaying-max cosine restarts** (5 cycle, ×10 가속) |
| **2-phase 학습** | 단일 phase | **phase1: f_1 frozen / f_2 학습** → phase2: joint with low f_1 lr (1.5e-5) |
| **prev image Δt** | 100ms (DTICK=10) | **200ms (DTICK=20)** — a_2 v4 PREV_DTICK 와 align |
| **ci_aug 데이터** | pkl (5min deserialize) | **npz mmap (12× 축소, 즉시 load)** |
| **scheduler 버그** | — | **N-1 denom 버그 fix** (last cycle max=0 → 학습 정지) |

---

## 2. f_1 / f_2 분리 학습 — 왜?

### 2.1 v18-v22 실패 원인 진단
- v18-v22 의 self-attn side branch 6L (parallel to 6L main) → side 가 main 보다 **더 크고 별도 학습**
- 의도: side 가 image-기반 안전 정보 보강
- 결과: close-lead ax loss plateau 1.00 m/s² (v18 → v22 동일)

User 진단 (5-24 18:00) :
> "main이 controller imitator raw pt잔아. 입출력이 조금 바꼈는데 그에 맞춰서 finetunning안하고 적용하고있었네. main혼자서 waypoint tracking 이 되는지 먼저 검증해야해."

### 2.2 새 전략
1. **f_1 단독 finetune 으로 baseline 확정** (v25b/v25c)
   - main_no_image=True → text-only (traj+ego). controller_imitator 의 input/output spec 으로 맞춤
   - Professor #654 ckpt 에서 직접 resume — image encoder weights 무효, transformer/head 만 학습
2. **f_2 (Side SCC) 추가** (v25d)
   - cross-attn (Q=traj, KV=image_mem) 1L — main 의 traj token 이 image_mem 을 query
   - zero-init α gate → 학습 초기 f_2 효과 0 → f_1 wp tracking 유지
   - α 가 점진 학습되며 safety 보강

---

## 3. f_1 학습 (waypoint tracking baseline)

### 3.1 v25b → v25c 진화
| run | lr_prof | scheduler | best ep | val | 비고 |
|---|---|---|---|---|---|
| v25b | 3e-5 | constant | ep 16 | 0.5507 | lr 너무 작아 수렴 느림 (1h 16 ep) |
| **v25c** | **3e-4** (×10) | **decaying-max cosine 5 cycles** | ep 38 | **0.3706** | -33% val ↓, 1h 30 ep |

steer 4.40° → 3.00°, ax 1.11 → 0.70 m/s² (전 채널 개선).

### 3.2 Decaying-max cosine restarts scheduler (자작)
```python
class _CosineDecayingMaxRestarts:
    def _apply(self):
        c = step // T_0                                  # cycle index
        t = step % T_0
        max_factor   = 0.5*(1 + cos(π·c/n_cycles))       # cycle 마다 max lr 감쇠
        cycle_factor = 0.5*(1 + cos(π·t/T_0))            # cycle 내 cosine
        lr = base_lr * max_factor * cycle_factor
```

cycle factors (n_cycles=5):
```
c=0  max=1.000  → lr_max 3.0e-4
c=1  max=0.905  → lr_max 2.71e-4
c=2  max=0.655  → lr_max 1.96e-4
c=3  max=0.345  → lr_max 1.04e-4
c=4  max=0.095  → lr_max 2.85e-5   ← 마지막 cycle 도 살아있음
```

### 3.3 ⚠️ 버그 fix (v25c 에서 발견)
- **기존**: denom = `max(n_cycles-1, 1)` → c=4 시 cos(π·4/4) = cos(π) = -1 → max_factor = 0 → **lr=0**
- **결과**: v25c ep 41-50 (cycle 4) `lr_prof=0.00e+00` 으로 학습 정지. 14 분 dead loop
- **fix**: denom = `n_cycles` (위 표대로 마지막 0.095 살아있음)
- **확인**: v25c log `[ep 40] lr_prof=1.08e-06 → [ep 41] lr_prof=0.00e+00` 정확 확인

### 3.4 f_1 데이터 구성
| source | total | 사용 | 비율 |
|---|---|---|---|
| SCC sangam_lead_0520 prebake | 72,319 | 1,000 (`--limit`) | 0.14% |
| ci_aug v0237+v0301+v0302 | 729,915 | 729,915 (전체) | 99.86% |
| **합계** | | **730,915** | 100% |

`main_no_image=True` 이므로 SCC 와 ci_aug 의 신호 차이 없음 (둘 다 traj+ego 만). limit 1000 영향 작음 — waypoint tracking 만 학습 목적이므로 적합.

---

## 4. f_2 (Side SCC module) 아키텍처

### 4.1 모듈 구조 (`iy_model.py`)
```python
if side_cross_attn:
    self.side_cross       = nn.MultiheadAttention(d_model, nhead, batch_first=True)
    self.side_cross_ln    = nn.LayerNorm(d_model)
    self.side_fc_cross    = nn.Linear(d_model, 1)                  # ax 만 보정
    self.side_alpha_cross = nn.Parameter(torch.zeros(1))           # zero-init gate

# forward (main_no_image=True 시):
#   tokens_text = [obj_token, traj_embed, state_embed]
#   main_out    = transformer(tokens_text)                          # f_1
#   ax_main     = fc_out(traj_out)[..., 1]
#   side_attn   = side_cross_ln(side_cross(Q=traj_out, K=img_mem, V=img_mem))
#   ax_side     = side_fc_cross(side_attn).squeeze(-1)
#   ax_total    = ax_main + side_alpha_cross * ax_side             # gated residual
```

### 4.2 핵심 설계 선택
- **Q=traj, KV=image_mem** : trajectory 의 각 step 이 image 의 관련 정보 (lead car) 를 query
- **single layer**: f_2 capacity 작게 (f_1 dominate 유지) — params 0.07 M (전체 1.70 M 의 4%)
- **ax-only output**: steer 는 f_1 만, ax (제동) 만 f_2 보정 — 안전 영역만 image 활용
- **zero-init**: α=0 으로 시작 → ckpt-resume 시 f_1 출력 그대로 → 점진 학습 가능

### 4.3 a_2 v4 feature 주입 (use_a2_feat=True)
- 입력: per-frame (40, 7) [obj, vL_abs, vY_abs, vL_rel, vY_rel, prox, dist]
- 정규화: dist × 40, v_*_rel × 20 (scale match for d_model=128)
- early inject: `a2_proj: Linear(7, d_model)` → traj_embed 와 concat 또는 add
- `a2_proj` weight/bias zero-init → f_1 backward-compatible

### 4.4 TTC label augmentation
- in-lane gating: a_2 cache 의 bin 16-25 (중앙 10 bin), obj logit > 0 인 frame 만
- closing rate: dist time-series diff (DTICK=10, 100ms window)
- 조건: `closing > 0.1 m/s AND (dist < 50m OR ttc < 4s)`
- 안전 감속도: `safe_decel = closing² / (2·(dist - 2.0))` (margin 2m)
- 최종 target: `aug_ax = min(driver_ax, -safe_decel)` (보수적 max brake 20 m/s²)

전체 in-lane sample 의 약 22% 가 augment 적용. driver_ax mean -1.8 → aug_ax mean -3.1 m/s² (-1.3 추가 brake).

---

## 5. 학습 phase 분리 (v25d 핵심)

```
phase 1 (ep 1-15)   : f_1 frozen, f_2 학습
  - lr_prof = 1.5e-5 (f_1, but frozen so no effect)
  - lr_other = 3e-4 (f_2: side_cross, side_alpha_cross, a2_proj)
  - lr_side = 3e-4
  - α gate 가 0 → 학습 신호 받아 점진 활성

phase 2 (ep 16-50)  : joint
  - lr_prof = 1.5e-5 (f_1 lowered for fine-tune)
  - lr_other = 3e-4 (f_2 그대로)
  - 두 모듈 동시 학습, 단 f_1 는 매우 느린 lr 로 보호
```

phase 전환 logic (`train_iy.py:_set_phase`):
```python
has_aux = use_side_branch or side_cross_attn or use_a2_feat or use_a2_late
if has_aux and epoch < phase1_epochs:
    # freeze main params
    for p in main_params: p.requires_grad = False
```

⚠️ v23c-v23e 버그: `has_aux` 가 `use_side_branch` 만 체크 → side_cross_attn / use_a2_feat 만 켜면 phase logic 작동 안 함. v23e fix.

---

## 6. 데이터 파이프라인 개선

### 6.1 ci_aug pkl → npz (12× 축소)
- `scripts/convert_ci_aug_pkl_to_npz.py`
- 기존: 7.07 GB pkl, pickle deserialize 5분 소요 (random access 불가)
- 신규: 598 MB npz, mmap 지원, load 즉시
- 추가 효과: 학습 startup 시간 5min → 10s

### 6.2 a_2 cache 생성 (`scripts/build_a2_cache.py`)
- 기존: a_2 학습마다 i_1 backbone re-compute (encoder + fusion + head)
- 신규: i_1 cache 재사용 → fusion + head 만 계산 → **20× 가속**
- ego_motion=0 simplification (학습용 정적 cache, inference 시 ego_5d 실측)
- per-frame (40, 7) tensor pre-compute

### 6.3 prev_dtick 5 → 20 align
- 이전: prebake prev_dtick=5 (50ms), 학습/inference 시 다름
- 신규: prebake 와 a_2 v4 PREV_DTICK=20 일치 (200ms)
- closed_loop `PREV_IMG_DT = 0.20` 으로 align

### 6.4 closing_rate / inlane_dist precompute
- `SCCSynthIyDataset.__init__` 에서 a_2 cache 로 미리 계산
- inlane_dist (bin 16-25 obj 의 min dist × 40) + inlane_closing (prev frame 과 diff)
- 학습 시 collate 에서 즉시 사용 (TTC augment 조건 판정 빨라짐)

---

## 7. closed_loop_node 업데이트

| flag | 값 | 비고 |
|---|---|---|
| MODEL_CKPT_IY | `iy_v25c_best_loss.pt` | f_1 단독 baseline |
| MODEL_USE_OBJ_GATE | False | main_no_image 와 동시 사용 불가 |
| MODEL_USE_SIDE_BRANCH | False | v18-v22 self-attn side. v23+ 는 사용 안 함 |
| MODEL_MAIN_NO_IMAGE | True | f_1 = text-only |
| MODEL_SIDE_CROSS_ATTN | False | v25c (f_1 only), v25d+ 시 True |
| MODEL_USE_A2_FEAT | False | v25c (f_1 only), v25d+ 시 True |
| MODEL_TRAJ_ZERO_AFTER | 15 | traj 1.5s 만 — 학습/inference 일치 |
| PREV_IMG_DT | 0.20 | a_2 v4 PREV_DTICK align |

forward output tuple 처리:
```python
out = self.model(model_in)
pred = out[0] if isinstance(out, tuple) else out
```

v25c smoke test 결과: `load_state_dict missing=0 unexpected=0`, forward shape (1, 40, 2) 정상.

---

## 8. 학습 시리즈 history (v15 → v25d)

| 시리즈 | 의도 | 결과 |
|---|---|---|
| **v15-v17** | wandb + difficulty-aware sampler + lr 5x | sampler 효과 미미 |
| **v18-v22** | ControlNet self-attn side branch (multi variant) | close-lead ax loss plateau 1.0 m/s² |
| **v18c** (v8 resume) | brake_aug + side 2-phase | 정성 best (개선폭 작음) |
| **v19** | side ax-only | 동일 plateau |
| **v20** | a_2 v4 feat early inject | plateau 일부 완화 |
| **v22** | asymmetric brake loss + val close/far split | brake under-prediction 분포 가시화 |
| **v23-v23e** | redesign: main_no_image + side_cross_attn | phase logic 버그 다수 (fix iter) |
| **v24** | TTC augment + v23e resume | 적용 데이터 22% 만 (in-lane 한정) |
| **v25** | Professor RAW + 2-phase | 첫 RAW resume — 입출력 spec mismatch 인지 |
| **v25b** | f_1 alone finetune (lr 3e-5) | val 0.55 — lr 너무 작음 인지 |
| **v25c** | f_1 alone (lr 3e-4 + decaying-max) | **val 0.37 baseline 확정** + scheduler 버그 발견/fix |
| **v25d** | f_1 + f_2 (Side SCC) 가동 | **진행 중** (5-24 21:20~) |

---

## 9. 가동 중 (현재 21:20)

### v25d 학습
- pid 1909954, log `/tmp/v25d.log`
- init: v25c best (missing=11 = f_2 신규 모듈)
- phase 1 ep 1-3: val 0.3707 (init 유지), α gate -0.069 → -0.106 (느린 활성)
- ep time ~40s (v25c 84s 대비 절반 — phase1 main frozen)
- 50 ep 예상 완료: 22:00 경

### Monitor
- task `bwdp59f1w` (Monitor tool) — 매 ep 알림
- best_loss.pt 자동 copy → `data/ckpts/iy_v25d_best_loss.pt`

---

## 10. 다음 단계

### 즉시 (오늘 밤)
- [ ] v25d phase2 (ep 16+) 진입 시 val 변화 관찰 — joint lr 1.5e-5 로 f_1 살짝 폴리시
- [ ] v25d 완료 후 closed_loop test (이미지 grounding 확인 — lead car 시나리오)

### 단기 (1-2일)
- [ ] v25d 결과 양호하면 multi-prebake 확장 (sangam_lead_0521 / pertO / pertX 추가)
- [ ] f_2 ablation: a_2 feat 만 / cross-attn 만 / TTC augment 만
- [ ] 신규 raw bank (#31) — 4 policy 가동 (v25d 끝나고 CPU 자원 풀려서)

### 장기 (이번 주)
- [ ] backbone v2_s → v2_l swap (#13)
- [ ] DriverB_new traj order 수정 (#15)

---

## 11. 시계열 commit 요약 (v4 → v5, 약 40 시간)

```
d2326ec feat(inference+prebake): v25c ckpt + main_alone config + prev_dtick 20 align
d25c9a9 feat(train): main_no_image + side cross-attn + a_2 feat + TTC augment + scheduler fix
[다음] feat(scripts): a_2 cache builder + ci_aug npz convert + v25d launcher
[다음] docs(report): v5 - f_1 / f_2 (Side SCC) 분리 학습 + scheduler 버그 + 명명 규약
... (v15-v25 시리즈 commits 다수)
```

상세는 `git log --oneline 473aa28..HEAD` 참조.

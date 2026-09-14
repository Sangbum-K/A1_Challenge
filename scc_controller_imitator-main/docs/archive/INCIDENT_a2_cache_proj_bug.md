# INCIDENT REPORT — a_2 cache `proj` 누락 버그 (v18-v25 시리즈 모두 영향)

**발견 일시**: 2026-05-24 22:30  
**Severity**: CRITICAL (시리즈 6주 학습 결과 모두 영향)  
**Status**: ROOT CAUSE 확인 ✓ / FIX 검증 ✓ / 6 cache 재생성 진행 중  

---

## TL;DR

`build_a2_cache.py` 가 a_2 model 의 `encoder.proj` (Conv1d C_b→d_token) 통과 step 을 누락. 학습 시 model 이 보는 input 과 cache 가 만든 input 의 분포가 다름 → a_2 의 fusion+head 가 random 분포 input 을 처리 → output 도 random.

| metric | 버그 이전 | **fix 후** | 학습 시 reported |
|---|---|---|---|
| ROC AUC (obj detect) | 0.509 | **0.982** | — |
| Average Precision | 0.099 | **0.879** | — |
| dist MAE (m) | 575 | TBD (재측정) | 1.03 |
| pred obj logit range | -19 ~ +4 (mean -0.4) | **-22 ~ +8** (mean -13, full dynamic) | — |
| TTC augment 발동률 (thr=0) | **0.02%** | **76% × base rate** | — |

**영향 받은 학습 시리즈**: v18-v25 (실제 image-grounded 모듈 사용한 모든 실험). 즉 ControlNet side branch, side cross-attn, a_2 feat early/late inject — 모두 noise input 으로 학습됐음.

**살아남은 학습**: v25c (main_no_image=True, f_1 baseline) — a_2 feature 자체를 사용 안 했으므로 정상 학습. val 0.3706, 향후 base init 로 활용 가능.

---

## 1. 발견 경위 (timeline)

**2026-05-24 21:32 ~ 22:00**: v25e launch 후 val=1.49, ax 10 m/s² 의 비정상 결과. data 분포 점검:
- DriverB prebake target 모두 0 발견 (별도 bug — 동일 일 발견)
- pertO/X 가 sample 76% 점유, ACC 아님

**22:10**: pertO/X limit 적용 + DriverB prebake 재생성으로 v25f 재구성 시작.

**22:14**: a_2 cache 점검 시 — `in-lane obj > 0 비율 0.3% 만` 발견. obj logit 분포 정상 (mean -0.4, std 2.07) 하지만 threshold 0 이 너무 strict 한 듯 보임.

**22:17**: 추가 점검 — DriverB, spd_v2 의 a_2 cache 가 all zeros (std=0). 다른 cache 도 의심.

**22:23**: aux_targets (GT) vs a_2 cache prediction 비교:
- GT obj in-lane 98.6% / pred 0.3% → recall ~0.3%
- **ROC AUC = 0.509** (random) ⚠️
- dist MAE 575m (학습 시 reported 1.03m, 100× 차이)

**22:25**: ckpt state_dict 점검 — missing=0, unexpected=0. ckpt 자체는 정상.

**22:27**: encoder forward 비교 — `LateralStripEncoder.forward` 는 `backbone → adaptive_avg_pool → self.proj` 3 step. cache 빌더는 backbone+pool 까지 ✓, **proj 누락** ❌.

**22:29**: fix 적용 (1 줄 추가) + 검증 → **ROC AUC 0.982** ✓.

---

## 2. Root Cause 상세

### 2.1 i_1 cache 의 구조
`scripts/compute_i1_tok_cache.py:131-163`:
```python
# 주석 명시: "proj 은 model-side 학습 가능 → backbone 직후 (C_b, 1, 40) 까지만 cache.
#            그러면 학습 시 model 의 proj (Conv1d C_b → D) 를 거치도록."
feat = backbone(x)[-1]                                # (B, C_b=256, ~8, ~40)
feat = adaptive_avg_pool2d(feat, (1, 40))             # (B, C_b=256, 1, 40)
# proj 적용 안 함 — raw backbone token 저장
```
→ i_1 cache 의 token shape = `(N, 40, C_b=256)`, **proj 적용 전** raw backbone feature.

### 2.2 A2Model 의 forward
`src/model/image_models/a2_model.py:124-128`:
```python
def forward(self, image_curr, image_prev_rot, ego_motion):
    tok_c = self.encoder(image_curr)         # encoder 내부에 proj 포함
    tok_p = self.encoder(image_prev_rot)
    fi_mem = self.fusion(tok_c, tok_p, ego_motion)
    logits = self.head(fi_mem)
```

`LateralStripEncoder.forward`:
```python
def forward(self, x):
    feat = self.backbone(x)[-1]
    feat = F.adaptive_avg_pool2d(feat, (1, 40)).squeeze(2)
    feat = self.proj(feat)                              # ← KEY: Conv1d C_b → d_token=256
    return feat.transpose(1, 2)
```

→ 학습 시 fusion 의 input = `proj` 통과 후 token. Conv1d weight 가 학습됨 → backbone feature 분포 → d_token 분포로 변환.

### 2.3 build_a2_cache.py 의 버그 (수정 전)
```python
# tc = i_1 cache token (proj 전 raw backbone)
tp_rot = yaw_shift_tokens_batch(tp, dy_t, fx=fx_t)
ego_mot = torch.zeros(tc.shape[0], 2, device=args.device)
fi_mem = a2.fusion(tc, tp_rot, ego_mot)  # ← BUG: proj 누락! raw backbone token 직접 inject
out = a2.head(fi_mem)
```

→ fusion 이 학습 시 본 적 없는 (raw backbone) 분포의 input 을 받음. fusion 의 layer norm / cross-attn weight 가 d_token 분포에 학습됐는데, raw backbone (다른 statistics) 가 들어오니 output 완전 무의미.

### 2.4 FIX (1 줄 추가)
```python
# i_1 cache 는 backbone+pool 후, proj 전 token. a_2 encoder.proj 통과시켜야 학습 시 input 분포와 일치
tc_proj = a2.encoder.proj(tc.transpose(1, 2)).transpose(1, 2)      # (B, 40, d_token)
tp_proj = a2.encoder.proj(tp_rot.transpose(1, 2)).transpose(1, 2)
fi_mem = a2.fusion(tc_proj, tp_proj, ego_mot)                       # ← 학습 시와 정합
```

---

## 3. 영향 분석

### 3.1 학습 시리즈별 영향
| 시리즈 | a_2 사용 여부 | 영향 |
|---|---|---|
| v6-v17 | obj_gate 학습 (a_2 cache logit 사용) | obj_gate 가 noise 학습. 실제 효과 거의 없음 |
| v18-v22 | side_branch + obj_gate | side 의 image grounding 부분 모두 noise. plateau 1.0 m/s² 의 원인 |
| v20 | use_a2_feat (early inject) | a_2 raw value (40, 7) 모두 noise → model 이 ignore 학습 |
| v23-v23e | side_cross_attn + use_a2_feat | image_mem (i_1 cache) 자체는 정상, but a_2 feat noise + cache 도 noise |
| v25 | TTC augment + a_2 cache | TTC 가 0.02% 만 발동 → label noise |
| v25b/v25c | main_no_image=True, **a_2 미사용** | ✓ 정상 학습 (f_1 baseline val 0.37) |
| v25d/v25e | use_a2_feat + side_cross_attn | 부분적으로 image_mem 은 valid (cache 정상), a_2 feat 만 noise |

→ **v25c 만 진짜 학습**. 나머지는 image-grounded 의도 vs 실제 가 다름.

### 3.2 메모리 (앞서 기록된 metric) 의 진실
- a_2 v4 학습 시: Precision 70→98%, dist MAE 5.09→1.03m, vL_rel 5.53→0.94 m/s
  → ckpt 자체 학습 metric. **valid**. cache 빌더만 잘못.

- 이전 보고서 (REPORT_v3/v4) 의 "obj_gate 효과" / "side branch a_2 활용" 등 image-grounded 효과 묘사 모두 노이즈 input 기반 → 실제 효과 미상.

---

## 4. 검증 (fix 전/후)

### 4.1 sangam_lead_0520 prev20 — 728 unique frames (prebake hit)

| metric | 전 | 후 |
|---|---|---|
| pred obj logit mean | -0.38 | **-13.42** |
| pred obj logit std | 2.07 | **7.41** |
| pred obj logit range | [-19, 4] | **[-22, 8]** |
| **ROC AUC** | **0.509** | **0.982** |
| AP (avg precision) | 0.099 | **0.879** |
| thr=0 precision | 0.355 | **0.791** |
| thr=0 recall | 0.002 | **0.763** |
| thr=0 F1 | 0.004 | **0.777** |
| thr=-2 recall (강조 시) | 0.866 (base rate) | **0.863** (prec 0.672) |
| dist correlation | — | 0.314 (계산 unit 추가 점검 필요) |

### 4.2 GT base rate (sangam_lead_0520)
- GT obj=1: 9.3% (per-bin per-frame, 5000 frames × 40 bins = 200k entries)
- aux GT in-lane (bin 16-25): 98.6% frames 에 lead 존재

---

## 5. Fix 적용 후 후속 작업

### 5.1 6 cache 일괄 재생성 (진행 중, 2026-05-24 22:30~)
```bash
bash scripts/rebuild_a2_caches_then_v25f.sh
```
- sangam_lead_0520_prev20 ✓ (검증용 첫 빌드)
- sangam_lead_0521_prev20
- sangam_pertO_v2_prev20
- sangam_pertX_v2_prev20
- DriverB_new_prev20 (DriverB fixed prebake + 새 cache)
- spd_v2_prev20

→ 결과 file: `a2_out_cache_{name}_prev20_fixed.npz`

### 5.2 v25f auto launch (chain)
- launch_v25f_fixed_a2.sh 자동 생성 (`_fixed.npz` cache 사용)
- DriverB prebake done + 6 cache 완료 → 자동 launch

### 5.3 향후 결과 비교
- v25e (broken cache): val 1.49 (ax 10 m/s²)
- v25f (fixed cache + proper data): 예상 val < 0.5

---

## 6. 재발 방지 (action items)

### 6.1 build_a2_cache.py 에 self-validation 추가 (TODO)
- cache 생성 직후 aux_targets 중 일부 sample 로 GT vs pred AUC 측정
- AUC < 0.7 면 ERROR 출력 + fail-fast

### 6.2 i_1 cache 의 명확 documentation
`compute_i1_tok_cache.py` 상단 주석:
> ⚠️ i_1 cache 의 token 은 **proj 적용 전 raw backbone output** (C_b channels).
> 학습/inference 시 model 의 `proj` Conv1d 를 통과시켜야 d_token 분포로 변환됨.
> a_2 cache 빌더, i_y model 의 self.img_proj 등에서 이 단계 누락 시 output 무의미.

### 6.3 ckpt-cache 정합 test
- pytest 또는 smoke test: `tests/test_a2_cache_quality.py`
- 학습 시 reported metric (precision 98%) 과 cache inference metric 비교
- 차이 > 10% 면 정합 의심

### 6.4 i_y model 의 image 처리도 점검 필요
- `iy_model.py` 의 `self.img_proj = nn.Conv1d(C_b, d_model, kernel_size=1)` 존재 ✓
- forward 시 i_1 cache → `img_proj` 통과 → fusion → transformer. 
- → i_y 는 자체 proj 보유, 정상. v25c (f_1) 가 살아남은 이유 확인.

### 6.5 보고서 / 문서 업데이트
- REPORT_v5 에 본 incident 요약 + 영향 받은 시리즈 표기
- memory: `a2_cache_proj_bug.md` 저장 (이런 류 cache mismatch 항상 의심)

---

## 7. 교훈

1. **cache 의 raw 분포는 항상 의심**. ckpt 학습 metric 과 cache inference metric 을 정량 비교 — 침묵하는 mismatch 가 가장 위험.
2. "missing=0, unexpected=0" 는 weight load 성공만 보장. **forward graph 가 학습 시와 동일한지** 별도 검증 필요.
3. ROC AUC 0.5 (random) 같은 명확한 signal 을 빨리 측정. 우리는 6주 동안 cache 의 obj > 0 비율 (0.02%) 만 보고 "데이터에 lead 거의 없음" 으로 잘못 해석.
4. **encoder.proj 같은 작은 layer 도 학습 시 weight 가 분포 변환에 critical**. 단순 cache step skip 의 영향이 매우 큼.
5. fix 가 1 줄. 발견이 어려운 것 — 항상 self-test 안전망 필요.

---

## 8. 관련 commit / file
- `scripts/build_a2_cache.py` (fix 적용 line 95-105)
- `scripts/compute_i1_tok_cache.py:131-163` (i_1 cache 가 proj 전 raw 인 이유 주석)
- `src/model/image_models/a2_model.py:19-45` (LateralStripEncoder 의 proj 위치)
- `scripts/rebuild_a2_caches_then_v25f.sh` (재생성 chain)
- `docs/REPORT_v5.md` (본 incident 추후 추가 예정)
- memory `a2_cache_proj_bug` (재발 방지용)

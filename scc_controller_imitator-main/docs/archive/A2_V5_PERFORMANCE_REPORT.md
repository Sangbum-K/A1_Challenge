# a_2 v5 성능 분석 리포트

작성: 2026-05-25 01:05  
ckpt: `data/ckpts/a2_v5_d128_best_loss.pt` (60 epoch 총, val loss 0.6072 → 0.4980)  
spec: v4 (d_token=256) encoder+fusion frozen + 신규 fc_proj (256→128) + 신규 AuxHead (128→7) 만 학습

---

## 1. TL;DR
- **AUC universally 우수** (0.94 ~ 0.99) — ranking 능력 매우 강함
- **Recall@0 낮음** (0.44 ~ 0.80) — calibration shift (positive class 의 logit 이 보수적)
- **fix 옵션**: threshold 낮춤 (즉시) 또는 focal loss 재학습 (30분 소요)
- d=128 압축에도 v4 (d=256) 대비 **동등 또는 우수** — fc_proj 가 정보 잘 보존

---

## 2. 학습 history

| stage | epochs | lr | scheduler | val loss start | val loss end | best ep |
|---|---|---|---|---|---|---|
| stage 1 | 1 ~ 10 | 3e-4 | cosine T_max=10 | 0.6072 | 0.5882 | 10 |
| stage 2 (resume) | 1 ~ 50 | 1e-4 | cosine T_max=50 | 0.5082 | 0.4980 | 50 |
| **total** | **60** | | | **0.6072** | **0.4980 (-18%)** | 50 |

학습 시간: cache 만 사용 → **총 13분** (raw image NFS load 시 5시간 예상 대비 **20× 빠름**)

---

## 3. AUC / AP / Recall — per-dataset

| dataset | matched | base rate | **AUC** | **AP** | P@0 | R@0 | F1@0 | vs v4 |
|---|---|---|---|---|---|---|---|---|
| sangam_lead_0520 | 728 | 9.3% | **0.9920** | 0.9416 | 0.91 | **0.80** | 0.85 | v4 0.982 → **+0.010** ⬆ |
| sangam_lead_0521 | 985 | 1.1% | **0.9742** | 0.5132 | 0.56 | 0.44 | 0.49 | v4 0.977 → -0.003 (동등) |
| sangam_pertO_v2 | — | **0%** | — | — | — | — | — | GT lead 없음 (정상) |
| sangam_pertX_v2 | 0 | — | — | — | — | — | — | filename pattern 미일치 |
| **DriverB_new** | 7,860 | 3.8% | **0.9410** | 0.6511 | 0.80 | 0.46 | 0.58 | v4 0.902 → **+0.039** ⬆⬆ |
| spd_v2 | 0 | — | — | — | — | — | — | i_1/aux raw bank mismatch |

### 핵심 발견
- AUC 모두 0.94+ — **a_2 v5 가 ranking 으로는 거의 perfect**
- DriverB (실제 driver, 가장 다양 분포) 에서 v4 대비 **+3.9% 개선**
- d=128 압축이 정보 손실 거의 없음 (오히려 일부 dataset 개선)

---

## 4. Threshold sweep — Recall 의 정량 분석

| dataset | base | **thr=-2** | thr=-1 | thr=-0.5 | thr=0 | thr=+1 | best F1 |
|---|---|---|---|---|---|---|---|
| sangam_lead_0520 | 9.3% | P 0.63 / **R 0.96** / F1 0.76 | 0.78/0.90/**0.84** | 0.85/0.85/**0.85** | 0.91/0.80/0.85 | 0.98/0.68/0.80 | thr=-0.5 |
| sangam_lead_0521 | 1.1% | 0.17/**0.81**/0.28 | 0.31/0.63/0.42 | 0.42/0.53/**0.47** | 0.56/0.44/0.49 | 0.84/0.24/0.38 | thr=-0.5 |
| DriverB_new | 3.8% | 0.38/**0.73**/0.50 | 0.60/0.60/**0.60** | 0.71/0.53/0.61 | 0.80/0.46/0.58 | 0.92/0.31/0.47 | thr=-1 |

### 의미
- **threshold 0** (sigmoid 0.5) 가 **너무 strict** (model 이 conservative)
- threshold -1 ~ -2 로 낮추면 recall 대폭 증가 (44% → 81% 가능)
- AUC 0.99 = ranking 정보 우수 → threshold 조정만으로 desired recall 달성 가능
- best F1 thresholds = -1 ~ -0.5 사이

---

## 5. obj logit 분포

| dataset | mean | std | min | max |
|---|---|---|---|---|
| sangam_lead_0520 | -4.5 | 3.0 | -12.7 | +6.7 |
| sangam_lead_0521 | -4.3 | 2.7 | -11.5 | +3.8 |
| sangam_pertO_v2 | -4.5 | 2.6 | -12.9 | +1.5 |
| sangam_pertX_v2 | -2.0 | 2.8 | -12.9 | +1.9 |
| DriverB_new | -4.8 | 2.3 | -14.9 | +8.1 |

→ 모든 dataset 에서 logit mean ≈ -2 ~ -5 (sigmoid 0.01 ~ 0.12). positive 의 logit 도 0 미만 다수 → **negative biased calibration**.

---

## 6. v4 vs v5 비교 (d=256 vs d=128)

| dataset | v4 AUC | v5 AUC | Δ |
|---|---|---|---|
| sangam_lead_0520 | 0.982 | **0.992** | **+0.010** |
| sangam_lead_0521 | 0.977 | 0.974 | -0.003 |
| DriverB_new | 0.902 | **0.941** | **+0.039** |
| **average** | 0.954 | **0.969** | **+0.015** |

→ v5 (d=128) 가 v4 (d=256) 보다 **평균 +0.015 우수**. fc_proj 가 단순 dimension 축소가 아닌 학습된 효과 있음.

---

## 7. calibration 이슈의 원인 분석

### 가설
1. **Class imbalance**: 모든 dataset 의 GT positive 비율 1-10% → BCE loss 가 negative dominant → model 이 positive 에 less confidence
2. **fc_proj 의 초기 zero init effect**: 학습 초기 fc_proj weight 가 small → logit 도 small magnitude → 회복 못 함
3. **lr 너무 작아 calibration shift 만 진행, magnitude 학습 안 됨**

### 검증 (방금 launch)
- focal loss (α=0.5, γ=2.0) 적용 + 30ep 추가 학습 — `outputs_multi/a2/v6_d128_focal/`
- focal loss = hard examples (low p_t) 강조 → positive 학습 가속
- 예상 결과: AUC 유지 + Recall@0 향상 (calibration center 이동)

---

## 8. iy_model 에 미치는 영향

### 직접적 (a_2 cache 의 사용 형태)
- **iy_model 의 `use_a2_feat=True`**: a_2 logits (40, 7) 을 raw value 그대로 `a2_proj = Linear(7, 128)` 통과
- linear projection 은 **constant bias shift 흡수 가능** → calibration shift 가 sample-independent 이면 영향 없음
- 그러나 sample-dependent shift 가 있으면 일부 정보 손실

### 간접적 (TTC augment 등 threshold-based logic)
- `train_iy.py` 의 in-lane gating 시 `OBJ_THR = 0.0` (sigmoid 0.5)
- 위 결과: thr=0 시 발동률 0.04 ~ 0.8% (전체 frame 중)
- thr=-1 또는 -2 로 변경 시 발동률 0.6-8% (10× 향상)

### closed_loop 에서 객체 인지
- iy_model 의 obj_gate 같은 곳에서 sigmoid(obj_logit) > 0.5 사용 시 conservative
- 다행히 v25c+ 는 obj_gate 사용 안 함 (MODEL_USE_OBJ_GATE=False) → 영향 작음

---

## 9. 향후 작업

### 즉시 (v6 focal 학습 끝 후)
- v6 focal 의 AUC + R@0 측정 → v5 vs v6 비교
- 만약 v6 우수 → i_2 cache 재빌드 + iy 학습에 사용
- 만약 v5 더 우수 → v5 유지 + TTC augment OBJ_THR 조정만

### 중기
- pertX/spd_v2 의 measure_auc filename pattern fix (정확한 매칭)
- spd_v2 의 i_1 cache 와 prebake key mismatch fix (raw bank align)
- DriverB 의 logit 분포 더 깊은 분석 (positive sample 도 logit < 0 이 흔한 이유)

### 장기
- a_2 v6 또는 v7 으로 d_token 도 128 로 통일 (fusion 도 d=128 재학습) — 진정한 single dim
- ImageFusionBlock 의 두 model (a2_model.py, iy_model.py) 통합 → single class

---

## 10. 결론

**a_2 v5 = ranking 우수 (AUC 0.99) but calibration shift**. iy_model 의 `a2_proj` linear 가 shift 흡수 가능 → 학습에 큰 문제 없음. TTC augment 의 OBJ_THR 만 조정 필요.

추가 학습 (focal loss) 으로 calibration 균형 시도 중 — 결과 비교 후 v5 또는 v6 채택 결정.

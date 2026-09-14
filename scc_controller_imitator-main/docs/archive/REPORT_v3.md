# SCC Controller Imitator — Data & Training Report (v3)

작성일 2026-05-22 · branch `main` · 모델 `src/model/image_models/iy_model.py`

REPORT_v2 (2026-05-21) 이후 추가/변경 내역.

---

## 1. 변경 요약 (v2 → v3)

| 항목 | v2 (5-21) | v3 (5-22) |
|---|---|---|
| **MPC GT 합성** | acados (legacy) — ax ±170 spike, jerk 1000 | **HPIPM mpc/ fork** — state ax cap ±20, jerk ±100, `--warmup 20` |
| **합성 dataset** | scc_synth_v2_* (이전 cap 무 ckpt) | **전체 재합성 중 (10.13M sample 목표, 진행 7.8%)** |
| **f_1 학습 모드** | 1회 prebake → fixed-set train | **streaming reload** — 합성 진행 중 매 epoch 데이터 누적 |
| **i_g (image gate)** | 코드만 존재, 미학습 | **`use_obj_gate=True` + obj_loss_w=0.5 학습 시작** |
| **close-lead** | meta 만 존재, 미적용 | **WeightedRandomSampler ×3 oversample + loss_mult ×2** |
| **검증 도구** | viz_a2_inference, fi_mem_qualitative | **viz_iy_gradcam, viz_iy_fimem_norm, viz_arch_diagram** |
| **closed_loop** | V2 model only | **IyModel inference path (MODEL_TYPE 분기)** |

---

## 2. 합성 재구동 (mpc/ HPIPM fork)

### 2.1 동기
- acados `scc_control` 의 state `ax` constraint 가 commented out → ill-posed input 시 ax ±170 m/s² 출력
- legacy `data/scc_synth_*` 는 그 결과 → MPC GT 가 학습 시 outlier 로 작용

### 2.2 mpc/ fork 변경
```
mpc/src/app/control/longitudinal_control/.../longitudinal_delayed_kinematic_lmpc.cpp:
  constexpr double kMaxAxAbs = 20.0;       // 차량 brake 한계
  constexpr double kMaxJerkAbs = 100.0;
  qp[i].idxbx = {1};                       // state index 1 = ax
  qp[i].lbx = -kMaxAxAbs; qp[i].ubx = kMaxAxAbs;
  for input idxbu = {0} (jerk): clip ±100

scc_batch_synth/src/synth_batch.cpp:
  RunOneSample 에 lat_ctrl.Reset() + long_ctrl.Reset() + N 회 warmup-solve
  → 매 sample 의 첫 step 이 cold-start spike 없이 안정 상태 출력
  CLI: --warmup N (default 5, 운영 20)

mpc/src/lib/spline/spline.cpp:
  assert(...) → throw std::runtime_error(...)
  synth_batch RunOneSample try/catch 가 catchable → bad sample skip
```

### 2.3 warmup 수렴 비교 (12 frame 동일 sample)
- viewer: `outputs_multi/viz_resynth/warmup_cmp/` (w0 / w5 / w10 / w20 4-way)
- w0 (cold): ax curve 첫 step ±20 saturation
- w5: 대부분 안정. 일부 sample 1-2 step 잔여 spike
- w10: w5 와 거의 동일
- w20: 일관적으로 부드러움. **최종 채택**

### 2.4 가동 plan
- 24 shard × 3 track 병렬 (CPU 24 core fully 점유, load ~30)
- 시작 2026-05-22 10:55, ETA ~58시간 (warmup 20 은 w5 의 ~3.5× 느림)
- 합성 dir: `data/scc_synth_v2_{spd, sangam_pertO, sangam_pertX}`

---

## 3. Streaming 학습 인프라

### 3.1 동기
- 합성 완료 (~58시간) 까지 GPU idle 손실
- 데이터 점진 누적 → 모델이 "보던 분포" 가 매번 확장 → online distribution shift 학습 효과

### 3.2 3 component
```
streaming_prebake_loop.sh (bg, 10분 cycle):
  1) build_pair_manifest --filter-existing  →  현재 존재하는 .scc.txt 만 manifest
  2) prebake_iy_dataset --aux-policy-dir POLICY:AUX_DIR  →  npz overwrite (atomic)

prebake_iy_dataset.py:
  np.savez_compressed(out + ".partial.npz", ...) ; os.replace → atomic
  ← streaming train 의 reload-on-change race 방지 (BadZipFile crash 회피)

train_iy.py:
  --reload-on-change → 매 epoch 시작 시 prebake mtime 체크
  변경 시 Dataset/DataLoader/sampler 재build (warm-start 유지)
```

### 3.3 실측 (spd 단독, 90분 누적)
- 데이터: **187K → 256K samples** (7 reload, 매 10분 ~10K 증가)
- ep_time: 100s ~ 130s (데이터 증가 + closed_loop GPU 경쟁)

---

## 4. spd aux_targets 생성 (i_g 학습 준비)

### 4.1 동기
v2 까지 spd 에 대한 a_2 GT 미생성 → prebake `obj_target` 모두 0 → obj_gate 학습 불가

### 4.2 실행
```bash
python scripts/prepare_a2_aux_targets.py \
  --fx 640 --out-tag spd_fx640 \
  --meta speedway_agent1_meta.json \
  --raw-base /home/ailab-12/AILabSSD/.../00_raw \
  --out-base data/aux_targets
```
- 37,040 frame, 35분 소요 (단일 worker, CPU 합성과 약간 경쟁)
- 출력: `data/aux_targets_spd_fx640/{frame_id}.npz`, target (40, 7)
  - ch 0: obj 0/1
  - ch 1-2: v_long_abs, v_lat_abs (ego basis)
  - ch 3-4: v_long_rel, v_lat_rel
  - ch 5: prox = exp(-d/20)
  - ch 6: dist (m)

### 4.3 spd obj 분포 통계
- 226K prebake sample 의 **100% 에 1+ bin 객체**
- bin 16-23 (화면 중앙 1/4) 에 hits 집중, bin 28-39 모두 0 (트랙이 우측으로 안 휘어서)
- frame 당 평균 4.24 bin 점유

### 4.4 close_lead 와 obj_target 의 차이
- obj_target: 옆 차/뒤 차 포함 모든 viewable 차량 bbox projection → 거의 모든 frame 에 1+
- close_lead_meta: ego forward raceline 가장 가까운 차의 dist → 63% frame 에 "lead 없음"

---

## 5. f_1 v3 학습 (`spd_streaming_objgate_close`)

### 5.1 명령
```bash
python -u scripts/train_iy.py \
  --prebake data/prebake/f1_prebake_iy_spd_streaming.npz \
  --i1-cache data/cache/i1_tok_cache_spd_v2.npz \
  --reload-on-change \
  --resume-from outputs_multi/iy_spd_streaming_w20/best_loss.pt \
  --use-obj-gate --obj-loss-w 0.5 \
  --close-lead-meta data/cache/close_lead_meta_v3.npz \
  --close-weight 3.0 --close-loss-mult 2.0 --close-thr 30 \
  --fx 640 --epochs 200 --batch-size 64 --workers 4 \
  --name spd_streaming_objgate_close
```

### 5.2 loss 구성
```
sw = where(min_dist < 30, 2.0, 1.0)                        # per-sample
L1 steer  = mean( sw · |pred − target| / 10 )
L1 ax     = mean( sw · |pred − target| / 10 )
obj_bce   = BCEWithLogits(obj_logit, obj_target)
total     = L1_steer + L1_ax + 0.5 · obj_bce
sampler   = WeightedRandomSampler(weight=3.0 if close else 1.0)
```

### 5.3 학습 진행 (5-22 14:18 시작, 1.5 시간 누적)

| ep | val | steer | ax | close ax | far ax | data |
|---|---|---|---|---|---|---|
| 1 | 0.242 | 0.35° | 2.07 | 1.93 | 3.18 | 187K |
| 5 | 0.205 | 0.32° | 1.74 | 1.72 | 1.89 | 187K |
| 12 | 0.188 | 0.27° | 1.61 | 1.60 | 1.74 | 197K (+reload) |
| 24 | 0.176 | 0.28° | 1.48 | — | — | 216K |
| 34 | 0.170 | 0.31° | 1.39 | — | — | 236K |
| 45 ✨ | **0.166** | 0.31° | 1.35 | 1.32 | 1.68 | 256K (best) |

- resume 시점 best 0.151 (no gate) 보다 약간 높음 — obj_gate random init + BCE 추가 + close oversample 효과
- close vs far 격차 좁아짐 (ep1: far 1.5× close → ep45: 1.27× close)

### 5.4 i_g (obj_gate) 효과 검증

**viz_iy_fimem_norm.py** (16 sample: 8 close + 8 lead 없음)

| | mean(|fi_mem|) gate 전 | mean(|fi_mem|) gate 후 | mean(obj_prob) |
|---|---|---|---|
| WITH close-lead | 15.32 ± 0.84 | 1.25 | 0.089 |
| WITHOUT lead | 15.37 ± 0.86 | 1.21 | 0.086 |

- ✅ gate 코드 작동 (norm 후가 1/12 수준)
- ❌ 객체 구분은 아직 학습 못 함 (obj_prob WITH/WITHOUT 거의 동일)
- 원인: 24 ep 학습 부족 + obj_target 모든 frame positive → binary supervision 정보 부족

**viz_iy_gradcam.py** (close-lead 12 sample, ax-target attribution)
- 빨간 lead car 위치에 heatmap 집중 ✅
- pred ax 와 GT ax 일관성 양호
- 단, 광고판/벽 같은 high-texture 영역에 spurious activation 잔존

---

## 6. closed_loop_node IyModel 호환

### 6.1 변경
```python
MODEL_TYPE         = "iy"            # "v2" legacy / "iy" v3
MODEL_USE_OBJ_GATE = True
MODEL_CKPT_IY      = outputs_multi/iy_spd_streaming_objgate_close/best_loss.pt

_build_model() iy 분기:
  timm.create_model(tf_efficientnetv2_s, features_only=True)
  seg backbone preload (774 keys)
  IyModel(use_obj_gate=True, fx=640)

_inference():
  fc = backbone(ic)[-1]
  tok_c = F.adaptive_avg_pool2d(fc, (1, 40)).squeeze(2).transpose(1, 2)
  # 동일 fp, d_yaw_t, fx_t
  pred, obj_logit = model({tok_curr, tok_prev_raw, d_yaw, ego_5d, traj_40, fx})
```

### 6.2 sanity test
- seg backbone 774 keys preload ✓
- IyModel ckpt missing=0 unexpected=0 ✓
- Token shape (1, 40, 256), forward (1, 40, 2) 정상 ✓

### 6.3 실측 inference latency (학습+합성 경쟁 환경)
- fwd 120-360 ms (목표 ≤50ms 의 ~6× 느림)
- 원인: GPU 학습과 cuda kernel 경쟁
- 학습 일시 정지 시 ~20ms 복귀 예상 (별도 측정 미실시)

---

## 7. 다이어그램

### 7.1 모델 구조 (paper-style)
- `outputs_multi/viz_resynth/arch/iy_architecture.png` / `.svg`
- viz_arch_diagram.py 로 자동 생성
- input row / image branch (i_1, i_y, img_proj, i_2 stages, i_g) / text branch / concat / f_1 transformer / fc_out / outputs / supervision arrows

### 7.2 viewer port
- http://localhost:8010/ — 통합 dashboard
  - `warmup_cmp/` w0/w5/w10/w20 비교
  - `gradcam_spd/` ax-target GradCAM
  - `fimem_spd/` no-gate norm 비교
  - `fimem_objgate/` gate 학습 후 norm + obj_prob
  - `arch/` 모델 구조 다이어그램

---

## 8. 시계열 + 주요 commit (5-22)

```
21247b5 chore: .gitignore — mpc/ symlink + build 산출물 제외
e216251 feat(mpc): synth_batch --warmup N CLI + WARMUP env
65ad7f9 feat(train_iy): streaming reload + atomic prebake + loop script
60f6445 feat(closed_loop_node): IyModel inference path + __init__ alias
ad0377c docs(DESIGN): T=30→T=40 + loss scale + obj_loss_w 0.5
84f...   feat(viz): IyModel 검증 도구 4종 추가
```

---

## 9. 다음 단계 / 미해결

| | 항목 | 우선도 |
|---|---|---|
| ☐ | 합성 완료 대기 (~50시간 더, sangam 2 track 도 시작 필요 검토) | High |
| ☐ | obj_gate 학습 더 진행 (ep 60~100) 후 fimem norm 재측정 — 객체 구분 학습 검증 | High |
| ☐ | obj_target 재정의 (frame-level lead_present binary, close_lead_meta dist<30 → 1) — binary supervision 명확화 | Med |
| ☐ | close-lead loss_mult / weight 튜닝 (현재 3.0/2.0 → 5.0/3.0 시도) | Med |
| ☐ | sangam track streaming 학습 (현재 spd 만) | Med |
| ☐ | closed_loop 실차/bag 검증 (학습 일시 정지 후 latency 측정) | Med |
| ☐ | backbone v2_s → v2_l 또는 b0/b1 경량화 ablation | Low |

---

## 부록: 가동 중 process 요약 (5-22 15:48 기준)

- **합성**: 24 synth_batch shard (8 per track × 3) — 7.8% / ETA 50h
- **streaming prebake loop**: 1 bg (10min cycle, --aux-policy-dir 적용)
- **train_iy**: 1 main + 4 dataloader workers — ep 48 진행 중
- **closed_loop_node**: 1 가동 (사용자 직접 launched)
- **viewer http server**: port 8010 (outputs_multi/viz_resynth)
- **load**: 30~37 (24 core, 1.5× oversub)
- **GPU**: 930 MiB / 12 GiB, util 50~60% (train + closed_loop)

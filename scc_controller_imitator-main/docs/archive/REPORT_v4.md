# SCC Controller Imitator — Data & Training Report (v4)

작성일 2026-05-23 (02:30) · branch `main` · 모델 `src/model/image_models/iy_model.py`

REPORT_v3 (2026-05-22 14:00) 이후 추가/변경 (약 12시간 분량).

---

## 1. 변경 요약 (v3 → v4)

| 항목 | v3 (5-22 14:00) | v4 (5-23 02:30) |
|---|---|---|
| **합성 정책** | 24 shard, 3 track 전체 재합성 진행 중 (3%) | 합성 중단 (1.0M sample 보존), 신규 4 policy 별도 시작 후 spd_nolead 중단 |
| **신규 데이터** | spd 1 정책만 | **+ sangam_lead 2 + ci_aug (v0237+v0301+v0302) + DriverB_new (processing 중)** |
| **i1_cache backbone** | combined_v1 (mIoU 0.892) 만 | **fx640_v3 (mIoU 0.921)** 으로 신규 4 policy 통일 |
| **i_g obj_gate 학습** | spd 단독 ckpt | multi-prebake (spd + sangam_lead) + ci_aug 통합 |
| **GT augmentation** | 없음 | **brake_aug** (dist<15+vrel<-3 → ax=-15 floor) — ~72K sample 강제 brake |
| **closed_loop_node** | IyModel 패치 완료 | **NaN fix** (amp fp32) + **pedal post-process** (음의 ax 시 accel=0 강제) |
| **ckpt registry** | 분산 path | **CURRENT_BEST.json** + cache backbone consistency check |
| **viz 도구** | 1종 (eval samples) | + dashboard / per-ep eval / first-step brake / gt audit / dataset_dist / arch diagram (총 6종) |

---

## 2. 데이터 다양화 — multi-source ConcatDataset 학습

### 2.1 학습 sample 구성 (현재 best ckpt)
| source | samples | 비중 | 비고 |
|---|---|---|---|
| spd 기존 prebake | 339K | 27% | combined_v1 cache, close 92% |
| sangam_lead_0520 | 72K | 6% | fx640_v3 cache, close 85% |
| sangam_lead_0521 | 97K | 8% | fx640_v3 cache, obj_mean 0.011 |
| ci_aug v0237 | 720K | **58%** | image=0 padded, no-lead 시나리오 |
| ci_aug v0301+v0302 | 18K | 1% | .pt 풀린, image=0 padded |
| **합계** | **1.25M** | 100% | |

### 2.2 ControllerImitatorAugDataset (신규)
- `src/datamodule/controller_imitator_aug_dataset.py`
- controller_imitator 의 .pt / .pkl 양 형식 지원
- 각 sample 변환:
  ```
  tok_curr/tok_prev_raw = zeros(40, 256)  # image=0 padded
  ego_5d = ego_info[:5]
  traj_40 = trajectory_points_local
  target = [steer (clip 30°), acceleration (clip 20 m/s²)]
  obj_target = zeros  # no obj
  min_dist = inf  # no lead
  ```
- 의도: obj_gate 의 "객체 없으면 image=0" 행동 학습 + panic brake 감소

### 2.3 train_iy multi-source 옵션
- `--prebake-extra A.npz B.npz` + `--i1-cache-extra A.npz B.npz` (1:1 매핑)
- `--ci-aug-dirs DIR1 DIR2 DIR3` (image=0 padded 추가)
- `--reload-on-change` 시 multi-prebake mtime aggregate check
- close-lead 가중치는 ConcatDataset 의 ds_min_dist 로 매핑 (ci_aug 는 inf → weight 1.0)

---

## 3. closed_loop_node 3-bug fix

### 3.1 IyModel ckpt → 최신 streaming train
```python
MODEL_CKPT_IY = ".../iy_spd_sangam_ciaug_brakeaug15/best_loss.pt"
MODEL_USE_OBJ_GATE = True
```

### 3.2 NaN inference fix
- 원인: `autocast(enabled=True)` (fp16) — 학습은 fp32, transformer softmax exp() 가 fp16 dynamic range (~65504) 초과 → NaN
- fix: `enabled=False` (fp32 강제). 학습 dtype 일치

### 3.3 Pedal map post-process
- 원인: `pedal_map_a1_f4.csv` 의 accel pedal map 음의 acc 영역에 비제로 (vx=30, ax=-1.0 → accel 0.32 false-positive)
- fix: ax_target 부호 따라 한 쪽만 활성 강제
  ```python
  if ax_target < 0: accel_pedal = 0.0
  elif ax_target > 0: brake_pedal = 0.0
  ```

### 3.4 SEG backbone 일치
- 이전: closed_loop SEG_BACKBONE_CKPT = v2s_full (mIoU 0.731)
- spd_v2 cache backbone = combined_v1 (mIoU 0.892) → mismatch → live FP
- fix: combined_v1 로 일치. 신규 cache (fx640_v3) 와는 여전히 mismatch — 다음 라운드에 통일

---

## 4. GT brake augmentation (brake_aug)

### 4.1 GT 결함 발견 (audit)
3,197 close-lead logged sample 분석:
| dist (approach vrel<-2) | n | GT ax mean | brake% |
|---|---|---|---|
| 5-10m | 126 | **-0.04** | 12% ❌ |
| 10-15m | 343 | -1.18 | 22% |
| 15-20m | 105 | -2.36 | 39% |
| 20-30m | 140 | **-5.92** | 53% ✅ |
| 30-50m | 25 | -4.49 | 100% ✅ |

→ **MPC 합성 GT 가 가까울수록 brake 약함** (역설). 5-10m 매우 가까운데 cruise 결정. MPC 의 SCC replanner 결함.

### 4.2 brake_aug 적용
- `scripts/augment_prebake_brake.py`
- rule:
  ```
  dist < 15 + vrel < -3 → target[:, 1] = min(orig, -15) m/s²
  dist < 25 + vrel < -2 → target[:, 1] = min(orig, -3) m/s²  (also-mild)
  ```
- 결과: spd 62K + sangam_lead 9.6K = **~72K sample 의 GT 강제 brake floor**
- target prebake: `*_brakeaug.npz` (별도 파일, 원본 보존)

---

## 5. cache 시스템 개선 (registry + consistency check)

### 5.1 발견 (5-23 01:30)
신규 4 policy cache 만들 때 backbone = v2s_full (mIoU 0.731). 그러나 더 좋은 `fx640_v3 (mIoU 0.921)` 가 5-21 15:44 에 있었음. 단순 path convention 불일치로 잘못된 backbone 사용.

### 5.2 개선 (단일 source-of-truth)
**`data/ckpts/CURRENT_BEST.json`** 신설:
```json
{
  "seg_backbone": {"path": "outputs_multi/seg_pretrain_fx640_v3/best_miou_backbone.pt", "miou": 0.921},
  "f1_iy_model": {"path": "outputs_multi/iy_spd_sangam_ciaug_v0237_noreload/best_loss.pt", ...}
}
```

**`src/utils/ckpt_registry.py`** — get_best / register_best API

### 5.3 자동 default + consistency
- `compute_i1_tok_cache.py` 가 `--seg-ckpt` 미명시 시 registry 의 best 사용
- cache npz meta 에 backbone metric 도 저장
- `train_iy.py` 가 multi-cache 시작 시 backbone 불일치 warning

### 5.4 i1_cache 재생성 (신규 4 policy, fx640_v3)
- spd_nolead_0520 (34K), spd_nolead_0521 (111K), sangam_lead_0520 (19K), sangam_lead_0521 (24K)
- 4 process 병렬 GPU, ~30분 (예상 70분 보다 빠름)
- 신규 cache 모두 fx640_v3 통일. spd_v2 (combined_v1) 만 다름

---

## 6. f_1 학습 진화

| ckpt | best val | steer | ax | 변경 |
|---|---|---|---|---|
| iy_spd_v1_full | 0.337 | - | - | acados |
| iy_spd_streaming_w20 | 0.151 | 0.24° | 1.27 | mpc warmup 20 |
| iy_spd_streaming_objgate_close | 0.139 | 0.26° | 1.15 | + obj_gate + close-lead |
| iy_spd_multi_ciaug | 0.123 | 0.26° | 1.15 | + sangam + spd_nolead + ci_aug(v0301/v0302) |
| iy_spd_sangam_ciaug_v0237_noreload | 0.145 | 0.77° | 0.68 | + ci_aug v0237 (720K). **ax 40% 개선** |
| **iy_spd_sangam_ciaug_brakeaug15** (가동중) | TBD | TBD | TBD | + brake_aug -15 + close_loss_mult 3 |

### 6.1 첫 ep 결과 (brakeaug15)
- ep 1: val 0.149, steer 0.71°, ax 0.78. close ax 1.36 / far ax 0.53
- 이전 ckpt (v0237) 의 ep 1 (0.176) 보다 ↓ — brake_aug 효과 가시

---

## 7. 검증 도구 (6종 + dashboard)

| script | 용도 |
|---|---|
| `dashboard_status.py` | 60s auto-refresh live dashboard (학습/합성/CPU/GPU) |
| `dataset_dist_report.py` | 기존+신규 4 policy 분포 비교 (close-lead/vx/steer/ax) |
| `viz_iy_eval_samples.py` | 24 mixed sample inference (GT vs pred curve) |
| `viz_iy_per_epoch_loop.py` | best_loss.pt mtime watch → 4 category (no-lead/approach/cruise/recede) × 4 sample grid |
| `viz_iy_first_step_brake.py` | close-lead first-step ax (즉시 제어 명령) brake 정확도 + worst case |
| `viz_gt_brake_audit.py` | GT (MPC 합성) brake quality audit. dist×vrel heatmap |
| `viz_arch_diagram.py` | IyModel 모델 구조 paper-style 다이어그램 |

http://localhost:8010/ — 통합 viewer (dashboard + 모든 viz)

---

## 8. first-step brake validation (이전 ckpt v0237_noreload, ep 8 ckpt)

500 close-lead sample 평가:
- **first-step ax MAE: 1.17 m/s²**
- **brake sign-match: 82.2%** (TP 194 / FN 75 / FP 14 / TN 217)
- **brake recall: 72.1%** (28% brake 상황 놓침)
- **brake precision: 93.3%** (false brake 매우 낮음)
- **strong brake (GT<-3) detection: 89.9%**

→ 강한 brake 잘 인식, mild brake 영역에서 cruise 회귀. brake_aug 후 개선 기대 (목표 85%+).

---

## 9. DriverB_new 처리 (진행 중)

### 9.1 데이터
- 위치: `/home/ailab-12/AILabDataset/02_Custom_Dataset/57_A1_E2E/00_raw/sangam_track/DriverB_new`
- **569 runs, 135K frame**, fx=640
- multi-agent (Car_2, Car_4 등 여러)
- 분포 추정: **close 24% / no-lead 54% / far 22%** — 균형 좋음

### 9.2 처리 plan (logged-only, 합성 안 함)
1. frame_list (135K, stride 1) ✅
2. meta json ✅
3. logged bank (진행중) — 30분 ETA
4. aux_targets (진행중) — 30-60분 ETA
5. i1_cache (fx640_v3, GPU)
6. close_lead_meta v5 추가
7. **prebake-logged-only script** — EGO log 의 future 40 step steer/ax 직접 target. MPC 합성 X

---

## 10. 가동 중 (현재 02:30)

| process | 상태 |
|---|---|
| train_iy (PID 219782) | brakeaug15 학습 진행 (ep 2 in progress) |
| viz_iy_per_epoch_loop (PID 81038) | best ckpt watch (per-ep viz) |
| dashboard_loop (PID 4033369) | port 8010 |
| logged_traj (DriverB_new) | 진행 중 |
| aux_targets (DriverB_new) | 진행 중 |
| augment_prebake 2 (zombie) | wind-down |
| synth_batch | 0 (모두 중단) |
| streaming_prebake_loop | 0 (kill 됨) |

---

## 11. 다음 단계 / 미해결

| | 항목 | 우선 |
|---|---|---|
| ☐ | DriverB_new 처리 완료 (i1_cache + prebake-logged-only + 학습 추가) | **High** |
| ☐ | brake_aug 학습 ep 5-10 후 first-step brake recall 재측정 (목표 85%+) | **High** |
| ☐ | spd_v2 cache 도 fx640_v3 으로 통일 + 학습 재시작 (현재 mixed backbone) | Med |
| ☐ | closed_loop 실차/bag 재테스트 (학습 안정화 후) | Med |
| ☐ | pedal_map_a1_f4.csv 자체 데이터 보정 (음의 acc 영역) | Med |
| ☐ | sangam_lead 합성 더 진행 여부 결정 (현재 10%) | Low |
| ☐ | other controller_imitator versions (v0229/v0230/v0231) 평가 | Low |
| ☐ | #13 backbone v2_l / #14 backbone b0/b1 ablation | Low |
| ☐ | GT 결함 근본 해결 — MPC SCC replanner close-lead brake floor 추가 | Low (큰 작업) |

---

## 12. 시계열 commit 요약 (v3 → v4, 6 commit)

```
364a0e0 data(raceline_sangam_w_pit): pit 포함 원본 보존
600c414 data(raceline_sangam): pit 제거
c696820 feat(viz): dashboard + per-ep eval + dist + eval_samples
7dc90c2 fix(closed_loop_node): NaN inference + pedal map 3건 fix
5b190b7 feat(train_iy): multi-prebake/cache ConcatDataset + ci_aug 옵션
15c54fb feat(datamodule): ControllerImitatorAugDataset — image=0 padded no-lead aug
```

추가 (commit 예정):
- ckpt_registry + compute_i1 default + consistency check
- augment_prebake_brake.py (GT brake floor)
- viz_iy_first_step_brake.py + viz_gt_brake_audit.py + viz_iy_per_epoch_loop.py
- closed_loop_node seg_backbone 통일 (combined_v1)

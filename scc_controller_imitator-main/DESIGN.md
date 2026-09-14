# scc_controller_imitator 설계

> 구현 전 합의용 문서. 본 문서가 확정된 후에 코드(특히 `src/model/image_models/`,
> `src/datamodule/scc_dataset.py`)를 본 설계에 맞게 재구성한다.

---

## 0.0 변경 이력 (Update Log)

| 버전 | 일자 | 핵심 변경 | 영향 섹션 |
|------|------|----------|-----------|
| v1 | 2026-05-15 | 초기 설계 (pixel-level yaw warp, fx=320 sangam MultiAgnet only) | 전체 |
| **v2** | 2026-05-20 | **i_y refactor** — pixel warp → token-level integer bin shift (`np.roll`), per-image i_1 cache 도입 | §3, §4 |
| **v2** | 2026-05-20 | **fx-aware per-sample tensor** — DriverB_new / agent1 등 fx=640 데이터 대응 | §1, §3 |
| **v2** | 2026-05-20 | **close-lead 가중** — WeightedRandomSampler + per-sample loss weight (가까운 차량 학습 강화) | §7 |
| **v3** | 2026-05-21 | **i_g image gate** — fi_mem 에 a_2 mini-head 예측 obj_prob per-bin gate. 객체 없는 frame 의 image 영향 동적 차단 | §4, §5 |
| **v3** | 2026-05-21 | **fx=640 only 정책** — MultiAgnet fx=320 학습 데이터 폐기. speedway agent1 + 신규 sangam fx640 (0519_pert{O,X}_pitX) 만 사용 | §2 |
| **v3** | 2026-05-21 | **multi-track data** — sangam + speedway 통합 학습 (1.seg, 2.a_2). f_1 은 track 별 ckpt 분리 | §2, §7 |
| **v4** | 2026-05-23 | **brake_aug** (close lead 강제 brake floor) + **ci_aug** (controller_imitator data image=0 padded). closed_loop NaN/pedal fix + ckpt registry | §6, §7 |
| **v4** | 2026-05-23 | **ControlNet self-attn side branch** (v18-v22) — parallel 6L side. close-lead ax loss plateau 1.0 m/s² 결과 | §5 |
| **v5** | 2026-05-24 | **redesign: f_1 (main) + f_2 (Side SCC) 명명 통일.** main_no_image + side cross-attn (Q=traj, KV=image_mem) + a_2 v4 feat early inject + TTC label augment | §4, §5, §7 |
| **v5** | 2026-05-24 | **🚨 critical fix: `build_a2_cache.py` proj 누락 버그.** i_1 cache 는 proj 전 raw. 빌더가 a_2.encoder.proj 통과 안 시킴 → fusion 입력 분포 mismatch → AUC 0.51 (random). fix 후 AUC 0.98. v18-v25 시리즈 모두 영향. [INCIDENT_a2_cache_proj_bug.md](docs/INCIDENT_a2_cache_proj_bug.md) | §4 |
| **v5** | 2026-05-24 | **모듈 분리 issue 발견** — IyModel 의 image 처리 (img_proj/img_fusion) 가 main 에 융합. main_no_image=True + main frozen 시 image_mem 학습 안 됨 (random init 유지). [architecture.md §3](architecture.md) | §4, §5 |
| **v5** | 2026-05-24 | f_1 baseline 확정: **v25c val 0.37** (Professor #654 RAW finetune, main_no_image, lr 3e-4 + decaying cosine 5 cycles). 명명 [`f1_f2_naming`](memory/f1_f2_naming.md) | §0, §7 |
| **v6** | 2026-05-25 | **architectural redesign: 모듈 ckpt-level 분리**. 발견된 핵심 design failure: a_2 의 `ImageFusionBlock` 과 iy 의 `ImageFusionBlock` 이 같은 이름이지만 (1) class 별도 정의, (2) dimension 다름 (256 vs 128), (3) spec 약간 다름 → a_2 학습된 image fusion weights 가 iy 에 활용 안 됨. img_proj 가 학습 path 에 없는 고아 모듈. | §4, §5 |
| **v6** | 2026-05-25 | **plan: a_2 v5 = encoder+fusion frozen (v4 ckpt) + 새 fc_proj (256→128) + 새 AuxHead Linear(128→7) 만 학습.** 학습 후 fusion+fc_proj 까지 inference → **i_2 cache (B, 40, 128) 빌드** → iy_model 의 image module (img_proj+img_fusion) 제거 → i_2 cache 직접 image_mem 입력 → main (traj+ego+image_mem concat self-attn) + f_2 (cross-attn) 학습 | §4, §5, §7 |
| **v6** | 2026-05-25 | 🚨 **3일 손실 진단 (5-22~5-24)**: ① v18-v22 plateau = a_2 cache proj 누락 버그 (image-grounded 학습 신호 noise). ② v23-v25 plateau = Professor RAW resume 으로 img_proj reset + main_no_image=True 우회. img_proj 는 5-21 IyModel 처음 도입부터 존재 (설계는 OK), 그러나 학습 path 보장 부재가 본질 문제. v25c 만 image 비사용 baseline 으로 살아남음. | history |

---

## 0. 목표 재정의

**controller_imitator 의 출력 계약(steer + 종방향 명령)을 유지·확장**하면서, 입력에
**전방 이미지 (현재 + 이전, 이전은 Δyaw 회전 보정)** 를 추가하여 선행차 인지 등의
SCC 기능을 신경망이 얻도록 한다.

종방향 명령은 두 가지 컨벤션을 **공존** 시킨다:
- (기존) **페달 신호** `accel_brake ∈ [-1, +1]` — controller_imitator 호환
- (신규·기본) **가속도 명령** `accel (m/s²)` — 실제 차량 ax 와 단위 일치, raw 사용 (정규화 없음)

⇒ 본 프로젝트는 controller_imitator 의 **이미지 + 가속도-명령 확장판**. target /
output_shape / 손실 계약은 controller_imitator 와 동일한 분기 구조를 유지하되,
`accel` / `steer_accel` 두 옵션을 추가하고 **default 는 `steer_accel`** 로 둔다.

---

## 1. 입력 / 출력 요약

### 1.1 입력
| 키 | shape | 좌표계 | 비고 |
|----|-------|--------|------|
| `image_curr` | `(B, 3, 250, 1280)` | 현재 카메라 프레임 | 전방 RGB, 1280×720 원본에서 horizon strip y=235~485 crop, ImageNet 정규화 |
| `image_prev_rot` | `(B, 3, 250, 1280)` | 현재 카메라 프레임 | t-Δt 프레임을 **Δyaw 만큼 회전만** 보정 후 동일 ROI crop |
| `trajectory_points_local` | `(B, 30, 2 or 3)` | 현재 ego 로컬 프레임 | jiwon_e2e 의 `wp_rel_gt` 를 누적 후 (x, y, v) 로 정렬 — controller_imitator 의 `trajectory_points_local` 규약과 호환 |
| `trajectory_time` | `(B, 30, 1)` | — | `input_time_to_traj != "none"` 일 때만. concat 시 최종 `traj_dim` = 2/3/4 |
| `ego_info` | `(B, 5)` | — | `[vx, vy, ax, ay, yaw_rate]` — vehicle frame, SI 단위 (m/s, m/s², rad/s) |

### 1.2 출력
controller_imitator 의 분기 구조 유지 + **가속도 명령(accel) 옵션 추가**:
- `target ∈ {"steer", "accel_brake", "steer_accel_brake", "accel", "steer_accel"}`
  - `steer`             → 1-d  (조향만)
  - `accel_brake`       → 1-d  (페달 신호, ∈ [-1, +1])           — controller_imitator 호환
  - `steer_accel_brake` → 2-d  (조향 + 페달)                     — controller_imitator 호환
  - `accel`             → 1-d  (가속도 명령, m/s², 정규화 없음)  — **신규**
  - `steer_accel`       → 2-d  (조향 + 가속도 명령)              — **신규 · 기본값**
- `output_shape ∈ {"control", "control_ego"}`
  - `control` → `(B, 30, D_control)`
  - `control_ego` → `(B, 30, D_control + 4)` (ego x, y, heading, speed 추가)
- 채널 순서: `[control...] + (옵션) [ego x, y, heading, speed]`
- `steer_accel` 의 채널 순서는 `[steer, accel]` (controller_imitator 의 `[steer, accel_brake]` 와 동일한 순서)

### 1.3 horizon 길이 (T = 40, v3)
- v3 (2026-05-21) 변경: **T=40** 으로 controller_imitator 와 동일 (이전 30 → 40).
- 100 ms × 40 = **4 s**. trajectory / target / mask 모두 40-step.
- 코드: `IyModel(traj_seq_len=40)`, prebake `traj (N, 40, 3)`, `target (N, 40, 2)`, `mask (N, 40)`.

---

## 2. 데이터셋

### 2.1 원본 위치 (jiwon_e2e 자산 재활용)
```
data/raw/<track>/<policy>/<run_id>/
  ├ CAMERA_1/<tick>_Intensity.jpeg   ← 1280×720 JPEG (전방)
  ├ EGO_INFO/<tick>.txt              ← yaw/vel/accel/brake/steer 포함
  └ OBJECT_INFO/<tick>.txt           ← 선행차 (멀티에이전트 정책에서만)
```

### 2.2 학습에 쓸 정책
| 정책 | 상태 | 비고 |
|------|------|------|
| `MultiAgnet` (sangam_track) | 6,877 frame, pkl 캐시 완료 | 1차 학습 데이터 |
| `MultiAgentSCC` (sangam_track) | ~150+ run, raw 만 존재 | 추후 인덱싱 후 추가 |
| `expert_jw`, `rule_v2`, `pitstop` | OBJECT_INFO 비어있음 (단일차량) | 선행차 학습 신호 없음 → **제외** |

### 2.3 jiwon_e2e 모듈 재사용
- `src/data/ego_parser.py::parse_ego_info` + `to_si_pose6`
- `src/data/future_traj.py::build_future_traj` (30-step, 100 ms)
- `src/data/calib_loader.py::load_camera_calibration` (CAMERA_1 K=fx=fy=320, cx=640, cy=360)
- `src/data/raw_index.py` 의 tick 인덱싱 + dt_per_tick_ms 처리
- `src/train/dataset.py::FramePklDataset` 의 temporal pair 로직 (`step_ticks = round(100/dt_per_tick_ms)`) + `_delta_ego` (이미 wp_rel[0] 로부터 Δyaw·Δx·Δy 추출)

### 2.4 jiwon_e2e 의 pkl 그대로 쓸 수 없는 이유
- pkl 에 `accel/brake/steer` **학습 라벨이 없음** (BEV·tracking 중심으로 빌드됨)
- 5채널 BEV `I_bev`, `sem_bev`, `free_bev`, `objects` 는 SCC 확장에 불필요 → 로딩 비용 낭비

⇒ 본 프로젝트는 jiwon_e2e index.json (840 run / 175,801 frame) 의 frame_id 만 가져오고,
**raw EGO_INFO + CAMERA_1 만** 직접 읽는 **얇은 자체 dataset** 을 만든다.

### 2.5 라벨 (supervision) 추출
raw `EGO_INFO/<tick>.txt` 에서:
- `accel: 0..1` (가속 페달)
- `brake: 0..1` (브레이크 페달)
- `steer: 도(deg)` (±7° 관측)
- `ax: m/s²` (종방향 가속도 측정값, vehicle frame)

라벨 변환:
- **조향**: `target_steer = steer / 40.0` (controller_imitator 의 `loss_weight_steer=40` 컨벤션, 즉 1.0 = 40°)
- **페달 (controller_imitator 호환)**: `target_accel_brake = accel - brake ∈ [-1, +1]` (1-d 통합 신호)
- **가속도 명령 (신규, 기본)**: `target_accel = a_cmd (m/s²)` — **raw, 정규화 없음**
  - 1차 소스: **MPC 솔브 결과** (`run_MPC.py` 등 별도 파이프라인에서 사전 산출, run 별 캐시)
  - fallback: **EGO_INFO 의 ax** (측정값. MPC 캐시 미보유 run 에서 사용)
  - 두 경로 모두 `m/s²`, vehicle frame, 부호 그대로 (감속 = 음수)
  - `target == "accel" | "steer_accel"` 일 때만 필요. 캐시 미보유 frame 은 `valid_mask=0`

미래 30-step (100 ms 간격) 동안 선택된 target 값을 시퀀스로 수집:
- `target_steer_accel:       (30, 2)` ← 기본
- `target_steer_accel_brake: (30, 2)` ← controller_imitator 호환
- 단일 target 은 `(30, 1)`

---

## 3. 이전 이미지 회전 보정 — i_y (v2: token-level integer bin shift)

### 의도
두 이미지가 같은 카메라 방향으로 정렬 → 이동 시차는 그대로 → 선행차 접근률·차로 변화 정보가 신경망에 노출.

### v1 (deprecated): pixel-level warp
- pure-rotation homography H = K · R_y(Δyaw) · K⁻¹, prev 이미지에 `cv2.warpPerspective` 적용
- 단점: prev_dt 바뀌면 i_1 backbone forward 재계산. border replication artifact. 다양한 Δt 실험 불편

### v2 (현재): token-level integer bin shift
- prev 이미지를 raw 로 i_1 backbone 통과 → `fi_tok_prev` (B, 40, D)
- `dx = round(fx · Δyaw / bin_w_px)` (bin_w = 1280/40 = 32 px)
- `fi_tok_prev_y = np.roll(fi_tok_prev, dx)` (border 는 wrap)
- **i_1 backbone forward 는 prev_dt 무관, 이미지 단위로 1 회 cache 가능**
- fx=320: 1 bin = 5.7°. fx=640: 1 bin = 2.86°. Δyaw 1.7° (50ms × 34°/s) → 0.6 bin (round=1)
- sub-bin 정밀도 손실 있지만 실험 유연성/속도 ↑

### Δyaw 계산
- `Δyaw = yaw_t - yaw_{t-Δt}` (curr_ego.orientation[2] - prev_ego.orientation[2], rad)
- prev_dt 기본 200ms (prev_dtick=20 tick, 1 tick = 10ms 가정)

### fx (per-sample tensor, v2)
- 단일 batch 에 fx 다른 sample 가능 (speedway fx=640 + sangam fx=320 등)
- 코드: `yaw_shift_tokens_batch(prev_tok, d_yaw, fx)` 에서 fx 가 scalar 또는 (B,) tensor 둘 다 지원
- DRIVER_FX 매핑 (prebake_iy_dataset.py): policy name substring 으로 자동 결정
  - MultiAgnet → 320 (deprecated)
  - 그 외 (agent1, DriverB_new 등) → 640

### 카메라 intrinsic
- `K` (3x3) — `calib_loader` 에서 자동 로드 (legacy). v2 에서는 fx 만 필요 (token shift)
- `Δt` — prev_dtick × dt_per_tick_ms = 200ms (canonical 10ms × 20 tick)

---

## 4. 이미지 인코더: `LateralStripEncoder`

### 4.1 ROI
- 원본 1280×720 → ROI `y ∈ [235, 485)`, `x ∈ [0, 1280)` → **250×1280** strip
- 픽처 횡축 (1280) 전체 보존, 횡방향 위치가 lateral 토큰 인덱스로 직접 매핑됨

### 4.2 토큰화: (3, 250, 1280) → (40, D)

이미지 백본은 **`img_backbone` config 로 swap 가능**. 모든 옵션이 ImageNet pretrained 이며
total stride 32 → 1280 / 32 = 40 lateral 토큰을 자연스럽게 생산.

```
input (3, 250, 1280)
  ↓ backbone (timm features_only, out_indices=[-1])
spatial map (C_b, 8, 40)        ← 250/32≈8, 1280/32=40
                                 ← C_b 는 백본별로 다름 (표 참고)
  ↓ AdaptiveAvgPool2d((1, 40))   ← 세로 평균 풀링
(C_b, 40)
  ↓ Conv1d(C_b → D=256, k=1)    ← 채널 정규화 (백본 무관 통일)
(D, 40)
  ↓ permute
lateral tokens (40, D)
```

**지원 백본** (timm 기준):

| `img_backbone` 값 | params | FLOPs @ 224² | 마지막 spatial 채널 `C_b` | 비고 |
|-------------------|--------|---------------|----------------------------|------|
| `resnet18` | 11.7 M | 1.82 G | **512** | 기준선. wall-clock 빠름 |
| `efficientnet_b0` | 5.3 M | 0.39 G | **320** | **권장 기본값**. jiwon_e2e BEV encoder 와 동일 |
| `efficientnet_b1` | 7.8 M | 0.57 G | 320 | B0 보다 정확도 ↑ |
| `tf_efficientnetv2_s` | 21.5 M | 2.85 G | **256** | 모던 V2. 더 큼이지만 학습 속도 ↑ |
| `tf_efficientnetv2_b0` | 7.1 M | 0.50 G | **192** | V2 의 small variant |
| `tf_efficientnetv2_b1` | 8.1 M | 0.69 G | 192 | |
| `mobilenetv3_large_100` (옵션) | 5.5 M | 0.22 G | 960 | 가장 가벼움 |

**구현 노트**:
- 1×1 `Conv1d(C_b → D)` 가 백본별 채널 차이를 흡수 → 모델 구조의 나머지는 동일.
- `timm.create_model(name, pretrained=True, features_only=True, out_indices=[-1])` 한 줄로 백본 교체.
- siamese 공유 가중치는 백본 선택과 무관 (인스턴스 1개 만들어서 양쪽에 호출).

### 4.3 두 프레임 처리 (siamese)
공유 가중치 `LateralStripEncoder` 를 `image_curr` / `image_prev_rot` 양쪽에 적용:
- `tokens_curr  : (B, 40, D)`
- `tokens_prev  : (B, 40, D)`

두 토큰 셋은 그대로 cross-attn 에 들어가지 않고, 먼저 §4.4 의 `ImageFusionBlock` 에서
**curr Q × prev K/V 의 cross-attn 으로 40 토큰 (=image_memory)** 으로 축약된다.

### 4.4 ImageFusionBlock (두 프레임 → 40 토큰)

**의도**: 현재 lateral 위치 `i` 의 출력 토큰이 **"위치 i 에 있는 객체의 상태"** 를 담도록 한다:
- 해당 위치에 객체가 있는지 (objectness)
- 횡방향 / 종방향 속도 성분 (lateral pixel shift + depth-flow 단서)
- 가까움 / 인식 크기 (feature 활성화 강도 + lateral 폭)

#### 4.4.1 왜 4 단계인가 — 거리별 토큰 span

카메라 K=320 (fx), 차폭 1.8 m, 토큰 폭 1280/40 = 32 px:

| 차량 거리 | 픽셀 폭 | 토큰 span | 의미 |
|----------|---------|-----------|------|
| 3 m | ~190 px | **~6 토큰** | 매우 가까움. self-attn 필수 |
| 5 m | ~114 px | **~3.5 토큰** | SCC 핵심 거리. self-attn 효과 큼 |
| 10 m | ~57 px | ~1.8 토큰 | 경계 |
| 20 m | ~28 px | ~0.9 토큰 | 토큰 1 개 |
| 50 m | ~11 px | < 0.5 토큰 | 토큰 1 개에 흡수 |

→ SCC 가 가장 중요한 ≤10 m 영역에서 객체가 다중 토큰을 span. cross-attn 매칭 전에
**같은 객체에 속한 토큰끼리 응집** 시키지 않으면 토큰 단위 매칭이 객체 단위와 어긋남.

#### 4.4.2 구조 (4 단계)

```
tokens_curr (B, 40, D) ─┐
tokens_prev (B, 40, D) ─┤
                        │
              ┌─────────▼──────────────────────────────┐
              │ Step 0: Per-Frame SelfAttn (shared)     │
              │   - same weights applied to curr & prev │
              │   - tokens_curr' = SA(curr) + curr      │
              │   - tokens_prev' = SA(prev) + prev      │
              │   + lateral_pos_emb [0..39]             │
              │   + FFN                                  │
              │   → 동일 객체 토큰들 응집 (intra-frame) │
              │   layers = image_fusion_pre_self_attn_layers (기본 1)
              └─────────┬──────────────────────────────┘
                        │
              ┌─────────▼──────────────────────────────┐
              │ Step 1: CrossAttention                  │
              │   Q  = tokens_curr'                     │
              │   K,V= [ego_motion_token, tokens_prev'] │
              │         └─ Linear(2 → D) on [vx, wz]    │
              │            (B, 1, D) prepended          │
              │            ⇒ K,V shape (B, 41, D)       │
              │   + frame_type_emb (Q=0, KV_prev=1,     │
              │                     KV_ego=2)           │
              │   + residual from Q + FFN               │
              │   → 객체 단위 매칭 (curr↔prev tracking) │
              │     + ego motion prior (apparent motion │
              │       ↔ 객체 실제 motion 분해 가능)     │
              │   layers = image_fusion_cross_attn_layers (기본 1)
              └─────────┬──────────────────────────────┘
                        ▼
              matched_tokens (B, 40, D)
                        │
              ┌─────────▼──────────────────────────────┐
              │ Step 2: (옵션) Post SelfAttn over 40   │
              │   - image_memory side 추가 정제         │
              │   - 인접 lateral 위치 간 정보 교환      │
              │   layers = image_fusion_post_self_attn_layers (기본 1)
              │   + FFN                                  │
              └─────────┬──────────────────────────────┘
                        ▼
              image_memory (B, 40, D)
                        │
                        └──► (downstream: trajectory cross-attn)

              [§4.4.5 보조 supervision heads — train-only, 추론 영향 없음]
```

#### 4.4.5 보조 supervision heads (train-only, 모두 baseline 비활성)

두 종류의 옵션 head 가 ImageFusionBlock 내부의 서로 다른 단계에 분기하여 붙는다.
추론 시에는 제거되므로 본 forward path 에 영향 없음.

##### (a) Token-level SegHead — 분기점: **Step 0 출력 (tokens_curr')**

목적: Step 0 self-attn 토큰이 "어디까지가 한 객체인가" 를 학습하도록 직접 supervision.

```
tokens_curr' (B, 40, D)              ← Step 0 출력 (self-attn 후)
       │
       ▼ tile vertically → (B, 8, 40, D)        ← 같은 token 을 H=8 만큼 복제
       │
spatial_feat (B, C_b, 8, 40)                    ← LateralStripEncoder pre-pool feature
       │   (백본에서 이미 계산된 spatial map. 별도 forward 없음)
       │
       └─► concat ch → (B, D + C_b, 8, 40)
                          │
                          ▼ Conv2d(D + C_b → K, kernel=1)
                  seg_logits (B, K, 8, 40)
                          │
                          ▼ CrossEntropy / focal loss
                  GT: SAM2 mask (250, 1280) → AdaptiveMaxPool2d to (8, 40)
                       → cell mode class, 5 classes
                       (void / road / vehicle / lane_line / wall, jiwon_e2e remap)
```

- 추가 파라미터: ~20K (Conv2d 1×1 한 개)
- 추가 compute: 백본 대비 < 0.1%
- LateralStripEncoder 가 `spatial_feat` 도 함께 반환하도록 시그니처 확장 필요.
- **이 head 가 `True` 일 때 `tokens_prev'` 에도 동일 SegHead 적용 가능 (shared weights)** —
  `image_fusion_seg_apply_to = both` 이면 supervision 2× 강화.

##### (b) Per-lateral AuxHead — 분기점: **image_memory (Step 2 출력)**

목적: 매칭이 끝난 토큰이 "각 lateral 위치의 객체 운동 / proximity" 를 명시적으로 회귀.

```
image_memory (B, 40, D)
       │
       ▼ Linear(D → 4)
  [objectness (sigmoid), v_lat (m/s), v_long (m/s), proximity (1/depth)]
       │
       ▼ BCE + L1 손실 (GT: OBJECT_INFO → 카메라 평면 lateral bin)
```

- 추가 파라미터: ~1K
- GT 빌더 별도 작업: OBJECT_INFO 의 cuboid → 카메라 평면 투영 → 40 bin 별 대표 객체

##### 두 head 의 관계

| 보조 head | 신호 종류 | 분기점 | 비용 | 역할 |
|----------|-----------|--------|------|------|
| SegHead | 분류 (K 클래스) | Step 0 후 | 거의 무료 | 객체 정체성 + 경계 (= 자아 인식) |
| AuxHead | 회귀 (4-d) | image_memory | 거의 무료 | 객체 운동 + 거리 (= 운동 인식) |

→ **보완 관계**. 둘 다 켜도 무방. 단 학습 안정성 위해 처음엔 SegHead 부터 켜는 게 자연스러움 (정체성 → 운동 순서).

#### 4.4.3 비용 (D=256, nhead=8, 모두 켤 때)

| 단계 | attention score 수 | 비고 |
|------|---------------------|------|
| Step 0 (pre, shared) | 2 × (40×40) = **3,200** | curr + prev 각 1 회 |
| Step 1 (cross) | 40×41 = **1,640** | K/V 에 ego_motion_token 1 개 prepend (+40 score) |
| Step 2 (post, 옵션) | 40×40 = **1,600** | |
| **합계** | ≤ 6,440 / layer-set | 백본 forward (수 G FLOPs) 대비 무시 가능 |

#### 4.4.4 Config 키

```ini
image_fusion_pre_self_attn_layers  = 1   ; Step 0. per-frame self-attn (shared weights, 0=off)
image_fusion_cross_attn_layers     = 1   ; Step 1. cross-attn Q=curr, KV=prev (+ego token)
image_fusion_post_self_attn_layers = 1   ; Step 2. post self-attn over 40 (0=off, ablation)

; ---- Step 1 ego motion prior (token prepend) ----
image_fusion_ego_prepend           = True       ; Step 1 K/V 에 ego_motion_token 1 개 prepend
image_fusion_ego_channels          = vx, wz     ; ego 5채널 중 사용할 인덱스. 기본 [vx, wz]
                                                ;   - vx : forward zoom-in (Δyaw warp 가 보정 못 함)
                                                ;   - wz : Δyaw warp residual prior
                                                ; (full = vx,vy,ax,ay,yaw_rate. 후속 ablation)

; ---- 보조 supervision heads (모두 baseline 비활성, train-only) ----
image_fusion_seg_head              = False  ; (a) token-level SegHead (Step 0 분기, 8×40 cells)
image_fusion_seg_classes           = 5      ;     void/road/vehicle/lane_line/wall
image_fusion_seg_weight            = 0.5
image_fusion_seg_apply_to          = curr   ;     curr | both (curr+prev shared head)

image_fusion_aux_head              = False  ; (b) per-lateral 4-d AuxHead (image_memory 분기)
image_fusion_aux_weight            = 0.1

image_fusion_nhead                 = 8
image_fusion_dim_feedforward       = 512
image_fusion_dropout               = 0.1
image_frame_type_emb               = True   ; Step 1 에서만 사용. 3가지 type id: Q=0, KV_prev=1, KV_ego=2
image_lateral_pos_emb              = True   ; Step 0, 1, 2 공통. ego_motion_token 은 pos_emb 부여 안 함 (위치 없음)
```

ego_motion_token 구현 메모:
- 입력: `ego_info[:, idx(vx,wz)]` → `(B, 2)`
- `Linear(2 → D)` + (옵션) LayerNorm → `(B, D)` → `unsqueeze(1)` → `(B, 1, D)`
- Step 1 의 K, V 시퀀스 맨 앞에 concat: `[ego_token, tokens_prev']` → `(B, 41, D)`
- `frame_type_emb` 가 ego token / prev token 을 구분 (각자 다른 embedding id)
- Q 쪽 (`tokens_curr'`) 에는 영향 없음 (40 토큰 유지)
- 후속 trajectory↔image cross-attn 에서는 image_memory (B, 40, D) 만 사용 (ego token 은 image_memory 에 포함하지 않음)

---

## 5. 전체 모델 아키텍처

### 5.1 단일 모델 (베이스라인): `TwoImgTrajEgoController` → v3: `IyModel`

v2/v3 변경 (DESIGN §0.0 참조):
- **i_1 backbone** 은 image 단위 cache (token level shift 와 짝)
- **i_y**: prev image 의 fi_tok 에 `np.roll` integer bin shift (pixel warp 대체)
- **i_g (v3 신규)**: fi_mem 에 a_2 mini-head 예측 obj_prob per-bin gate. 객체 없을 시 image 영향 차단
- **fx-aware**: per-sample fx tensor (DriverB_new fx=640, MultiAgnet fx=320 등)

```
[image_curr]  ─► i_1 LateralStripEncoder (frozen / cached) ─► fi_tok_curr (B, 40, C_b=256)
[image_prev]  ─► i_1 (동일, 같은 frame 의 1회 cache) ──────► fi_tok_prev (B, 40, C_b)
                                                                  │
                                                  ┌───────────────▼─────────────┐
                                                  │ i_y  Δyaw token shift       │
                                                  │   dx = round(fx · Δyaw/32)  │
                                                  │   np.roll(fi_tok_prev, dx)  │
                                                  └───────────────┬─────────────┘
                                                                  ▼
                                                          fi_tok_prev_y (B, 40, C_b)
                                                                  │
fi_tok_curr ─┐     ┌────── img_proj (Conv1d C_b → D=128) ────────┘
             ├─────┤
             ▼     ▼
       ┌─────────────────────────────────────────────────┐
       │ i_2  ImageFusionBlock (3 단계)                  │
       │   0) Per-frame SelfAttn (shared)               │
       │   1) CrossAttn Q=curr, KV=[ego_token, prev']   │
       │      ego_token = Lin(2→D)([vx, wz])             │
       │   2) Post SelfAttn over 40                     │
       └────────────────────┬────────────────────────────┘
                            ▼
                     fi_mem (B, 40, D)
                            │
                ┌───────────▼────────────┐    (v3 신규)
                │ i_g  Image Gate         │
                │  obj_logit = Linear(D→1)│
                │  gate = sigmoid(.)      │
                │  fi_mem_gated =         │
                │    fi_mem * gate        │
                └───────────┬────────────┘
                            ▼
                     fi_mem_gated (B, 40, D)        ← 객체 없으면 ≈ 0
                            │
trajectory (B, 40, 3) ─► t_1 traj_embed (3→D) ──┐  │
                                                ▼  │
ego (B, 5)  ─► e_2 state_embed (5/6 pad → D) ──→ t=0 토큰에 add
                                                ▼
                              tokens_text (B, 40, D) + pos_enc
                                                ▼
                              concat([tokens_text, fi_mem_gated]) (B, 80, D)
                                                ▼
                              f_1: Professor 6L TransformerEncoder (warm-start #654)
                                                ▼
                              fc_out (B, 40, 2)  = [steer °, ax m/s²]

train-time aux: obj_logit → BCE vs obj GT (gate 학습)
loss = L1(steer/ax) / scales + α × BCE(obj_logit, obj_target)
```

(legacy v1 그림은 §5.4 참조)
                                                        │
trajectory_points_local (B, 30, F_traj) ─► [traj embed] ─► traj_tokens (B, 30, D)
                                                        │
ego_info (B, 5) ─► [ego MLP] ─► ego_emb (B, D) ──┐      │
                                                  ▼      │
                              traj_tokens[:, 0, :] += ego_emb
                              (t=0 토큰에만 더함, 나머지 29 토큰은 그대로)
                                                        │
                                                        ▼
                                              [Cross-Attention × N
                                               + self-attn + FFN]
                                              Q = traj_tokens (30, D)     ← ego-conditioned at t=0
                                              K, V = image_memory (40, D)
                                                        │
                                              self-attn 을 통해 ego 정보가
                                              30 토큰 전체로 전파
                                                        │
                                                  fused_tokens (B, 30, D)
                                                        │
                                                  [head] → (B, 30, D_out)
                                                        │
                                                  controller_imitator 와 동일
                                                  D_out = 1 / 2 / 5 / 6 / ...
                                                  (target × output_shape 조합)
```

### 5.2 controller_imitator 와의 동치성
이미지 토큰을 cross-attention 의 K/V 로만 추가하므로, 이미지 입력이 없을 때 (`image_curr=zeros`)
원래 controller_imitator 의 `CrossAttentionControlPredictor` 와 거의 동일한 동작.
→ 이미지 ablation 시 모델 구조 변경 없이 평가 가능 (ImageFusionBlock 도 zero 입력에 대해 zero-like 출력).

### 5.3 forward 시그니처
```python
output = model(
    trajectory,               # (B, 30, F_traj)  positional, controller_imitator 와 동일 자리
    ego,                      # (B, 5)
    image_curr=...,           # kwarg
    image_prev_rot=...,       # kwarg
)
# returns (B, 30, D_out)
```
- 첫 두 positional 인자는 `forward(traj, ego)` 계약 유지 → 기존 train loop 와 호환.
- 이미지 두 장은 kwarg → train loop 에서 `batch["image_curr"]`, `batch["image_prev_rot"]` 추가 전달.

---

## 6. batch dict 키

| 키 | shape | dtype | 비고 |
|----|-------|-------|------|
| `image_curr` | `(B, 3, 250, 1280)` | float32 | ROI crop + 정규화 완료 |
| `image_prev_rot` | `(B, 3, 250, 1280)` | float32 | Δyaw 보정 + ROI crop |
| `trajectory_points_local` | `(B, 30, 2 or 3)` | float32 | `wp_rel_gt` 누적 → 절대 (x, y[, v]) |
| `trajectory_time` | `(B, 30, 1)` | float32 | `input_time_to_traj != "none"` 일 때만 |
| `ego_info` | `(B, 5)` | float32 | `[vx, vy, ax, ay, yaw_rate]` (vehicle frame, SI) |
| `target_steer` | `(B, 30, 1)` | float32 | `target == "steer"` 일 때 |
| `target_accel_brake` | `(B, 30, 1)` | float32 | `target == "accel_brake"` 일 때 (controller_imitator 호환) |
| `target_steer_accel_brake` | `(B, 30, 2)` | float32 | `target == "steer_accel_brake"` 일 때 (controller_imitator 호환) |
| `target_accel` | `(B, 30, 1)` | float32 | `target == "accel"` 일 때. m/s², 정규화 없음 |
| `target_steer_accel` | `(B, 30, 2)` | float32 | `target == "steer_accel"` 일 때 (**기본**). `[steer/40, accel(m/s²)]` |
| `target_ego` | `(B, 30, 4)` | float32 | `output_shape == "control_ego"` 일 때만. controller_imitator 의 MPC ego target 자리 |
| `valid_mask` | `(B, 30)` | bool | jiwon_e2e `wp_valid` 그대로 |
| `meta` | dict | — | frame_id 등. 모델에 들어가지 않음 |

학습 루프 측에서 `preprocess_data_for_preload(batch)` 가
`trajectory_points_local` (+ 옵션 `trajectory_time`) 을 합쳐 `(B, 30, F_traj)` 텐서를 만들고
`model(traj, ego, image_curr=..., image_prev_rot=...)` 로 전달.

---

## 7. 손실 / 학습 루프

- controller_imitator 의 `src/losses.py::calculate_losses` 와 `target × output_shape` 분기 그대로 재사용 + `accel` / `steer_accel` 분기 추가.
- 손실 scale 컨벤션 (v3 코드, divide style — `L1(pred/scale, target/scale)`):
  - `scale_steer = 10.0` (steer deg, controller_imitator 의 `loss_weight_steer=40` 와 실효 동등)
  - `scale_ax    = 10.0` (ax m/s², raw — divide 로 균형)
  - obj_gate 학습 시: `loss = L1_steer + L1_ax + obj_loss_w × BCE(obj_logit, obj_target)`
    - `obj_loss_w = 0.5` 기본
- **(옵션) ImageFusionBlock 보조 supervision** — 모두 baseline 비활성:
  - **(a) SegHead loss** — `image_fusion_seg_head=True` 일 때만:
    - `L_seg = CrossEntropy(seg_logits (B, K, 8, 40), seg_gt (B, 8, 40))`
    - `image_fusion_seg_apply_to = both` 이면 curr + prev 각각 손실 평균
    - GT: SAM2 mask → AdaptiveMaxPool2d to (8, 40) cell mode. jiwon_e2e `gt_pipeline.py` 재활용
  - **(b) AuxHead loss** — `image_fusion_aux_head=True` 일 때만:
    - `L_aux = w_obj·BCE(obj) + w_vlat·L1(v_lat) + w_vlong·L1(v_long) + w_prx·L1(prx)`
    - GT: OBJECT_INFO → 카메라 평면 lateral bin (40 칸)
  - 총 손실: `L_total = L_controller_imitator + image_fusion_seg_weight · L_seg + image_fusion_aux_weight · L_aux`
- 학습 스크립트는 `train_multi_output_split_loss_visualization_renew.py` 를 fork:
  1. dataset import → SCC dataset
  2. `preprocess_data_for_preload` 에 image kwarg 전달 로직 추가
  3. `model(traj, ego, image_curr=..., image_prev_rot=...)` 호출
- optimizer / scheduler / wandb / csv 자동화는 동일.

---

## 8. 파일 구조 (확정안)

```
scc_controller_imitator/
├── src/
│   ├── datamodule/
│   │   ├── scc_dataset.py            # raw image + raw EGO + future traj 빌더
│   │   └── image_warp.py             # Δyaw 회전 homography
│   ├── model/
│   │   ├── models.py                 # get_model(args)
│   │   ├── image_encoders/
│   │   │   ├── __init__.py
│   │   │   ├── lateral_strip_encoder.py   # 단일 이미지 → (40, D) 토큰 (siamese)
│   │   │   └── image_fusion_block.py      # (curr, prev) → image_memory (40, D)
│   │   │                                  #   cross-attn(Q=curr, KV=prev)
│   │   │                                  #   + 옵션 self-attn + 옵션 aux head
│   │   ├── traj_encoders/
│   │   │   ├── __init__.py
│   │   │   └── traj_transformer.py        # controller_imitator SelfAttn 패턴
│   │   ├── fusion/
│   │   │   ├── __init__.py
│   │   │   └── cross_attn_fusion.py       # Q=traj, KV=image+prev
│   │   ├── temporal/
│   │   │   ├── __init__.py
│   │   │   └── (필요 시 GRU 헤드 — cross-attn 출력이 이미 (B, 30, D) 라 별도 헤드 불필요)
│   │   └── scc_predictors/
│   │       ├── __init__.py
│   │       └── two_img_traj_ego.py        # TwoImgTrajEgoController
│   └── utils/
│       └── camera.py                      # CameraIntrinsics + ROI scaling
├── scripts/
│   ├── preprocess_raw_to_scc_pkl.py       # (옵션) 얇은 SCC pkl 캐시 빌더
│   └── sanity_check_env.py                # 더미 forward (4-input)
├── config/
│   └── scc_config.ini                     # controller_imitator 키 + 이미지 키 추가
├── outputs_multi/                         # 학습 산출물
├── training_status/                       # CSV 자동화
├── requirements.txt
└── environment.yml
```

**기존 스텁 처리** (현재 디렉터리에 이미 있는 파일):
- `src/model/image_models/image_scc_models.py` — `ImageEgoSCCPredictor` / `ImageTrajEgoSCCPredictor`
  → `target=accel_brake` 가정으로 작성됨. **구현 시 삭제** 하고 `scc_predictors/two_img_traj_ego.py` 로 대체.
- `src/datamodule/scc_dataset.py` — 단일 이미지 + .pkl 기반. **구현 시 전면 재작성** (raw EGO + 두 이미지 + jiwon_e2e index 연동).
- `config/scc_config.ini` — `target=accel_brake` 고정. **구현 시 `target=steer_accel` 기본값으로 변경** (가속도 명령 m/s²). 페달 컨벤션(`accel_brake` / `steer_accel_brake`) 도 옵션으로 유지.

---

## 9. 확정된 의사결정

| 항목 | 결정 | 근거 |
|------|------|------|
| **target / output_shape** | controller_imitator 의 모든 옵션 + 신규 `accel` / `steer_accel` 옵션 공존. **기본: `steer_accel` + `control`** | 사용자 확정 — 종방향을 페달 대신 가속도 명령(m/s²) 로 학습. 페달 컨벤션은 호환용으로 유지 |
| **종방향 명령 컨벤션** | 두 컨벤션 공존: (a) `accel_brake` 페달 ∈ [-1, +1] (controller_imitator 호환), (b) `accel` 가속도 명령 m/s² (raw, 정규화 없음) | (a) 는 기존 데이터·코드 호환용. (b) 는 실제 차량 ax 와 단위 일치 → 실차 이식·MPC 비교 용이 |
| **가속도 명령 GT 소스** | 1차: **MPC 솔브 결과** (사전 산출 캐시). fallback: **EGO_INFO 의 ax** (측정값). 부호 그대로, m/s², vehicle frame | MPC 가 ground-truth-like 명령. 캐시 미보유 run 은 측정 ax 로 대체 (실차 응답 ≈ 명령 가정) |
| **가속도 정규화** | **없음** (raw m/s²). 손실 균형은 `loss_weight_accel` 로만 조정 | 단순·해석 용이. 정규화 스케일 hyperparam 제거 |
| **ego_info 채널** | `[vx, vy, ax, ay, yaw_rate]` (5-d, vehicle frame, SI) | 사용자 확정. yaw/pitch/roll 제거, acceleration 추가 |
| **trajectory 입력 포맷** | controller_imitator 와 동일 (`trajectory_points_local` (30, 2 or 3) + `trajectory_time` (30, 1) 옵션) | 사용자 확정. 전처리·학습 코드 재사용 |
| **trajectory horizon T** | **30** (jiwon_e2e `wp_rel_gt` 기준) | jiwon_e2e 데이터 직접 결합. 비교 ablation 시 controller_imitator 도 30 으로 재학습 |
| **trajectory 채널 변환** | `wp_rel_gt (Δx,Δy,Δyaw,v)` → 누적 → `(x, y, v)` 3-d (또는 `include_speed_profile=False` 면 2-d) | controller_imitator 컨벤션과 정렬 |
| **이미지 ROI** | 1280×720 → y ∈ [235, 485) → **250×1280** 고정 strip | 사용자 확정 (pitch 변동 작음) |
| **lateral 토큰 수 `N_lat`** | **40** (1280 / 32, 지원 백본 모두 stride 32 라 자연스럽게 떨어짐) | controller_imitator trajectory 와 길이 매칭은 안 되지만, cross-attention 에서는 무관 |
| **이미지 백본** | **configurable** (`img_backbone` config 키). 기본값 `efficientnet_b0`. 지원: `resnet18` / `efficientnet_b0` / `efficientnet_b1` / `tf_efficientnetv2_s` / `tf_efficientnetv2_b0` / `tf_efficientnetv2_b1` / (옵션 `mobilenetv3_large_100`). 모두 ImageNet pretrained, **siamese 공유 가중치** | 사용자 확정. jiwon_e2e 컨벤션 일치 + 백본 ablation 지원. timm `features_only=True` + 1×1 Conv1d 정규화로 swap |
| **Δyaw 산식** | `wz_t · Δt` (rad/s × s) | jiwon_e2e ego_pose 의 wz 직접 사용 |
| **Δyaw warp 처리 시점** | dataset `__getitem__` on-the-fly | 디스크 절감 |
| **prev 프레임 부재 처리** | 학습 데이터에서 제외 (`valid_mask` 0) | 학습 안정성 |
| **카메라 intrinsic 출처** | jiwon_e2e `calibration/CAMERA_1__intrinsic.npy` 직접 로드 | 단일 시뮬레이터 데이터셋, intrinsic 공유 |
| **데이터 소스 (1차)** | `sangam_track/MultiAgnet/*` (6,877 frame, OBJECT_INFO 채워짐) | 선행차 학습 신호 확보 |
| **데이터 소스 (확장)** | `sangam_track/MultiAgentSCC/*` (raw, 미인덱싱) | 추후 자체 인덱서로 추가 |
| **이미지 융합 (image-side)** | `ImageFusionBlock` (4 단계): (0) **per-frame SelfAttn (shared weights, 기본 1 layer)** — 동일 객체에 속한 인접 토큰 응집 / (1) **CrossAttn(Q=curr, KV=[ego_motion_token, prev])** — 객체 단위 매칭 + ego prior / (2) (옵션) post-fusion SelfAttn over 40 / (3) (옵션) AuxHead 4-d. 출력: **(B, 40, D)** | 거리 ≤10 m 차량이 3~6 lateral 토큰을 span 하므로 cross-attn 매칭 전 intra-frame 응집 필수. 토큰 i 가 lateral 위치 i 의 객체 상태 (존재/속도/proximity) 를 의미하도록 anchor |
| **Step 1 ego prior 주입** | `[vx, wz]` 2채널 → `Linear(2 → D)` → `ego_motion_token (B, 1, D)` → Step 1 K/V 에 prepend (`(B, 41, D)`). `frame_type_emb` 3-way (Q=0, KV_prev=1, KV_ego=2). image_memory 출력은 40 토큰만 (ego token 제외) | Δyaw warp 가 회전만 보정 → forward zoom-in (vx) 과 warp residual (wz) 이 apparent motion 에 남음. cross-attn 매칭 단계에서 prior 로 작동 → apparent motion ↔ 객체 실제 motion 분해. ego 5채널 전체가 아닌 2채널만 → 노이즈 (ax, ay 측정 잡음) 와의 결합 최소화. 나머지 3채널은 trajectory 단계 ego_mlp 가 담당 |
| **trajectory–image 융합** | Cross-attention (Q=trajectory tokens, KV=**image_memory** (40, D)) | controller_imitator `CrossAttentionControlPredictor` 와 동치. K/V 가 80 → 40 으로 줄어 비용·메모리 절감 |
| **ego (full 5-d) 주입 위치** | cross-attn **이전**, `traj_tokens[:, 0, :] += ego_emb` (t=0 토큰 1 개에만). 끝쪽 broadcast add 제거 | (1) cross-attn 의 Q 가 ego-conditioned 되어 image_memory attend 패턴이 자차 상태에 맞춰 변함 (예: 고속에서 더 먼 lateral 토큰에 attend). (2) self-attn 으로 ego 정보가 나머지 29 토큰에 자연 전파. (3) broadcast add 대비 위치 차별성 유지 (t=0 이 ego 와 가장 직접 연관된 시점이라 합리적 앵커) |
| **token-level SegHead** (보조 supervision (a)) | **baseline 비활성** (`image_fusion_seg_head=False`). Step 0 출력에 분기, 8×40 cell × 5 class. GT: SAM2 mask → AdaptiveMaxPool2d to (8, 40) cell mode. jiwon_e2e `gt_pipeline.py` 재활용 | 거의 무료 (~20K param). Step 0 self-attn 이 객체 경계를 학습하도록 직접 supervision. baseline 학습 후 ablation 으로 켬 |
| **per-lateral AuxHead** (보조 supervision (b)) | **baseline 비활성** (`image_fusion_aux_head=False`). image_memory 분기, [obj, v_lat, v_long, prx] 회귀. GT: OBJECT_INFO → 카메라 평면 투영 | implicit 학습으로 시작. SegHead 효과 검증 후 추가 |
| **steer 라벨 변환** | `target_steer = steer_deg / 40.0` (1.0 = 40°) | controller_imitator `loss_weight_steer=40` 컨벤션 유지 |
| **accel/brake 라벨 변환** | `target_accel_brake = accel - brake ∈ [-1, +1]` | controller_imitator 컨벤션 (1-d 통합). `target == accel_brake | steer_accel_brake` 일 때만 사용 |
| **accel 라벨 변환** | `target_accel = a_cmd (m/s²)` — raw, 부호 유지. 소스 우선순위: MPC 캐시 → EGO_INFO.ax | 단위·부호를 실 차량 ax 와 일치시켜 학습/평가/실차 이식이 동일 스케일에서 이루어짐 |

---

## 10. config 추가 키 (`scc_config.ini` 예시)

```ini
# ---- 이미지 ----
image_input_height = 250
image_input_width  = 1280
image_roi_y_top    = 235
image_roi_y_bottom = 485
img_backbone       = efficientnet_b0   ; resnet18 | efficientnet_b0 | efficientnet_b1
                                       ; | tf_efficientnetv2_s | tf_efficientnetv2_b0
                                       ; | tf_efficientnetv2_b1 | mobilenetv3_large_100
img_pretrained     = True
img_feature_dim    = 256               ; 백본 출력 채널 C_b 를 1×1 Conv1d 로 D 로 통일

# ---- ImageFusionBlock (4 단계 + 보조 heads) ----
image_fusion_pre_self_attn_layers  = 1   ; Step 0. per-frame self-attn (shared, 0=off)
image_fusion_cross_attn_layers     = 1   ; Step 1. cross-attn Q=curr, KV=prev
image_fusion_post_self_attn_layers = 1   ; Step 2. post self-attn over 40 (0=off)

; (a) token-level SegHead — Step 0 분기, 8×40 cell, ~5 class
image_fusion_seg_head              = False  ; baseline 비활성, GT 캐시 완성 후 ablation
image_fusion_seg_classes           = 5      ; void/road/vehicle/lane_line/wall
image_fusion_seg_weight            = 0.5
image_fusion_seg_apply_to          = curr   ; curr | both

; (b) per-lateral 4-d AuxHead — image_memory 분기
image_fusion_aux_head              = False  ; baseline 비활성
image_fusion_aux_weight            = 0.1

image_fusion_nhead                 = 8
image_fusion_dim_feedforward       = 512
image_fusion_dropout               = 0.1
image_frame_type_emb               = True   ; Step 1 에서만 사용
image_lateral_pos_emb              = True   ; Step 0, 1, 2 공통
img_lateral_tokens = 40

# ---- 카메라 (calibration npy 에서 자동 로드, 아래는 fallback) ----
camera_calib_dir = ../jiwon_e2e/e2e_racing/calibration   ; 또는 자체 복사 경로
camera_index     = 1                                     ; CAMERA_1 = front

# ---- trajectory (controller_imitator 호환, 길이만 30) ----
include_speed_profile = True
input_time_to_traj    = rel_time
input_trajectory      = relative_trajectory
input_pred_idx_list   = 0,3,7,11,15,19,24,29
output_pred_idx_list  = 0,3,7,11,15,19,24,29
tfr_seq_length        = 30
tfr_state_dim         = 5            ; [vx, vy, ax, ay, yaw_rate]
tfr_d_model           = 256
tfr_nhead             = 8
tfr_dim_feedforward   = 512
tfr_dropout           = 0.1

# ---- 학습 타겟 ----
target       = steer_accel          ; steer | accel_brake | steer_accel_brake | accel | steer_accel
                                    ;   accel_brake / steer_accel_brake : controller_imitator 호환 (페달 ∈ [-1,1])
                                    ;   accel / steer_accel             : 가속도 명령 (m/s², raw)
                                    ;   기본 = steer_accel
output_shape = control              ; control | control_ego
loss_weight_steer       = 40.0      ; 1.0 = 40°
loss_weight_accel_brake = 1.0       ; 페달 신호용
loss_weight_accel       = 1.0       ; 가속도 명령(m/s²)용. raw 단위라 weight 로만 균형

# ---- 가속도 명령 GT 소스 ----
accel_gt_source     = mpc_then_ax   ; mpc | ax | mpc_then_ax  (mpc 캐시 우선, 없으면 ax 폴백)
accel_mpc_cache_dir = data/processed/mpc_accel   ; run 별 a_cmd 캐시 (frame_id → m/s²)

# ---- 데이터 ----
data_index_path = ../jiwon_e2e/e2e_racing/data/processed/index.json
data_splits_path = ../jiwon_e2e/e2e_racing/data/processed/splits.json
data_raw_root    = ../jiwon_e2e/e2e_racing/data/raw
policies_whitelist = MultiAgnet, MultiAgentSCC
```

---

## 11. 구현 진입 순서 (구현 시작 시)

1. `src/utils/camera.py` — CameraIntrinsics + ROI scaling
2. `src/datamodule/image_warp.py` — pure-yaw warp + 단위 테스트
3. `src/datamodule/scc_dataset.py` — raw EGO + 두 이미지 + future_traj 빌더 (jiwon_e2e 모듈 직접 import 또는 복사). target 분기 시 `target_accel` 은 `accel_gt_source` 정책에 따라 MPC 캐시 / EGO_INFO.ax 중 선택
3.5. (선택) `scripts/build_mpc_accel_cache.py` — run 별 MPC 솔브 결과를 `accel_mpc_cache_dir` 에 frame_id → `a_cmd (m/s²)` 로 저장. 미구현 시 자동으로 ax 폴백
4. `src/model/image_encoders/lateral_strip_encoder.py` — timm 백본 (configurable) → `(spatial_feat (B, C_b, 8, 40), tokens (B, 40, D))` 튜플 반환 (SegHead 용 spatial_feat 노출)
5. `src/model/image_encoders/image_fusion_block.py` — 4 단계 (pre-SA → cross-attn → post-SA) → image_memory (40, D). + 옵션 SegHead (Step 0 분기) + 옵션 AuxHead (image_memory 분기)
6. `src/model/traj_encoders/traj_transformer.py` — controller_imitator 모듈 차용
7. `src/model/fusion/cross_attn_fusion.py` — Q=traj, KV=image_memory
8. `src/model/scc_predictors/two_img_traj_ego.py` — 최상위 모델
9. `src/model/models.py::get_model` 등록 (`model_name = TwoImgTrajEgoController`)
10. `scripts/sanity_check_env.py` 갱신 (4-input, target=steer_accel_brake, image_memory shape 검증)
11. `controller_imitator/train_multi_output_split_loss_visualization_renew.py` fork → image kwarg 전달
12. `MultiAgentSCC` 인덱싱 (`scripts/index_multiagent_scc.py`)
13. CSV 자동화 연결 (`ML_Experiment_Table.csv` 컬럼 추가)
14. (후속 1) SAM2 mask cache → 8×40 cell mode 변환 → SegHead 활성화 ablation
15. (후속 2) OBJECT_INFO → 카메라 평면 lateral-bin GT 빌더 → AuxHead 활성화 ablation

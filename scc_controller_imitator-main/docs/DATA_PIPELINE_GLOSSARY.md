# 데이터 파이프라인 용어 사전 (KR)

학습용 데이터를 만드는 stage 와 각 산출물의 의미. 파일 위치와 사용 시점 포함.

---

## 0. 전체 파이프라인 한눈에

```
[raw data]                                       원본 (외부, NFS)
    ↓
[1. seg pretrain]      → seg_backbone.pt        (image encoder 학습)
    ↓
[2. i_1 cache]         → i1_tok_cache_*.npz    (image → token per-frame)
    ↓
[3. aux_targets]       → aux_targets_*/         (a_2 학습 GT)
    ↓
[4. a_2 학습]          → a2_*.pt               (per-bin object features 학습)
    ↓
[5. a_2 cache]         → a2_out_cache_*.npz    (a_2 의 (40,7) features per-frame)
    ↓
[6. SCC 합성 / logged] → scc_logged_bank_*/    (sample 단위 binary)
    ↓
[7. prebake]           → f1_prebake_*.npz     (학습용 dict 통합 npz)
    ↓
[8. f_1 / f_2 학습]    → iy_*.pt              (최종 모델)
    ↓
[9. closed_loop ROS]                            (실차/시뮬 inference)
```

---

## 1. raw data — 원본 데이터

**위치**: `/home/ailab-12/AILabDataset/02_Custom_Dataset/57_A1_E2E/00_raw/` (NFS, read-only)

**구조**:
```
sangam_track/                       ← track 이름
  DriverB_new/                      ← policy 이름 (driver / simulator)
    260513_100/                     ← run ID (날짜+seq)
      CAMERA_1/                     ← 5 camera (학습은 CAMERA_1 만)
        54180_Intensity.jpeg        ← tick 54180 (ms 단위) 의 image
        54185_Intensity.jpeg
      EGO_INFO/
        54180.txt                   ← ego 의 [position, velocity, accel, steer, brake] (5ms step)
        54185.txt
      OBJECT_INFO/                  ← 다른 차량 (lead car) state
        54180.txt
```

**핵심**:
- 1 tick = 1 ms (file 이름)
- 5 tick = 5 ms step (default 간격)
- 100 tick = 100 ms = 1 inference dt (10 Hz)
- 한 run = 60-300 초 driving = 60k-300k tick

---

## 2. seg pretrain — image encoder 학습

**산출물**: `data/ckpts/seg_pretrain_*.pt`  
**용도**: image 를 segmentation 으로 학습 → 좋은 token 추출하는 backbone 만들기  
**모델**: `tf_efficientnetv2_s` (timm)  
**버전**:
- `combined_v1` (mIoU 0.892)
- `fx640_v3` (mIoU 0.921, fx=640 데이터 전용) ← 현재 사용

→ **이 backbone 의 출력 channel C_b = 256** (raw feature). 학습 종료 후 cache 만들기 (다음 step).

---

## 3. i_1 cache — image token per-frame

**스크립트**: `scripts/compute_i1_tok_cache.py`  
**산출물**: `data/cache/i1_tok_cache_<name>_prev20.npz`  
**크기**: 19k frames × 40 × 256 = 약 770 MB / dataset

**무엇**:
- 각 frame 의 image 를 seg backbone 으로 forward → `(40, 256)` token 추출
- 학습 시 image 자체 load (수십 KB JPG decode) 안 함, cache 의 token 만 mmap → 학습 속도 10× ↑
- token = 가로 폭 40 strip 별 256-channel feature

**중요**:
- ⚠️ token 은 **proj 적용 전 raw backbone output** (C_b=256). 학습 코드의 model 안 proj (Conv1d) 통과시켜야 d_token 분포로 변환됨.
- 이 단계 누락 시 학습/inference output 무의미 (2026-05-24 build_a2_cache 버그 사례 참조).

**구조**:
```python
np.load(path) → {
  'tokens': (N_frame, 40, 256) float16,
  'keys':   (N_frame,) <U25 — 'run__t0054180' format
  'meta':   json {backbone, seg_ckpt, mIoU, prev_dtick}
}
```

`prev20` suffix = "prev_dtick = 20 ticks = 200ms 전 frame 도 함께 cache". `_v3` 등은 backbone 버전.

---

## 4. aux_targets — a_2 학습 GT

**위치**: `data/aux_targets_<name>/sangam_track__<policy>__<run>__t<tick>.npz`  
**산출물 1개당 size**: 약 500 byte (작음)

**무엇**:
- a_2 model 의 학습 GT
- frame 별 (40, 7) tensor: `[obj, vL_abs, vY_abs, vL_rel, vY_rel, prox, dist]`
- `obj` = 해당 bin 에 lead 가 있는지 (binary 0/1)
- `dist` = lead 까지의 거리 (m, /40 scaled)
- `vL/vY rel/abs` = lead 의 속도 (longitudinal/lateral, ego frame)

**생성**: raw OBJECT_INFO + camera projection 으로 만듬. (별도 script, 학습 전에 한 번 만들어 둠)

---

## 5. a_2 학습 — per-bin object detection head

**스크립트**: `scripts/train_a2_head.py`  
**산출물**: `data/ckpts/a2_fx640_v4_best_loss.pt`  
**구조**:
- input: image_curr + image_prev (i_1 cache 활용 가능) + ego_motion
- forward: encoder (backbone + proj) → fusion (cross-attn) → head (Linear)
- output: per-bin (40, 7) — aux_targets 와 같은 channels
- loss: BCE (obj) + L1 (dist/vrel)

**v4 metric**: Precision 70→98%, dist MAE 1.03m, vL_rel MAE 0.94 m/s.

---

## 6. a_2 cache — a_2 의 (40,7) features per-frame

**스크립트**: `scripts/build_a2_cache.py`  
**산출물**: `data/cache/a2_out_cache_<name>_prev20.npz`  
**크기**: 19k frames × 40 × 7 × 2 bytes (fp16) = 5 MB / dataset

**무엇**:
- a_2 model 의 inference 결과를 frame 별 미리 저장
- f_1/f_2 학습 시 a_2 inference 매번 안 함 → 학습 속도 ↑
- 학습 시 `a2_feat` 으로 transformer 에 inject

**⚠️ 빌드 시 주의** (2026-05-24 incident 참조):
- i_1 cache 의 token 은 proj 전. a_2.encoder.proj 통과시켜 fusion 에 inject 해야.
- 이 step 누락 시 fusion output 분포 mismatch → cache 값 무의미.

---

## 7. SCC 합성 / logged — 학습 sample 단위 데이터

3 가지 형태:

### 7.a SCC 합성 (synth)
**스크립트**: `scripts/run_synth_batch_shards.sh` + `scripts/run_synth_*.py`  
**산출물**: `data/scc_synth_bank_<name>/<run>__t<tick>__k<i>.bin`

**무엇**:
- MPC (HPIPM 또는 acados) 가 trajectory + ego state 로부터 control target (steer, ax) 합성
- K=100 augmentation — 같은 frame 에 다양한 perturbation 으로 100 sample 생성
- target = MPC 의 첫 step 명령 + future 39 step prediction (40개)

**시간**: 한 frame 당 K=100 sample 합성 ~2 초. 22 shard parallel.

### 7.b logged-only
**스크립트**: `scripts/prebake_iy_logged_only.py`  
**산출물**: 직접 prebake npz 생성 (bin 단계 없음)

**무엇**:
- MPC 합성 없이 raw EGO 의 future 40 frame 의 (steer, ax) 그대로 target
- 빠름 (raw 만 읽음), 다양성 자연스러움 (driver 실제 driving)
- ⚠️ raw EGO 가 5 tick step (5 ms) 이므로 100 tick (100 ms) future 는 file 20 단계 후

### 7.c brake_aug / smoothbrake
**무엇**: 합성 target 에 추가 brake floor 적용
- `brake_aug`: close-lead + 빠른 approach (dist<15, vrel<-3) 인 frame 의 target ax 를 -15 m/s² 로 floor
- `smoothbrake`: 더 부드러운 deceleration profile

→ class imbalance (실제 brake 적음) 보완.

---

## 8. prebake — 학습용 dict 통합 npz

**스크립트**: `scripts/prebake_iy_dataset.py` (또는 `prebake_iy_logged_only.py`)  
**산출물**: `data/prebake/f1_prebake_iy_<name>_prev20.npz`  
**크기**: 19k samples × (40 × 7 + ...) → 100-500 MB / dataset

**무엇**:
- 학습 1 sample 에 필요한 모든 dict 형태 데이터를 한 npz 로 합침
- 학습 시 mmap → DataLoader 가 random access

**구조**:
```python
np.load(path) → {
  'sample_keys':   (N,) — '260513_100__t0054180__logged' 형식 (unique sample ID)
  'sample_tags':   (N,) — 'logged' | 'synth' | 'aug' 등
  'curr_keys':     (N,) — '260513_100__t0054180' (i_1 cache 의 frame key)
  'prev_keys':     (N,) — prev frame key
  'd_yaws':        (N,) float32 — curr - prev ego yaw (rad)
  'sample_fx':     (N,) float32 — camera fx
  'ego_5d':        (N, 5) — [vx, vy, ax, ay, yaw_rate]
  'traj':          (N, 40, 3) — raceline ego frame [x, y, speed]
  'target':        (N, 40, 2) — [steer (deg), accel (m/s²)] 학습 GT
  'mask':          (N, 40) bool — future 의 EGO 존재 여부
  'obj_target':    (N, 40) — per-bin lead 존재 GT (있을 때만)
  'prev_dtick':    scalar — prev frame 간격 (20 = 200ms)
}
```

**왜 npz?**: dict + tensor 의 mmap 지원, 학습 시 random sample 즉시 access, pickle 보다 12× 작음.

---

## 9. ci_aug — controller_imitator augmentation

**위치**: `02_Custom_Dataset/35_e2e_morai/controller_imitator/controller_imitator_data_multi_valid_v*/`  
**무엇**:
- Professor #654 (controller_imitator) 학습용 데이터
- image 없음, 오직 traj+ego → control. waypoint following only (ACC ✗)
- f_1 (main_no_image=True) 학습 시 추가 sample 로 사용
- DataLoader 에서 image token = zeros 로 채워서 합침

**변환**:
- 기존: pkl (7 GB, deserialize 5 min)
- 변환 후: npz (598 MB, mmap 즉시) — 2026-05-24 변환 script 추가

---

## 10. close_lead_meta — close-lead sample weighting

**스크립트**: `scripts/build_close_lead_meta.py`  
**산출물**: `data/cache/close_lead_meta_v4.npz`

**무엇**:
- prebake 의 어떤 sample 이 close-lead (dist<30m) 인지 + min_dist 분포 미리 계산
- 학습 시 loss weighting 강화 (close-lead × 3.0)
- DataLoader sampler 의 difficulty-aware sampling 에 활용

---

## 11. 폴더 구조 한눈에

```
data/
├── ckpts/                              # 학습된 모델 + registry
│   ├── seg_pretrain_fx640_v3_*.pt      # backbone (step 2)
│   ├── a2_fx640_v4_best_loss.pt        # a_2 model (step 4)
│   ├── iy_v25c_best_loss.pt            # f_1 baseline (step 8)
│   └── CURRENT_BEST.json               # registry (어떤 ckpt 가 best)
├── aux_targets_<name>/                 # a_2 학습 GT (step 4)
├── cache/                              # frame-level cache
│   ├── i1_tok_cache_*.npz              # step 3
│   ├── a2_out_cache_*.npz              # step 6
│   └── close_lead_meta_*.npz           # step 10
├── prebake/                            # 학습 sample dict (step 8)
│   └── f1_prebake_iy_*.npz
├── scc_logged_bank_<name>/             # logged sample bin (step 7.b 의 중간)
├── scc_synth_bank_<name>/              # MPC 합성 sample bin (step 7.a 의 중간)
└── scc_frame_list_*.txt                # frame list (어떤 run+tick 사용)
```

---

## 12. naming convention (suffix 의미)

| suffix | 의미 |
|---|---|
| `_prev20` | prev_dtick = 20 ticks = 200ms 전 frame (현재 표준) |
| `_prev5` | prev_dtick = 5 ticks = 50ms (legacy) |
| `_v1`, `_v2`, `_v3`, `_v4` | data 버전 (재합성 등) |
| `_fixed` | bug fix 후 재생성 (예: 2026-05-24 a_2 cache) |
| `_brakeaug` | brake augmentation 적용 |
| `_smoothbrake` | smoother brake floor |
| `_clipped` | target clip 적용 (steer ±30°, ax ±20 m/s²) |
| `_logged` | MPC 합성 없이 raw driver action 그대로 |
| `_pertO` | perturbed O (with-lead 의 perturbation) |
| `_pertX` | perturbed X (without-lead 의 perturbation) |
| `combined_v1` | seg pretrain 의 sangam+speedway 통합 backbone |
| `fx640_v3` | seg pretrain v3 (fx=640 only data) |

---

## 13. 학습 시 cache 모두 어디 들어가나

`scripts/train_iy.py` 의 SCCSynthIyDataset 가 다음을 합쳐서 dict 출력:

```python
sample_i = {
  'tok_curr':        i_1 cache 의 token [curr_key]      # (40, 256)
  'tok_prev_raw':    i_1 cache 의 token [prev_key]      # (40, 256)
  'd_yaw':           prebake.d_yaws[i]                   # scalar
  'fx':              prebake.sample_fx[i]                # scalar
  'ego_5d':          prebake.ego_5d[i]                   # (5,)
  'traj_40':         prebake.traj[i]                     # (40, 3)
  'target':          prebake.target[i]                   # (40, 2)
  'target_mask':     prebake.mask[i]                     # (40,) bool
  'obj_target':      prebake.obj_target[i]               # (40,)
  'min_dist':        close_lead_meta 계산값              # scalar (loss weighting 용)
  'inlane_dist':     a_2 cache 의 bin 16-25 in-lane lead dist  # scalar (TTC augment 용)
  'inlane_closing':  prev frame 과 dist diff             # scalar
  'a2_feat':         a_2 cache [curr_key]                # (40, 7)  (use_a2_feat 시)
}
```

→ model forward 시 `tok_curr/tok_prev_raw` 가 i_1 cache token. `img_proj` (Conv1d C_b → d_model) 통과 후 transformer 에 들어감.

---

## 14. 새 데이터 추가 시 표준 순서

memory `pipeline_order` 참조:

1. **raw 확보** — 외부 (NFS) sync
2. **seg pretrain 추가 학습** (필요 시) — 신규 데이터 일부 + 기존 통합
3. **i_1 cache 생성** — 신규 데이터 (compute_i1_tok_cache.py)
4. **aux_targets 생성** — 신규 데이터 (별도 script)
5. **a_2 cache 생성** — i_1 cache + a_2 ckpt → build_a2_cache.py
6. **scc_logged_bank** (logged-only) 또는 scc_synth_bank (MPC 합성) — 신규 데이터
7. **prebake 생성** — prebake_iy_dataset.py 또는 prebake_iy_logged_only.py
8. **close_lead_meta** (선택) — 신규 prebake 용
9. **학습 launch** — train_iy.py + prebake/cache path 지정

---

## 자주 헷갈리는 것

- **prebake vs cache**: prebake = 학습 sample 단위 (sample_keys 가 unique sample, 같은 frame 의 K=100 augment 가 다른 sample), cache = frame 단위 (한 frame 의 image 처리 결과, K 와 무관하게 1번 계산)
- **logged vs synth**: logged = raw driver action, synth = MPC 합성. logged 가 빠르지만 다양성 적음
- **aux_targets vs a_2 cache**: aux_targets = a_2 학습 GT (외부에서 만든 ground truth), a_2 cache = 학습된 a_2 의 inference 결과 (model output)
- **prev_dtick**: prev frame 이 얼마나 전 frame 인지. 20 = 200ms = 2 inference dt
- **K**: synth 시 같은 frame 의 augment 개수. 학습 sample 수 = N_frame × K
- **fx**: camera focal length. fx=640 (현재 표준) vs fx=320 (legacy MultiAgnet)

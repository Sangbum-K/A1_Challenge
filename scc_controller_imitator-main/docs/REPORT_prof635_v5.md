# Prof635 v5 — SCC Controller Imitator 모델 구조 / 학습 전략 리포트

작성 2026-05-25. 이 문서만으로 모델 구조·입출력·학습·배포를 재현하는 것을 목표로 한다.
대상 코드: `src/model/image_models/prof635_side_model.py`, `src/datamodule/prof635_dataset.py`,
`scripts/train_prof635_side.py`, `src/utils/{prof635_infer,raceline_search,pedal_label}.py`.

---

## 1. 개요

SCC(Smart Cruise Control) 종/횡 제어 imitation. 두 모듈로 분리한다.

| 모듈 | 입력 | 출력 | 학습 |
|---|---|---|---|
| f_1 (main) | trajectory(경로) + ego state | steer + pedal(base) | **frozen** (controller_imitator #654 그대로) |
| f_2 (side) | image(i2_mem) + ego + f_1 pedal | pedal **변화량(delta)** | **학습 대상** |

- f_1 은 경로와 자기상태만 보고 조향·기본 가속을 낸다. 이미지를 보지 않는다.
- f_2 는 카메라 인지(i2_mem)와 자차 상태, f_1 의 pedal 값을 보고 **앞차가 있을 때** pedal 을 보정한다.
- 최종 pedal = `base_pedal + gate * delta` (gate = a_2 in-lane 객체 인지). 앞차 없으면 gate≈0 → f_1 그대로.

설계 동기(검증된 사실): f_1(#654)의 brake 는 입력 trajectory 의 speed 채널에 박힌 감속을 따라간 것일 뿐
이미지를 보지 않으므로 inference(raceline nominal speed)에서는 앞차에 brake 하지 못한다. 따라서
앞차 감속은 전적으로 f_2(image)가 담당해야 한다. (근거: 4절 ablation)

---

## 2. 전체 파이프라인

```
camera(curr, prev) --+--> seg backbone(fx640_v3) --> adaptive_avg_pool2d(1,40) --> tokens(40, C_b=256)
                     |                                            |
                     |                                  yaw_shift_tokens_batch(prev, d_yaw, fx)
                     v                                            v
                a_2 v7b: encoder.proj --> fusion(tc, tp, ego_mot=ZEROS) --> fc_proj
                     |                                                          |
                     +--> head(7ch) = a2_logits (40,7)               i2_mem (40,128)
                                  |                                            |
raceline + ego pose --> traj6(6,3, ch2=speed)                                  |
ego state --> ego6(6)                                                          |
                     \                |                |                       /
                      v               v                v                      v
                   Prof635SideModel:  f_1(traj6,ego6)->steer,base_pedal ; f_2(ego6,base_pedal,i2_mem)->delta
                                  gate g = sigmoid(max(a2_logits[16:26,0]) + gate_bias)
                                  final_pedal = clamp(base_pedal + g*delta, -1, 1)
                     v
            steer(deg) = steer_n * 40 ;  pedal -> (accel=max(p,0), brake=max(-p,0))
```

핵심 불변식: i2_mem 은 학습 cache 와 동일하게 계산해야 한다 — seg=fx640_v3, ego_mot=zeros. (6절)

---

## 3. 입력 규격 (재현용 정확값)

상수: `WB = 2.35` (wheelbase), `STEER_NORM = 40.0`, `T6 = [0.05, 0.30, 0.55, 0.80, 1.05, 1.30]` s.

### 3.1 traj6 — (6, 3)
prebake traj_40(0.1s 간격, ego-frame `[x_fwd, y_left, speed]`)를 T6 시점에 선형보간. 모델은 0.05s 간격으로
학습되었으나 데이터는 0.1s → 반드시 보간한다.
- col0 = `-y_left`  (lateral, right-hand 재정렬)
- col1 = `+x_fwd`   (longitudinal)
- col2 = `speed`    (mode='speed', #654. abs_time 아님)

inference 시 raceline 에서: `sample_traj(raceline, ex, ey, e_yaw, cutoff, polygon, mode='speed')`.
speed 는 raceline xy 간격으로 `||p[i+1]-p[i]|| / 0.1` 유도(`_derive_speed`).

### 3.2 ego6 — (6,)
`[vx, vy, ax, ay, tire_angle, yaw_rate]`. tire_angle 은 EGO_INFO steering 있으면 그 값, 없으면 kinematic:
`tire = atan(WB * yaw_rate / vx)` (단 `|vx| >= 5`, 아니면 0).

### 3.3 i2_mem — (40, 128)
a_2 v7b 의 fc_proj 출력(per-column image memory). 생성 절차는 6절. fusion 의 ego_mot 은 **zeros(B,2)**.

### 3.4 a2_logits — (40, 7)
`a2.head(i2_mem)`. gate 는 in-lane bin `[16:26]` 의 채널0(obj) 최대값 사용.

---

## 4. 모델 구조

### 4.1 f_1 = Prof635Main (#654, frozen)

| 항목 | 값 |
|---|---|
| traj_embed | Linear(3, 128) |
| state_embed | Linear(6, 128) |
| token | `traj_embed(traj) + state_embed(state).unsqueeze(1)` (state broadcast) |
| pos encoder | sinusoidal, max_len 41 |
| encoder | TransformerEncoder, d_model 128, nhead 8, dim_ff 512, num_layers 6, dropout 0.1, batch_first |
| fc_out | Linear(128, 2) → `[steer/40, accel_brake]` |
| 사용 | `out = main(traj6, ego6)[:, 0, :]` (pred_first_only) |

로드: `load_main_frozen(ckpt)` — `pos_encoder.pe` pop 후 strict=False 로드, 전 파라미터 freeze.
ckpt = `data/ckpts/main_prof654.pth` (controller_imitator #654, config interp005_relspd_notime, channel2=speed).

### 4.2 f_2 = Prof635SideModel side branch (학습 대상)

| 항목 | 값 |
|---|---|
| side_state_embed | Linear(6, 128) |
| side_pedal_embed | Linear(1, 128)  (f_1 base_pedal conditioning) |
| side_traj_embed | Linear(3, 128)  — **use_traj_in_side=False 이면 미사용 (v5 기본)** |
| query q | `side_state_embed(ego) + side_pedal_embed(base_pedal.detach())` → (B, 1, 128) |
| img_proj | Linear(128, 128) → KV = img_proj(i2_mem) (B,40,128) |
| side_dec | TransformerDecoder, d 128, nhead 8, dim_ff 512, num_layers 4, dropout 0.1, batch_first |
| pedal_head | Linear(128, 1), **zero-init** (warm start: delta≈0 → final≈base) |
| gate_bias | Parameter(0) |

trainable ≈ 1.08M (main 은 frozen).

### 4.3 forward (gated residual)

```
out_main   = main(traj6, ego6)[:, 0, :]          # frozen
steer      = out_main[:, 0]                       # f_1 조향
base_pedal = out_main[:, 1]                       # f_1 기본 pedal
q          = side_state_embed(ego6) + side_pedal_embed(base_pedal.detach())   # (B,1,128), traj 미포함
side       = side_dec(q, img_proj(i2_mem))        # cross-attn KV=image memory
head       = pedal_head(side[:, 0, :])
delta      = 2 * tanh(head)                        # residual 보정, bound +-2
g          = sigmoid(max(a2_logits[:, 16:26, 0], dim=1) + gate_bias)   # 앞차 gate
final_pedal= clamp(base_pedal + g * delta, -1, 1)  # = (1-g)*base + g*(base+delta)
```

수식 주의: `final = base + g*delta`, **`base + g*f2` 아님**. 로그의 f2 = base+delta(절대 pedal).
g→1(앞차): final→base+delta. g→0(앞차 없음): final→base(f_1 그대로).

### 4.4 출력
- steer(deg) = `steer * STEER_NORM(40)`.
- pedal(-1~1) → `accel_pedal = max(pedal, 0)`, `brake_pedal = max(-pedal, 0)`. (단일 pedal scalar)

---

## 5. 학습 전략

스크립트: `scripts/train_prof635_side.py`. 모델 default `residual_gate=True, use_traj_in_side=False`.

### 5.1 데이터셋

| 이름 | 샘플수 | prebake | i2_cache | pedal_label |
|---|---|---|---|---|
| sangam_lead_0520 | 72,519 | `f1_prebake_iy_..._v1_brakeaug_prev20` | `i2_mem_cache_..._v7b_prev20` | `pedal_label_..._ttc` |
| sangam_lead_0521 | 97,753 | 〃 | 〃 | 〃 |
| sangam_pertO_v2 | 338,073 | `..._prev20_fixed` | 〃 | 〃 |
| sangam_pertX_v2 | 332,413 | 〃 | 〃 | 〃 |
| (옵션) DriverB_new | ~135,113 | `..._logged_fixed_prev20_trajfix` | `..._v7b_prev20` | `..._ttc` |

total 840,758 (DriverB 제외), val 5% (random_split seed 0). DriverB 는 객체 분포가 좋아 합류 권장
(pedal_label_ttc 생성 완료, lead-step0 brake 76%).

### 5.2 pedal label (핵심)

`pedal_label_{ds}_ttc.npz` 생성: `scripts/gen_ttc_pedal_labels.py`.
1. raw target_ax 에 **ACC-style ttc-augment** 적용(in-lane lead 접근 시 brake floor):
   `safe_decel = max(a_kin, a_gap, a_emerg)`, `aug_ax = min(raw_ax, -safe_decel)` (첫 KSTEPS step).
   params: `D0=5, HEADWAY=1, KGAP=0.2, KCLO=0.6, TTC_EMERG=2.0, DECAY=5.0, MAXB=20, MARGIN=2.0, KSTEPS=10`.
   in-lane lead 여부는 GT inlane cache(`inlane_gt_{ds}.npz`, `build_inlane_gt_cache.py`)로 판정.
2. `pedal_map(speed, aug_ax)` → 단일 pedal scalar (`src/utils/pedal_label.py`,
   quantize qv=0.05 qa=0.02 → unique → pedal_map → map-back). 규칙: brake>0 이면 `-brake`, 아니면 `+accel`.

주의: ttc-augment 안 한 raw ax 기반 label 은 앞차 접근 시 가속(+) 라벨이 나오는 버그가 있었음 → 반드시 `_ttc` 사용.
검증: sangam_lead lead-TTC label mean −0.364, brake 100%, accel 0%.

### 5.3 손실 / 하이퍼파라미터

| 항목 | 값 |
|---|---|
| loss | `L1(final_pedal, pedal_label)` (mask0=True 인 step0 만) |
| optimizer | Adam, lr 3e-4 |
| epochs | 25 |
| batch | 512 |
| side_layers | 4 |
| main | frozen (steer/base_pedal 고정) |
| init | pedal_head zero-init → 시작 시 final≈base (#654 warm start) |

v5 결과: **best val_pedal = 0.0237**, steer MAE 3.50° (frozen #654, val). 
(참고: v3 replace+traj 0.0127 이 더 낮았으나 traj-speed shortcut 의존 → v5 가 shortcut 없는 정직한 구조.)

### 5.4 학습 순서 (신규 데이터 추가 시)
1. seg pretrain(fx640) → 2. a_2 head(i_1+i_2) → 3. i_2 cache 생성 → 4. f_2(side) 학습.
main(#654)은 frozen 이므로 재학습하지 않는다.

---

## 6. i_2 cache 생성 (배포 일치 불변식)

학습이 사용한 i2_mem cache 빌더: `scripts/build_i2_cache.py` (+ `compute_i1_tok_cache.py`).
- i_1 token: seg backbone = **`seg_pretrain_fx640_v3`** (mIoU 0.921) → `adaptive_avg_pool2d(1,40)`.
- a_2: `a2_v7b_d128_alpha075_best_loss.pt` (focal alpha 0.75, d_token 256, d_out 128, 7ch).
- fusion 의 **ego_mot = zeros(B,2)**.

closed-loop / 어떤 live inference 든 i2_mem 을 위와 **정확히 동일**하게 계산해야 한다. 불일치 시
모델이 학습 분포 밖의 perception 을 받아 SCC/제동이 붕괴한다.

검증(`scripts/verify_i2_live_vs_cache.py`, cached i2_mem 과 cosine):

| live 경로 | cos_sim | L1 |
|---|---|---|
| combined_v1 + ego_real (구 버그) | 0.804 | 0.244 |
| fx640_v3 + ego_real | 0.833 | 0.219 |
| combined_v1 + ego_zeros | 0.931 | 0.119 |
| **fx640_v3 + ego_zeros (정답)** | **1.000** | **0.001** |

미해결: cache 빌더의 ego_mot=0 이 의도인지 builder 버그인지 불명. real ego_mot 으로 가려면
cache + a_2 i2cache + f_2 재학습 필요(큰 작업). 현재는 ego_mot=0 유지 = 배포 모델 재현 기준 정확.

---

## 7. 핵심 검증 / 발견

- base(f_1) = traj-speed artifact: lead-TTC 에서 traj-speed 를 flat(감속 제거)하면 base −0.476 → +0.282(가속).
  f_2 는 −0.371 → −0.373 불변 → **f_2 는 image 로 brake** (transferable). (`scripts/ablate_traj_speed.py`)
- loss 정직성: lead-TTC L1 0.0090 < non-lead 0.0122. imbalance masking 아님. cached i2_mem 에서 f_2 brake 100%.
- grad_cam(`scripts/viz_prof635_gradcam.py`): v5 lead-TTC 12샘플 전부 brake(−0.26~−0.41), gate≈0.99.
  제동 트리거는 a_2 in-lane obj(앞차에 sharply 국소화). pedal magnitude grad-cam 은 i2_mem 압축으로 diffuse.

---

## 8. 배포 / 재현

### 8.1 weight (git 동봉)
`data/ckpts/` : `main_prof654.pth`, `a2_v7b_d128_alpha075_best_loss.pt`,
`seg_pretrain_fx640_v3_backbone.pt`, `prof635_side_v5_best.pt`(이 보고서 기준 v5 weight).
(현행 배포 f_2 = `prof635_side_distbrake_v1.pt`(brake_only) — §9.3 참조. v5_best.pt 는 legacy.)

### 8.2 inference core
`src/utils/prof635_infer.py` — `infer_step(model, raceline, ex, ey, e_yaw, ego6, i2_mem, cutoff, polygon)`.
raceline: `load_raceline_dt` ([t,x,y,yaw] + 옵션 6-vertex polygon), `resolve_cutoff`
(filename `_XXXXXYYYYY.csv` 또는 track default: speedway 1088, sangam 1300), `sample_traj(mode='speed')`.

### 8.3 offline smoke test
`python scripts/test_prof635_infer.py` → steer MAE 2.61°, corr 0.93, raceline 파이프라인 유한 출력.

### 8.4 closed-loop (ROS)
`scripts/closed_loop_node.py` 는 i2_mem fix(seg=fx640_v3, ego_mot=0) 반영됨.
(작성 시점 기준 `MODEL_TYPE="prof635"` 분기 + dt0p1s raceline 은 미구현/미동봉으로 적었으나,
이후 완료됨 — 현행 repo 에 prof635 분기(closed_loop_node.py) 구현됨 + dt0p1s 통합맵
`data/control/{track}_{map}_{P:05d}_{M:05d}.csv` 4종 동봉. 구 `raceline_sangam.csv`/`raceline_speedway.csv` 는 제거됨.)

---

## 9. 최신 변경 (v6 — brake-only delta + fixed-thresh gate + distance-onset brake label)

v5(§1~8) 이후 변경분. **모델 구조(클래스)는 그대로**, forward 의 delta/gate 모드와 **pedal label 생성 방식**만 바뀜.

### 9.1 gate = fixed_thresh (gate_bias 폐기)

§2·§4.2 의 `g = sigmoid(max(a2 obj) + gate_bias)` 는 **폐기**. `gate_bias` 가 supervision 없이 +3.02 까지
drift → 앞차 없어도 g≈1 → **no-lead False-Positive brake**(공식 테스트 중 발견: 이미지 끄면 정상 주행). 원인:
residual_gate 구조에서 gate 와 delta 가 서로 redundant 해 gate 에 직접 학습 신호가 없었음.

현행 `gate_mode`:
```
fixed_thresh (배포 기본):  g = sigmoid((obj_amax − GATE_THRESH) · GATE_SCALE)   # GATE_THRESH=0.3, GATE_SCALE=5.0
open (학습 기본):          g = 1                                                # f_2 를 전 데이터로 학습, 추론서 fixed_thresh
learned_bias (legacy):     g = sigmoid(obj_amax + gate_bias)                    # drift → 폐기
obj_amax = a2_logits[:, 16:26, 0].amax(dim=1)   # in-lane 객체 인지 (no a2 → g=0)
```
학습은 `gate_mode=open`(g=1)으로 f_2 가 모든 프레임을 보게 하고, **추론(closed_loop_node)에서 fixed_thresh 로 게이팅**.
`PROF635_GATE_THRESH/SCALE` 환경변수로 노드에서 override.

### 9.2 delta_mode = brake_only

§4.3 의 `delta = 2·tanh(head)` (signed, ±2) 는 가속(+) 보정도 가능해 no-lead 상황에서 FP 가속 위험.
현행 기본 `brake_only`:
```
brake_only (기본):  delta = −2 · sigmoid(head)      # delta ≤ 0, 제동만 (가속 간섭 불가)
signed (legacy):    delta =  2 · tanh(head)          # ±2
```
`pedal_head` bias init = **−4** (brake_only): sigmoid(−4)≈0.018 → 시작 delta≈0 → final≈base (warm start 유지).
`final_pedal = clamp(base_pedal + g·delta, −1, 1)` 는 동일.

### 9.3 pedal label — distance-onset brake (`_distbrake`, closing 비의존)

동기: v5 의 `_ttc` 는 closing(상대 접근속도) 기반 ACC 인데, a_2 의 vL_rel corr 0.40(약함)이라 closing
modulation 이 신뢰 안 됨(§7, [STATE_AND_LESSONS]). 반면 a_2 의 **거리(dist) / 객체 폭(bin span)** 은 잘 추정됨.
→ **"가까우면 제동" — 거리만으로 감속**을 정의. closing 항 제거.

`scripts/gen_ttc_pedal_labels.py`, `BRAKE_MODE=distbrake`:
```
onset(v_ego) = DB_D0 + DB_HEADWAY · v_ego          # -1 제동 시작점(far). 빠를수록 멀리서 제동
u            = clip((onset − dist) / (onset − DB_DFULL), 0, 1)
sd           = DB_MINB + (MAXB − DB_MINB) · u^DB_P  # far(u=0)=MINB → near(u=1)=MAXB, convex
sd           = MAXB        if dist ≤ DB_DFULL        # 임박 full brake
sd           = 0           if dist ≥ onset(v)        # onset 바깥 → 제동 없음
params: DB_D0=10, DB_HEADWAY=1.0, DB_DFULL=7, DB_MINB=1, MAXB=20, DB_P=2
lead = isfinite(dist) & (dist < onset)              # closing 무관, onset 안쪽이면 fire
aug  = min(raw_ax, −sd)  (첫 KSTEPS=10 step)         # floor: 제동만 강화, 원래가 더 세면 유지
```
- onset 이 속도로 이동(headway) → 고속일수록 먼 거리서 제동 시작, 저속은 onset≈DB_D0(10m) 안쪽만(zero 로 죽지 않음).
- `floor` 라 코너 등 expert 의 더 큰 감속은 보존(`raw_ax < −sd` 면 raw 유지).
- 검증 plot: `scripts/viz_distbrake_verify.py` (x=거리, y=속도, color=감속).
- 생성: `BRAKE_MODE=distbrake python scripts/gen_ttc_pedal_labels.py` → `pedal_label_{ds}_distbrake.npz` (5 dataset).
  `_ttc` 는 보존(덮어쓰지 않음).

### 9.4 pedal_label 벡터화 (segfault fix)

`src/utils/pedal_label.py` `label()` 을 **벡터 bilinear** 로 재작성(이전 quantize→unique→scalar 678k 호출).
distbrake 의 연속 brake 값이 unique (v,a) 쌍을 ~678k 로 만들어 `pedal_map.get_pedal_values` scalar 를 그만큼
호출 → numpy 1.24 `__array_function__` dispatch 가 비결정적으로 깨짐(UnboundLocalError/TypeError → **segfault**).
clamp 후 grid 내부라 pedal_map 의 edge 분기는 도달 불가 → `_interp_vec` 결과가 scalar 경로와 **max diff 3e-8**
(사실상 동일, quantize 오차도 제거). pedal_map 자체는 무변경(배포 일치).

### 9.5 학습 (distbrake)

```
python scripts/train_prof635_side.py \
    --datasets sangam_lead_0520 sangam_lead_0521 sangam_pertO_v2 sangam_pertX_v2 DriverB_new \
    --pedal-suffix distbrake --gate-mode open --delta-mode brake_only \
    --name prof635_side_distbrake_v1 --main-ckpt data/ckpts/main_prof654.pth
```
`--pedal-suffix distbrake` 가 `pedal_label_{ds}_distbrake.npz` 를 로드. 나머지(loss=L1 step0, Adam 3e-4, main frozen)는 v5 동일.
distbrake fire 율: sangam_lead_0520 83% / 0521 8% / pertO·pertX 0%(lead 없음→raw cornering) / DriverB 23%.

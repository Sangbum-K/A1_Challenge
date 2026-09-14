# 모델 아키텍처 (4 모듈 상세)

perception(seg → a_2) → control(f_1 → f_2). 현행 배포 = **prof635 + distbrake_v1**(`prof635_side_distbrake_v1.pt`, delta_mode=**brake_only**). v5(signed)는 legacy. iy_model/v2 legacy 는 §5.

공통: 카메라 front(CAMERA_1) ROI crop `y[235:485]` → 250×1280, ImageNet normalize. 토큰화는 가로 40 strip.

---

## 1. seg backbone — `SegPretrainModel` (`seg_pretrain_model.py`)

- timm `tf_efficientnetv2_s`, `features_only`, `out_indices=[0..4]`(stride 2/4/8/16/32) + U-Net 디코더(4× `_UpBlock` ConvTranspose+skip, d_mid=64).
- **4-class** seg: 0 bg / 1 road(track+lane) / 2 vehicle / 3 wall (ignore→-100).
- IO: `(B,3,250,1280)` → logits `(B,4,250,1280)`.
- 용도: **backbone 만** 추출(`backbone_state_dict()`)해 downstream i_1 encoder 로 사용. seg head 는 학습 신호용(+debug overlay).
- 현행 ckpt: `seg_pretrain_fx640_v3_backbone.pt` (mIoU 0.921). **frozen** — 재학습 시 모든 i1/i2 캐시 cascade.

---

## 2. a_2 (AuxHead) — `A2Model` (`a2_model.py`)

3 서브:
- **i_1 (`LateralStripEncoder`)**: backbone `out_indices=[-1]` → `adaptive_avg_pool2d(1,40)` → `Conv1d` proj → `(B,40,d_token=256)`. 가로 40 컬럼 토큰.
- **i_2 (`ImageFusionBlock`)**: shared pre-self-attn → cross-attn (Q=curr, KV=`[ego_proj(2→d), prev]`, `frame_type_emb(3)` + `lateral_pos(40,d)`) → post-self-attn. curr/prev 2프레임 fusion.
- **a_2 (`AuxHead`)**: 단일 Linear.
- **`fc_proj`**: `Linear(d_token→d_out)`. v5/v6 는 `d_out=128` (i2_mem 차원, f_2 입력과 일치).

입력: `(image_curr, image_prev_rot, ego_motion(B,2)=[vx, wz])`.
출력 채널 **7ch (legacy)**: `[obj, vL_abs, vY_abs, vL_rel, vY_rel, prox, dist]` (in-lane 컬럼 bin 16-25 가 정면). `predict_var=True` 면 13ch interleaved(mean+logvar).
- `freeze_encoder_fusion()`: fc_proj+head 만 학습. `load_seg_pretrain(ckpt)`: seg backbone 주입.
- 현행 ckpt: `a2_v7b_d128_alpha075_best_loss.pt` (focal α=0.75). obj prec~66%, **vL_rel corr~0.40 (약함 — ACC 한계 원인)**.

배포 산출물: **i2_mem(40,128)** = `fc_proj(fusion(...))`, **a2_logits(40,7)** = `head(i2_mem)`. f_2 와 gate 의 입력.

---

## 3. f_1 (main, #654) — `Prof635Main` (`prof635_side_model.py`)

controller_imitator #654 (SelfAttentionControlPredictor). **이미지 안 봄.**
- `traj_embed Linear(3,128)`, `state_embed Linear(6,128)`, token = `traj_embed(traj)+state_embed(state).unsqueeze(1)`.
- sinusoidal pos(max_len 41) → `TransformerEncoder`(d128, nhead8, ff512, 6L, dropout0.1, batch_first) → `fc_out Linear(128,2)` = `[steer/40, accel_brake]`.
- 사용: `main(traj6,ego6)[:,0,:]` (pred_first_only). **frozen.**
- ckpt: `main_prof654.pth` (config interp005_relspd_notime, traj 채널2=speed).

입력 규격(재현):
- **traj6 (6,3)**: prebake traj_40(0.1s) 를 `T6=[0.05,0.30,0.55,0.80,1.05,1.30]s` 보간. col0=`−y_left`, col1=`+x_fwd`, col2=`speed`(mode='speed', #654). inference 는 raceline 에서 `sample_traj(mode='speed')`.
- **ego6 (6,)**: `[vx,vy,ax,ay,tire_angle,yaw_rate]`. tire = EGO steering 있으면 그것, 없으면 `atan(WB·yaw_rate/vx)` (WB=2.35, |vx|≥5 아니면 0).

---

## 4. f_2 (side) — `Prof635SideModel` (`prof635_side_model.py`) — 현행 학습 대상

f_1(#654) frozen 를 감싸고, 이미지 기반 pedal **보정(delta)**만 학습.
- `side_state_embed Linear(6,d)`, `side_pedal_embed Linear(1,d)`(f_1 base_pedal conditioning), `side_vrel_embed Linear(1,d)`(진단용 GT vL_rel 주입, optional).
- query `q = side_state_embed(ego) + side_pedal_embed(base_pedal.detach())` (B,1,d) — **traj 안 봄** (`use_traj_in_side=False`).
- KV = `img_proj(i2_mem)` (B,40,d). `side_dec TransformerDecoder`(d128, nhead8, ff512, **4L**) → `pedal_head Linear(d,1)` (zero-init → warm start).

forward:
```
out = main(traj6,ego6)[:,0,:]; steer=out[:,0]; base_pedal=out[:,1]   # frozen
head  = pedal_head(side_dec(q, img_proj(i2_mem))[:,0])
delta = −2·sigmoid(head)   (brake_only, 현행)  |  2·tanh(head)  (signed, legacy ±2)
g = sigmoid((a2_logits[:,16:26,0].amax − GATE_THRESH)·GATE_SCALE)     # fixed_thresh (기본 0.3/5.0)
final_pedal = clamp(base_pedal + g·delta, −1, 1)
```
- **delta_mode**: `brake_only`(현행) — delta≤0, 제동만(가속 간섭 불가). pedal_head bias init −4 → 시작 delta≈0(warm start). legacy `signed`(±2).
- **gate_mode**: `fixed_thresh`(현행, drift-free) — a_2 in-lane obj 임계 게이트. 학습은 `open`(g=1)으로 전 데이터 학습 후 추론서 fixed_thresh. legacy `learned_bias`(g=sigmoid(obj+gate_bias))는 +3.02 drift → no-lead FP brake 유발(폐기). [STATE_AND_LESSONS.md] 참조.
- **학습 target(pedal label)**: `_ttc`(closing 기반 ACC) 또는 `_distbrake`(거리-onset 곡선, closing 비의존 — a_2 vL_rel 약점 회피). `train_prof635_side.py --pedal-suffix`. [REPORT_prof635_v5.md §9].
- 출력: steer(deg)=steer·40, pedal(-1~1) → accel=max(p,0)/brake=max(-p,0).
- ckpt: **배포 = `prof635_side_distbrake_v1.pt`** (distbrake 라벨 + brake_only, 5 dataset 975K, val_pedal~0.52[no-lead 지배]). legacy: `prof635_side_v5_best.pt`(signed, ttc, val_pedal 0.0237), `prof635_side_v6_driverb`(+DriverB).

설계 의도: f_1=경로+자기상태로 조향·기본가속, f_2=이미지로 앞차 있을 때만 pedal 보정. 상세 재현: [REPORT_prof635_v5.md].

---

## 5. legacy 모듈 (참고, 현행 아님)

- **IyModel** (`iy_model.py`): i_y refactor (prof635 이전). i_1 토큰 캐시 입력 + `yaw_shift_tokens_batch`(prev 토큰 정수-bin yaw shift, `dx=round(fx·d_yaw/32)` circular roll) + i_2 fusion + Professor core(6L). main+side(cross-attn / ControlNet α-gate / min-arb) 변종. `closed_loop_node.py` 의 `MODEL_TYPE=iy` 경로가 아직 지원(v25h/v25k ckpt). **`yaw_shift_tokens_batch` 는 prof635 캐시 빌드·closed_loop 에서도 재사용** (현역 헬퍼).
- **SCCControllerImitatorV2** (`image_scc_models_v2.py`): one-shot(raw 이미지 직접) 구버전. `MODEL_TYPE=v2`.

prof635 가 #654 정통 I/O 를 frozen 으로 쓰고 side 만 학습하는 현행 방향. iy/v2 는 점진 폐기.

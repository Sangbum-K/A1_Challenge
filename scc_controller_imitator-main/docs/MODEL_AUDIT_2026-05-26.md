# MODEL AUDIT — prof635 + distbrake (2026-05-26)

정적 분석 + dimension/load smoke test 기반 audit. 코드 수정 없음(점검·문서화만).
범위: 추론(closed_loop_node prof635 분기) · 모델(prof635_side / a2_model) · 데이터/학습(prof635_dataset / train_prof635_side / gen_ttc_pedal_labels / build_i2_cache / compute_i1_tok_cache / pedal_label) · raceline(raceline_dt0p1s / raceline_search) · 문서 drift.

검증 방법: 코드 정적 분석 + `ml_con` 환경에서 실제 import·forward·ckpt load smoke test(GPU 미사용, CPU). 모든 ckpt load missing/unexpected=0 으로 확인.

---

## 1. 파이프라인 연결도 (배포 = prof635 + distbrake)

```
camera /raw_img_app/perc/camera_1 (720x1280 rgb8/bgr8, manual decode)
   │ ROI y[235:485] → (250,1280,3) → ImageNet norm
   ▼
seg backbone  tf_efficientnetv2_s (features_only,[-1])  ← seg_pretrain_fx640_v3_backbone.pt (frozen, eval)
   │ feat (B,256,8,40)  → adaptive_avg_pool2d(1,40) → (B,256,40) → transpose → fc/fp (B,40,256)
   │ C_b = 256  (★IyModel docstring 의 "320" 은 오기 — closed_loop MODEL_C_B=256 이 정답)
   ▼  prev: yaw_shift_tokens_batch(fp, d_yaw, fx=640)  (정수 bin roll, bin_w=32)
a_2  A2Model(d_token=256, n_ch=7, d_out=128)  ← a2_v7b_d128_alpha075_best_loss.pt (frozen, eval)
   │ encoder.proj(Conv1d 256→256) → fusion(tc,tp, ego_mot=ZEROS) (B,40,256)
   │ fc_proj(Linear 256→128)  = i2_mem (B,40,128)
   │ head(Linear 128→7)       = a2_logits (B,40,7) [obj, vL_abs, vY_abs, vL_rel, vY_rel, prox, dist]
   ▼
f_1 main  Prof635Main(traj_dim=3,state_dim=6,d128,6L,out2)  ← main_prof654.pth (frozen, no_grad)
   │ traj6 (6,3)[-y,x,speed] + ego6 (6)[vx,vy,ax,ay,tire,yaw_rate]
   │ main(traj6,ego6)[:,0,:] = [steer/40, base_pedal]
   ▼
f_2 side  Prof635SideModel.side_dec (TransformerDecoder 4L, d128)  ← prof635_side_distbrake_v1.pt
   │ q = side_state_embed(ego6) + side_pedal_embed(base_pedal)   (B,1,128)  (use_traj_in_side=False)
   │ kv = img_proj(i2_mem)  (Linear 128→128)  (B,40,128)
   │ head = pedal_head(side_dec(q,kv)[:,0])  → delta = -2·sigmoid(head)  (brake_only ∈(-2,0])
   ▼
gate (closed_loop 외부 재계산, model 반환 final_pedal 미사용)
   │ obj_amax = a2_logits[0, 16:26, 0].max()              # in-lane(bin16~25) ch0=obj
   │ g_raw = sigmoid((obj_amax - 0.3)·5.0)                # THRESH=0.3 SCALE=5.0
   │ gate  = EMA(0.7)·prev + 0.3·g_raw                    # GATE_EMA=0.7
   │ pedal = clamp(base_pedal + gate·delta, -1, 1)
   ▼
publish: command_steer(deg=steer·40) · command_pedal_accel(max(p,0)) · command_pedal_brake(max(-p,0))
raceline: raceline_dt0p1s.load_map(TRACK,MAP,REPO_ROOT/data) → sample_traj40 (40,3) → build_traj6(mode=speed) (6,3)
```

인터페이스 dim 은 전부 smoke test 로 확인:
i2_mem (1,40,128) · a2_logits (1,40,7) · traj6 (1,6,3) · ego6 (1,6) · steer_n scalar · delta scalar. side_dec layers=4, img_proj 128→128.

---

## 2. 검증 OK 목록

| 항목 | 결과 |
|---|---|
| prof635_side_distbrake_v1.pt load (deploy config) | missing=0 unexpected=0 (정확 일치) |
| a2_v7b ckpt load (A2Model d_token=256 d_out=128 n_ch=7) | missing=0 unexpected=0 shape-skip=0 |
| backbone C_b | tf_efficientnetv2_s feature[-1]=256, ROI(250,1280)→feat(256,8,40), pool(1,40) width 정확히 40 |
| seg ckpt 동일성 | `data/ckpts/seg_pretrain_fx640_v3_backbone.pt` == `outputs_multi/.../best_miou_backbone.pt` (md5 일치) |
| i2_mem live↔cache 불변식 | seg=fx640_v3 ✓, ego_mot=zeros ✓(L604, L690), prev yaw_shift fx=640 + encoder.proj 순서 = build_i2_cache.py 와 동일 |
| prev offset | closed_loop PREV_IMG_DT=0.20s(=20tick) = build_i2_cache prev_dtick=20 = prebake prev20 |
| delta_mode ↔ ckpt | DELTA_MODE 기본=brake_only, distbrake_v1 = brake_only 학습(REPORT §9.5) → 일치 |
| gate 채널 규약 | a2_logits[...,16:26,0] ch0=obj. A2Model AuxHead ch0=obj ✓. SingleObjDist 도 ch0=obj 유지 |
| raceline 4 맵 로드 | sangam/speedway × mild/high 전부 N/P/M/polygon(6) 정상. 파일명 P/M = CUTOFFS 일치 |
| sample_traj40 → build_traj6 | ego on-line → traj[0]≈원점, speed 채널 전파 정상, mode=speed 학습(prof635_dataset)과 일치 |
| traj6 부호 규약 | col0=-y_left, col1=+x_fwd, col2=speed — build_traj6 / raceline_search.to_prof654_traj 일치 |
| pedal_label 벡터화 | _interp_vec vs scalar max diff 3e-8 (배포 pedal_map 무변경) |
| 이번 세션 변경 호환성 | randomize_base 추가(eval 시 무영향), base.detach 제거(no_grad output이라 무영향), A2ModelSingleObjDist 추가(기존 A2Model 무손상) — 모두 distbrake_v1 load missing=0 유지 |

---

## 3. 발견된 이슈 / 불일치 / 위험

심각도: HIGH(배포 위험) / MED(혼란·유지보수) / LOW(정리·오기).

### [MED] 문서 drift — production 표기가 v5, 배포 실체는 distbrake
- `closed_loop_node.py:56-61` deploy default = `prof635_side_distbrake_v1.pt` + `DELTA_MODE=brake_only`.
- 그러나 `docs/ARCHITECTURE.md:3`, `docs/STATE_AND_LESSONS.md:9`, `docs/README.md:6,38,52`, `docs/ARCHITECTURE.md:69` 는 "production = prof635 v5"(`prof635_side_v5_best.pt`)·distbrake="실험중" 으로 표기.
- 즉 배포 코드는 distbrake 로 이미 전환됐는데 문서 3종이 v5 로 남음. delta_mode 도 v5=signed / distbrake=brake_only 로 다름.
- 영향: 다음 에이전트가 문서만 보면 deploy ckpt·delta_mode 를 v5(signed)로 오인 가능. (코드 주석 L57/L60/L431 은 정확히 brake_only 명시 — 코드 내부는 일관).
- 권고: REPORT_prof635_v5.md §9 처럼 "현행 배포 = distbrake_v1 / brake_only" 로 README·ARCHITECTURE·STATE 동기화.

### [MED] STATE_AND_LESSONS §2.4 "pedal label 은 _ttc 버전만"
- `docs/STATE_AND_LESSONS.md:36-37` 은 _ttc 만 언급. 실제 배포 ckpt(distbrake_v1)는 `_distbrake` 라벨로 학습. (REPORT §9.3·DATA_PIPELINE·ARCHITECTURE 은 distbrake 반영) → STATE 만 미반영.

### [MED] train_prof635_side.py `--pedal-suffix` default = ttc (배포 ckpt 는 distbrake)
- `train_prof635_side.py:41` default `ttc`. distbrake_v1 재현하려면 반드시 `--pedal-suffix distbrake --delta-mode brake_only` 명시 필요(REPORT §9.5).
- DATASETS dict 의 3번째 항목(`pedal_label_*_ttc.npz`)은 코드에서 무시되고 `f"pedal_label_{nm}_{suffix}.npz"` 로 재구성됨 → dict 값은 stale(혼란 유발). (LOW)
- 위험: default 로 `launch_prof635_v6_driverb.sh` 재실행 시 ttc 라벨로 학습돼 배포 ckpt 와 다른 모델 생성. launcher 가 suffix 미지정.

### [MED] 테스트 스크립트 stale — deploy 구성과 불일치
- `test_prof635_closedloop.py`: (1) `data/control/raceline_sangam.csv`·`raceline_speedway.csv` 를 `open()` — 현재 없음(dt0p1s 통합맵으로 대체) → FileNotFoundError 로 죽음. (2) `prof635_side_v5_best.pt` 사용(deploy=distbrake_v1). (3) 자체 `sample_raceline_traj`(구식 row-spacing) — closed_loop 의 `raceline_dt0p1s.sample_traj40` 와 다른 경로. **TESTING.md:34·42 가 이 스크립트를 "PASS 확인"으로 안내하나 실제 실패.**
- `test_prof635_infer.py`: default `--model-ckpt prof635_side_v5_best.pt`·`--raceline data/control/raceline_sangam.csv`. Test B 는 try/except 로 graceful skip(죽진 않음)이나 default ckpt 가 v5.
- `ablate_traj_speed.py`(TESTING.md:44 예시)도 v5_best default.
- 권고: 테스트 default ckpt 를 distbrake_v1 로, raceline 인자를 dt0p1s 통합맵(또는 raceline_dt0p1s.load_map)으로 갱신.

### [LOW] closed_loop_node.py dead code
- `_load_raceline`(L233-251)·`_sample_raceline_traj`(L254-290): prof635/iy 경로 모두 `raceline_dt0p1s.sample_traj40` 사용 → 미사용. 의도 점검 후 제거 가능.
- `MODEL_CKPT_V2`/legacy v2 분기(L46, L466-486): MODEL_TYPE=v2 미사용(점진 폐기).

### [LOW] IyModel docstring/default 오기
- `iy_model.py:115` `C_b: int = 320` + 주석 "tf_efficientnetv2_s = 320". 실제 256. closed_loop 는 `MODEL_C_B=256` 명시 호출이라 무해하나, IyModel 을 default 로 직접 생성 시 img_proj Conv1d(320→128) 가 (40,256) 토큰과 mismatch. (현재 그렇게 호출하는 경로 없음).

### [INFO/관찰] raceline speed 채널 크기
- sample_traj40 결과 speed 채널이 ~55 m/s(≈200km/h)까지. derive_speed 가 점 간격을 0.1s 로 나눈 값이라, dt0p1s csv 가 실제 0.1s 등간격이어야 정확. 맵 데이터 생성(build_raceline_map) 영역 — audit 범위 밖. 학습 prebake traj-speed 와 단위 일치 여부는 별도 검수 권장(과거 DriverB traj 손상 incident 있었음).

---

## 4. 핵심 불변식 요약 (위반 = 배포 붕괴)

1. i2_mem live == cache: seg=`fx640_v3` + ego_mot=`zeros` + prev offset=200ms(20tick) + prev yaw_shift(fx=640) → encoder.proj. 세 조건 다 closed_loop·build_i2_cache 일치 확인. (틀리면 cosine 0.80, SCC/제동 붕괴 — verify_i2_live_vs_cache.py)
2. delta_mode == ckpt 학습값: 배포 `DELTA_MODE=brake_only` == distbrake_v1 학습. (signed ckpt 배포 시 head 오해석 — 치명)
3. gate: closed_loop 가 model 반환 final 을 버리고 `base + EMA(g_raw)·delta` 외부 재계산. g_raw=sigmoid((obj_amax-0.3)·5.0), EMA=0.7. SIDE_OFF=1 → gate=0(f_1 단독). (학습은 gate_mode=open g=1, 추론만 fixed_thresh — 의도된 설계)
4. 상수: MODEL_FX=640, ROI=[235:485], ImageNet norm, C_b=256, T6=[0.05,0.30,0.55,0.80,1.05,1.30], TSRC=arange(40)·0.1, WB=2.35, STEER_NORM=40, build_traj6 mode="speed" — closed_loop·prof635_dataset·raceline_search 전부 동일 확인.
5. obj 채널: a2_logits ch0=obj 고정. in-lane = bin 16~25.

---

## 5. 현재 배포 구성 스냅샷

ckpt (전부 data/ckpts/ 존재, smoke load 통과):
- main(f_1, frozen): `main_prof654.pth`
- side(f_2): `prof635_side_distbrake_v1.pt`  (= outputs_multi/prof635_side_distbrake_v1/best.pt, md5 일치, 5 dataset 975,871 sample, brake_only, gate=open 학습, val_pedal 수렴 ~0.52)
- a_2: `a2_v7b_d128_alpha075_best_loss.pt`
- seg backbone: `seg_pretrain_fx640_v3_backbone.pt` (mIoU 0.921)

env 기본값 (closed_loop_node.py):
- MODEL_TYPE=prof635 · MODEL_VARIANT=A(prof635 에선 미사용) · DELTA_MODE=brake_only
- GATE_THRESH=0.3 · GATE_SCALE=5.0 · GATE_EMA=0.7 · SIDE_OFF=0
- TRACK=sangam · MAP=mild · MODEL_FX=640
- PREV_IMG_DT=0.20 · INFER_RATE_HZ=20 · ROI=[235:485]

raceline 맵: `data/control/{track}_{map}_{P:05d}_{M:05d}.csv` (raceline + 마지막 6행 pit polygon). P/M:
sangam mild (246,1333) · sangam high (252,1308) · speedway mild (268,1097) · speedway high (274,1084). 전부 CUTOFFS 와 일치.

---

## 6. 다음 에이전트가 알아야 할 것

함정 (우선순위 순):
1. **배포 = distbrake_v1 + brake_only** 이지 v5(signed) 아님. 문서(README/ARCHITECTURE/STATE) 의 "v5" 표기에 속지 말 것. delta_mode 를 ckpt 학습값과 반드시 일치시킬 것(불일치=치명).
2. **closed-loop 제동 이상 → 모델보다 i2_mem(perception) 먼저 의심.** seg=fx640_v3·ego_mot=zeros·prev=200ms 세 조건. verify_i2_live_vs_cache.py 로 cosine 확인.
3. **f_1(base)는 lead 제동 안 함**(이미지 안 봄, traj-speed artifact). 종방향 lead 제동은 전적으로 f_2(image)+gate. 그래서 i2_mem·a_2·gate 가 틀리면 제동 자체가 사라짐.
4. **test_prof635_closedloop.py 는 현재 죽는다**(raceline_sangam.csv 없음). 검증은 `test_prof635_infer.py`(graceful) + `verify_i2_live_vs_cache.py` 사용. 테스트 ckpt default 도 v5 라 distbrake 검증하려면 `--model-ckpt` 명시.
5. **distbrake 재학습 시** `train_prof635_side.py --pedal-suffix distbrake --delta-mode brake_only` 명시. default(ttc) 로 돌리면 배포와 다른 모델. launcher(launch_prof635_v6_driverb.sh)는 suffix 미지정 → default ttc 로 학습됨(주의).
6. **seg backbone 재학습 = 전체 cascade**(i1/i2 캐시·a_2·f_2 전부 재생성). 함부로 금지.

미해결 / 진행 중:
- ACC(상대속도 기반 제동) 약함 — a_2 vL_rel corr 0.40. → distbrake(거리 기반)로 우회 중. 근본 해결은 v7 재귀형 i_2(STATE §3).
- A2ModelSingleObjDist(single-frame obj·dist, n_ch=2) 학습 중(launch_a2_single_objdist.sh) — 아직 배포 안 됨. deploy 전환 시 closed_loop a2_v6 inference 경로(fusion 사용)와 다름(SingleObjDist 는 forward_tokens/self-attn) → 배포 코드 분기 추가 필요.
- f_2 가 fixed_thresh 면 no-lead(gate≈0) 무학습 → lead 샘플(~16%)만 효과적 학습. lead-upweight 검토 중.
- legacy iy/v2 경로 정리 미완(closed_loop 에 잔존).

검증 못 한 것(범위 밖):
- 실제 ROS 주행(roscore·시뮬 필요, GPU 미사용 조건). 본 audit 은 정적+CPU smoke 까지.
- raceline csv 의 0.1s 등간격 실측(speed 채널 정확도) — 맵 데이터 검수 영역.
- distbrake_v1 의 lead-specific 제동 품질(global val_pedal 은 no-lead 지배) — gradcam/lead-TTC 평가 필요.

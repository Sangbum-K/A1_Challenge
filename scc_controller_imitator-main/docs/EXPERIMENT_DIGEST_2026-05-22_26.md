# 실험 다이제스트 — 2026-05-22 → 2026-05-26

> 대상: `scc_controller_imitator` (A1 CHALLENGE 2026, 팀 한양 AutoHYU 카메라+ego→steer/pedal 레이싱 컨트롤러).
> 범위: 5/22~5/26. §0~8 = 5/22~5/25 약 4일(main + acc-test + qualifier-{beat,acc}-minimal + feat/moraibridge 브랜치). **§9 = 5/26 주간·야간 세션**(a_2 object separation 심화 + temporal ablation + bin80 설계 + latency/backbone 검토 + SLURM 셋업) 추가.
> 목적: TOPIC 이 유의미한 실험만 추려 목적/설계/결과로 구조화. tuning noise 는 시리즈로 묶어 결론만 남김.
> 수치는 전부 `outputs_multi/**/train.log` · eval JSON · ckpt 파일명에서 인용(추정은 명시). 코드 미수정.

---

## 0. 파이프라인 (이 다이제스트의 전제)

```
camera(curr,prev) → seg backbone(tf_efficientnetv2_s, fx640_v3, frozen) → 40×256 토큰
   → a_2 (A2Model, 2-frame fusion, d_out=128) → i2_mem(40,128) + a2_logits(40,7)
        ├ f_1 (main, Prof635Main #654, frozen) : traj6 + ego6 → steer, base_pedal  (이미지 안 봄)
        └ f_2 (side, TransformerDecoder cross-attn) : ego6 + base_pedal + KV=i2_mem → pedal delta
   → gate g = sigmoid((a2 in-lane[16:25] obj_amax − 0.3)·5), EMA(0.7)
   → final_pedal = clamp(base_pedal + g·delta, −1, 1)
```

핵심 불변식(STATE_AND_LESSONS): live i2_mem == cache (seg=fx640_v3 + ego_mot=zeros + prev=200ms). 위반 시 cosine 0.80 → SCC/제동 붕괴. 종방향 lead 제동은 **전적으로 f_2(image)+gate** 담당(f_1 의 brake 는 traj-speed artifact).

이번 4일의 큰 줄기: (1) a_2 perception 을 d128·focal·reg-fix 로 안정화하고 (2) **multi-frame 으로 vL_rel 지각을 근본 해결**(corr 0.34→0.92), (3) prof635(f_1 frozen + f_2 Side SCC) 구조로 재설계, (4) 제동 라벨을 closing-ACC(`_ttc`) → 거리-onset(`_distbrake`) → closing-비례(`_ttc2`)/pred-vrel(ACC)로 탐색, (5) 예선/본선 배포 패키징(규정 §1.1/§1.3 단일모델 + MORAI bridge 이전).

---

## 1. Perception — a_2 시리즈

a_2 = seg 토큰에서 per-bin [obj, vL_abs, vY_abs, vL_rel, vY_rel, prox, dist] 7채널을 회귀. 모든 학습은 `a2_v7b` 를 init 으로 warm start.

### 1.1 a_2 v3→v9 본류 (focal / reg-fix / fc_proj 256→128 / 분포보정)

목적: 앞차 인지(obj) FP 줄이기 + vL_rel/dist 회귀 정직화. 본 시리즈는 손실·헤드 구조 탐색이 TOPIC.

설계·결과(전부 best val total loss, 동일 multi-source cache):

| 버전 | 변경(설계) | best val loss | 결론 |
|---|---|---|---|
| v3/v4 | fx640-only, no-lead boost 제거, lead oversample | v4 best_val 0.0185(obj_acc 1.0) | obj 인지 안정화. (loss scale 달라 v5+와 직접 비교 불가) |
| v5 (d128) | **fc_proj 256→128** (i2_mem 차원 축소), cache-only | 0.5882 | d128 i2_mem 채택. 이후 전 cascade 가 d128 |
| v6 | focal loss 도입 (obj imbalance) | 0.4233 | focal 로 loss 구조 정리 |
| v7 | **reg-loss fix** (normalize + obj_mask 누락 버그 수정) | 0.2052 | ★ v5/v6 의 vL_rel pred std=0 버그 제거. 회귀가 비로소 학습됨 |
| v7b | focal **alpha 0.75** | 0.2048 | **배포 채택**(a2_v7b_d128_alpha075). alpha≈1−pos_rate |
| v7c | focal alpha 0.9 | 0.2015 | loss 소폭↓이나 실익 미미, v7b 유지 |
| v8 | Gaussian-NLL + LDS(label distribution smoothing) | 0.2170 | NLL+LDS 조합 이득 없음(v7b 대비 ↑) |
| v9 | LDS only | 0.2136 | 역시 v7b 미달. **LDS/NLL 계열 dead-end** |

status: v7 reg-fix [confirmed 핵심], v7b 배포 [confirmed], v8/v9 분포보정 [dead-end]. v7b 절대 corr 자체는 낮음(2-frame 한계, §1.3). reg-fix 버그는 STATE_AND_LESSONS·MEMORY 에 incident 등재.

### 1.2 yaw_shift (hand-crafted token roll) 무효 — [dead-end]

목적: prev 프레임 토큰을 d_yaw 만큼 정수 bin roll 해 회전 보정하면 obj/dist/vL_rel 이 좋아지는가.

설계: a2_2f_yaw vs a2_2f_noyaw 동일 조건 A/B.

결과: best 0.0943 == 0.0943 (동일). 고-yaw subset 에서도 ON==OFF. → **무효, 제거 가능**(규정상 단순화 이점). 게다가 cache 학습 경로는 d_yaw=0 하드코딩이라 v7b 는 yaw_shift 를 쓴 적도 없음.

### 1.3 ★ Multi-frame a_2 — vL_rel 지각 근본 해결 [confirmed, 이번 주 최대 결과]

목적: STATE_AND_LESSONS §3 의 이중 병목 중 **a_2 의 vL_rel 지각**(2-frame v7b corr 0.34) 을 프레임 수로 해결. (격리실험에서 GT vrel 주입 시 f_2 가 정확히 modulate → 병목은 f_2 capacity 아님, a_2 지각이었음.)

설계: A2ModelMultiFrame, offset 조합 sweep(10-tick granularity, ≤5프레임, span≤50). target=vL_rel·dist·obj. metric=corr / corr_near / MAE / obj F1 / flip_rate. eval=`a2/mf_sweep_eval.json`.

결과 (vL_rel_corr 중심, 동일 held-out):

| 모델 | obj F1 | dist MAE | vLrel MAE | vLrel corr | corr_near | flip |
|---|---|---|---|---|---|---|
| v7b (2f, 기준) | 0.657 | 6.81 | 4.53 | **0.337** | 0.238 | 0.111 |
| 1f curr | 0.770 | 3.53 | 3.46 | 0.697 | 0.587 | 0.180 |
| 2f (10) | 0.780 | 3.11 | 3.18 | 0.735 | 0.642 | 0.193 |
| 3f (20,40) | 0.782 | 2.92 | 2.78 | 0.801 | 0.745 | 0.186 |
| 4f (10,20,40) | 0.784 | 2.78 | 2.73 | 0.807 | 0.746 | 0.186 |
| **5f (10,20,30,40,50)** | **0.789** | **2.66** | **2.52** | **0.836** | **0.777** | 0.196 |

(독립 3-set eval `a2_b0_abl/eval_v2s_3set.json` 에서는 v2s 5f corr **0.920**, near 0.842, dist MAE 1.52 / v7b 2f corr 0.378 — 셋·분할이 달라 절대치는 다르나 추세 동일.)

결론: 프레임↑ → vL_rel corr **단조 개선**(0.34→0.84~0.92), dist/obj 동반 개선, flip 은 소폭↑(허용). **5f(span 500ms) 채택** → `a2_mf5f_acc.pt`. 이게 ACC(상대속도 기반 제동) 재학습의 perception 토대. status [confirmed]. 단 배포 라이브는 i2_mem/gate=v7b 2f 유지, **vrel scalar 만 5f 로 주입**(학습 분포 일치, ACC_TEST_BRANCH).

### 1.4 a_2 v8 균형 275K cascade 후보 [in-progress, 채택 대기]

목적: spd_v2 합류 + pertO/pertX run-filter 정합으로 speedway obj/vrel 개선, DriverB 무손상.

설계: 6 i1_cache(sangam_lead×2, DriverB_new, spd_v2, pertO_v2, pertX_v2) join, total **275,999**(train 262,200 / val 13,799), v7b resume, focal alpha0.75.

결과: best val total **0.1995**(a2 본류 최저). TODO_IDEAS 기준 speedway F1+4.5 / vL_rel+0.07 / DriverB 무손상으로 cascade 채택 가치 확정. **단 v8 vs 5f-sweep 중 무엇을 cascade base 로 할지 sweep 결과 보고 결정 — i2_mem 6개 재생성 + f_2 재학습 필요라 미실행.** status [in-progress].

### 1.5 A2ModelSingleObjDist (single-frame obj·dist, n_ch=2) [in-progress]

목적: gate 전용 경량 단일프레임 obj/dist 헤드. 설계: self-attn forward_tokens, n_ch=2. 결과: 학습은 됐으나 **배포 안 됨** — closed_loop a2_v6 inference 경로(fusion)와 다름(분기 추가 필요). status [in-progress / 미배포].

---

## 2. Backbone — B0 vs V2-S ablation [confirmed-부분, 교체 보류]

목적: closed-loop 추론 latency 단축 위해 seg backbone(v2_s) 를 경량 b0 로 교체할 가치 판단. (seg 는 a_2 의 frozen feature extractor → 판단 기준은 seg mIoU 가 아니라 a_2-on-b0 품질.)

설계·결과 두 축:

(a) seg 단독 (REPORT_seg_backbone_lite): b0 i_1 토큰경로 latency **2.14x**(5.01→2.34ms), mIoU 0.897(−0.024), **vehicle IoU 0.811(−0.055)** — lead car 직결 클래스에 저하 집중. b1/b2/b3 는 b0 에 속도-dominated → 중단. mobilenetv3 는 last ch 960 caveat 로 보류.

(b) a_2-on-b0 (`a2_b0_abl/eval_b0.json`, 둘 다 5f): 

| | obj F1 | dist MAE | vLrel corr | corr_near | flip |
|---|---|---|---|---|---|
| b0 5f | 0.812 | 2.02 | 0.888 | 0.782 | 0.123 |
| v2s 5f | **0.845** | **1.52** | **0.920** | **0.842** | 0.083 |

결론: b0 가 obj F1 −0.033, vLrel corr −0.032, flip +0.04 로 **전 지표 v2s 미달**(vehicle IoU 저하가 obj/vrel 로 전파 확인). latency 2.14x 이득 vs 인지 저하 trade-off → **교체 보류**. gate 안정성 매듭 후 재평가. status: ablation 자체 [confirmed], 교체 결정 [보류].

---

## 3. 제어 — f_1(main) / f_2(Side SCC) 재설계 + 제동 라벨

### 3.1 prof635 재설계: #654 frozen main + Side SCC cross-attn [confirmed 구조]

목적: f_1(이미지 비참조, 경로/ego→steer+base_pedal)과 f_2(image→pedal delta)를 **분리**. f_1 은 #654 frozen, f_2 만 학습. 동기: f_1 의 brake 는 traj-speed artifact라 inference 에서 앞차 제동 불가 → 제동을 f_2(image)+gate 로 이관.

설계: Prof635SideModel — q=side_state_embed(ego6)+side_pedal_embed(base_pedal), KV=img_proj(i2_mem), side_dec(TransformerDecoder 4L, d128), pedal_head. loss=L1(final_pedal, label) step0. trainable ≈1.08M.

결과(legacy v5): best val_pedal **0.0237**, steer MAE 3.50°(frozen). traj-speed shortcut 없는 정직 구조로 채택. ablate_traj_speed: traj-speed flat 시 base −0.48→+0.28(가속)이나 f_2 −0.37 불변 → f_2 의 image 제동 transferable [confirmed]. status [confirmed 구조].

### 3.2 iy 계열 v23→v25 재설계(prof635 흡수 전 과도기) — 시리즈로 묶음

목적: IyModel 에 main_no_image + side cross-attn + a_2 feat + TTC augment + 2-phase 학습을 얹어 prof635 와 같은 분리구조를 IyModel 로 실험. (이후 prof635 전용 모델로 정리됨, iy 는 점진 폐기.)

설계·결과(best_val, close/far 분리 평가):

| 버전 | 핵심 설계 | best_val | 비고 |
|---|---|---|---|
| v23e | main_no_img + side cross + phase1.5 freeze-fix | 0.2353 | freeze 버그 2회 수정(v23c/d/e) |
| v24 | + TTC augment (in-lane bin16-25 gating) | 0.2112 | TTC aug 로 close 0.262→0.252 |
| v25 2phase_ttc | 2-phase(main frozen→joint) + ttc | best 0.2565 | scheduler 버그(_CosineDecayingMaxRestarts denom N−1→N, cycle4 lr=0 정지) fix |
| v25c main_alone | main 단독 재학습(lr3e4 decmax) | val 0.3710 (steer 3.01° ax 0.70) | f_1 base 재현 |
| v25d side_a2_ttc | v25c resume + a2feat + ttc side | val 0.3707 | side 추가가 global val 거의 불변(no-lead 지배) |
| v25h_A image_aware | main 에 이미지 주입 시도 | best 0.1649 (steer 0.75° ax 0.90) | steer 매우 낮으나 train/val 분포 의존, openloop scatter 만 산출 |
| v25h_B / v25j_B0 / v25k_B4 | side cross-attn **layer 수 ablation**(B0=1L … B4=4L) + gttc | B4_gttc_acc val 0.5749(ax 1.85) | B 숫자 = side layer 수. label scale 달라 cross-compare 주의 |

status: 구조 실험은 prof635 로 수렴 [confirmed], iy 자체는 [폐기 진행]. close/far 분리 metric 은 유지가치 있음.

### 3.3 제동 라벨 탐색: `_ttc` → `_distbrake` → `_ttc2` / pred-vrel [핵심 결정]

목적: "앞차에 어떻게 제동을 가르칠 것인가" — closing(상대속도) 기반 vs 거리 기반.

배경 진단(STATE §3): a_2 vL_rel(ch3) ↔ GT closing corr **−0.279**, f_2 delta ↔ a_2 vL_rel **+0.023** → closing modulation 불신. GT vrel 주입 격리실험은 f_2 가 정확 modulate(멀어짐 +0.20 / 접근强 −0.23 monotonic) → 병목은 a_2 지각.

설계·결과(val_pedal — **라벨 분포가 달라 버전 간 직접 비교 불가**, 같은 라벨류 내에서만 비교):

| 라벨/모델 | 정의(설계) | best val_pedal | 결론·status |
|---|---|---|---|
| v5 `_ttc` signed | closing-ACC, delta=2·tanh(±2) | 0.0237 | legacy. closing 불신 + signed FP가속 위험 |
| v6 `_ttc` +DriverB | 〃 + DriverB | 0.0443 | legacy |
| **distbrake_v1** | **거리-onset, closing 비의존**, brake_only(−2·sigmoid), gate=open | **0.5003** | ★ **현행 배포**. onset=D0+headway·v, near=full brake. fire율 lead 83%/pertOX 0% |
| distbrake_randbase | base 랜덤화(over-fit 완화) | 0.5005 | distbrake 변형, 영향 미미 |
| predvrel_ttc | 5f pred vL_rel 주입 + ttc ACC 라벨, brake_only | 0.5118 | acc-test 배포 후보(ACC ON) |
| predvrel_ttc2 | **closing-비례 라벨 재설계** | 0.5111 | ttc2: 강접근에 비례 escalation 의도 |
| predvrel_signed | signed delta | 0.0450(별scale) | ACC modulation 약함 → 비권장 |
| gtvrel_match_ttc / gtvrel_signed | GT vrel 주입(상한 진단) | 0.5098 / 0.0428 | f_2 capacity 충분 재확인 |
| gtv_1L / gtv_4L | side layer 1L vs 4L | 0.4930 / 0.5247 | 4L 이 유리 → side 4L 채택 |

closing 비의존 distbrake 가 배포 기본인 이유: a_2 의 **거리/객체폭은 잘 추정**되나 closing 은 약함 → "가까우면 제동"이 신뢰가능. ACC(pred-vrel) 는 acc-test 브랜치로 분리 검증 중. status: distbrake 배포 [confirmed], pred-vrel ACC [in-progress].

### 3.4 ttc-augment NO-OP 버그 + fix [confirmed 버그]

목적/발견: B 경로(i2_cache)에서 ttc-augment 가 in-lane GT 를 못 받아 **NO-OP**. 원인: inlane 계산이 a2_cache path 에만 있고 i2_cache 빌더에 누락. fix: GT inlane cache(`build_inlane_gt_cache.py`, `inlane_gt_{ds}.npz`) 생성 + dataset wire. status [confirmed fixed]. (MEMORY: ttc_augment_inlane_gt)

### 3.5 ACC closing 제동 — 5f vrel 주입 라이브(acc-test) [in-progress]

목적: 5f pred vL_rel(corr 0.84) 를 closing scalar 로 f_2 에 주입해 ACC 제동.

설계: A2ModelMultiFrame(5f) → in-lane lead pred vL_rel → closing(=−vL_rel) → side_vrel_embed 주입. 오프라인 검증 `test_acc_closedloop.py` PASS(5f 추론 max|err| 0.007m/s, closing sweep brake 0.15→0.62 monotonic escalate).

결과: 약/정속 구분은 되나 **강접근 escalation 포화**(closing +2~+10 구간 delta −1.14→−1.17 평탄, corr 0.84 한계). iy_v26_Bminacc(별실험): close-approach(n=741) brake_dir **85%**, pred_ax ≈ −4.7m/s², ax_err 4.69 — 방향은 맞으나 크기 오차 큼. status [in-progress, escalation 약함이 한계].

### 3.6 gate FP brake fix + 평활 [confirmed + in-progress]

목적: no-lead FP brake 제거(가속 불가 문제).

설계·결과:
- gate_bias(learned) 가 supervision 없이 +3.02 drift → no-lead g≈0.92 → FP brake. **fix: fixed_thresh g=sigmoid((obj_amax−0.3)·5)** (no-lead obj −0.46 / lead +1.41 분리). no-lead brake 24%→0%, lead 제동 100% 유지. gate F1 0.728(P0.85/R0.64), 접근 lead recall 0.90. [confirmed]
- 시간평활: EMA(0.7) 기본 / learned temporal filter(distill·beat, 규정-clean in-graph 대체) 추가. 출발불가(pedal 깔짝거림) → EMA 복원으로 해결. [in-progress]
- soft-aggregation(mean/topk/logsumexp) → 실데이터 FP 92% 폭발/recall 34%. lead 신호가 1-bin 국소라 공간평균 희석. **[dead-end] — 평활은 시간축이 정답.**
- F2_DELTA_SCALE: f_2 brake delta 증폭 노브(기본 1.3, acc) 추가. [confirmed knob]

---

## 4. 데이터

### 4.1 a_2 join 매칭 버그 2건 fix [confirmed]

목적/발견: 학습 join 이 silent 하게 샘플 drop. (B1) pertO run-filter 누락, (B2) spd cache↔aux 날짜 불일치. → **join 211K → 275K** 복구. 가드 추가(join/aux 매칭률 <0.5 fail-fast). 교훈: dataset join 은 항상 매칭률 검증. status [confirmed].

### 4.2 i2_mem zero-fill regen [confirmed]

build_i2_cache 가 prebake∩cache 만 처리 → cache 전프레임 처리로 6개 재생성(100% coverage). 배포 불변식(seg fx640_v3 + ego_mot=0, cos 1.0) 재확인 필요(TODO). status [confirmed regen / verify 대기].

### 4.3 DriverB_new / spd_v2 [in-progress]

DriverB_new 569 runs / 135,113 frames logged-only 처리(i1_cache 진행). DriverB traj 손상(stale source speed 25-63 고정) → raw 재구성(`_trajfix`), #654 steer corr 0.28→0.42(이동프레임 0.72). spd_v2 fx640_v3(mIoU 0.921) i1_cache 3,407. fx640-only 규약 준수(MultiAgnet fx320 폐기). status [in-progress].

### 4.4 distbrake / ttc / inlane_gt 라벨 생성 [confirmed]

`gen_ttc_pedal_labels.py` BRAKE_MODE 분기로 `_ttc`(closing) / `_distbrake`(거리) 5 dataset 생성(공존). pedal_label 벡터화(bilinear)로 segfault fix(scalar 678k 호출 → numpy dispatch 깨짐, max diff 3e-8). `inlane_gt_*.npz` 생성. status [confirmed].

---

## 5. main (f_1) — #655 finetune [confirmed]

목적: Professor #654(best val 0.0355 steer) 개선형. v0301_0303 데이터 ft654.

설계: selfattention_professor interp005_relspd_notime, e3000, cos/coslinear scheduler, steer+accel+brake. ckpt 파일명 `model_<totalval>_<ep>_<steer>_<ax>.pth`.

결과: best `model_1.04510370_365_0.02306919_0.12233608` → **steer val 0.0231**(#654 0.0355 대비 개선), ax 0.1224. coslinear run 이 일반 cos(best 1.06~)보다 약간 우위. status [confirmed]. (배포 f_1 은 여전히 main_prof654.pth — #655 채택 시 cascade 영향 검토 필요.)

---

## 6. 인프라 / 패키징

| 항목 | 내용 | status |
|---|---|---|
| closed_loop 내제화 | A1_Rule 의존(autohyu_msgs/pedal_map/csv/ckpt) 제거, 경로 portable(REPO_ROOT/SCC_DATA_DIR), weight git 동봉 | [confirmed] |
| dt0p1s raceline | mild/high × sangam/speedway 4 통합맵(raceline+6-vertex pit polygon), CUTOFFS 일치 | [confirmed] |
| 예선 beat 패키징 | `A1E2EModel`(src/model/a1_e2e_model.py) 단일모델 통합(§1.1) + gate forward 통합(§1.3 differentiable single chain). qualifier-beat-minimal: pedal_map/legacy 제거(56,751줄 삭제, 464→97) | [in-progress, push 안 함] |
| 본선 ACC 패키징 | `A1E2EModelACC`(§1.1/§1.3) + acc-test 브랜치(MODEL_ACC=1, 5f vrel) | [in-progress] |
| MORAI bridge 이전 | A1_Rule MORAI bridge → moraibridge/ catkin-ws(src/) 3노드(vehicle_driver/control/camera_driver)+simulation.ini+run_bridge.sh. sangam/utm 기본(kcity 좌표 틀어짐 fix) | [confirmed 이전, 빌드검증 대기] |
| SLURM | sbatch runbook/script 준비(제출은 사용자, 키 없음) | [in-progress 문서] |
| cv_bridge NumPy2 segfault | ml_con conda(NumPy 2.4)에서 cv_bridge segfault → manual decode 우회 | [confirmed 우회] |

규정 미결(§=A1): §1.5 KIAPI 제공 reference line 사용 필요(자가 dt0p1s 추종 + mild/high 외부 switch 제거) — 해석/결정 필요.

---

## 7. 현재 배포 상태

- **예선 (beat)**: prof635 + **distbrake_v1**(brake_only, gate fixed_thresh+EMA0.7). 종방향 제동은 **거리-onset**(closing 비의존). f_1=main_prof654(frozen), a_2=v7b_d128(2f), seg=fx640_v3. 단일모델 `A1E2EModel`(§1.1/§1.3) + MORAI bridge. → "앞차 가까우면 제동" 의 보수적·신뢰가능 정책.
- **본선 (ACC)**: acc-test 브랜치 = 동일 cascade + **5f pred vL_rel 주입**(`a2_mf5f_acc.pt`) → closing 기반 제동(`predvrel_ttc`). i2_mem/gate 는 v7b 2f 유지, vrel scalar 만 5f. `A1E2EModelACC` + F2_DELTA_SCALE. → 상대속도 modulation 시도(약접근/정속 구분 OK, 강접근 escalation 약함).

문서 drift 주의(MODEL_AUDIT): README/ARCHITECTURE/STATE 일부가 아직 "v5(signed)"로 표기 — 실배포는 distbrake_v1(brake_only). delta_mode↔ckpt 불일치는 치명.

---

## 8. Open TODO (우선순위)

1. **a_2 cascade base 결정**: v8 균형275K vs 5f-sweep 중 택1 → i2_mem 6개 재생성 → f_2 재학습. [in-progress]
2. **i2_mem regen 후 verify_i2_live_vs_cache 재확인**(배포 불변식 cos 1.0). [배포 의존]
3. **ACC 강접근 escalation 보강**: pred vL_rel corr 0.84 한계 → 5f corr↑ 또는 closing-비례 라벨(ttc2) 검증. [in-progress]
4. **gate-stability clean 재산출**: 과거 flip@0.3 14.9% 는 sangam cache zero-fill artifact → clean i2_mem 으로 EMA/distill/beat flip 재계산. [대기]
5. **0523_validation_short 독립 held-out 화**(현재 random split leakage 우려). [방법론적 가치 큼]
6. **#655 채택 여부**: steer 0.0231 개선 확인 → cascade(i2_mem/f_2) 영향 검토 후 main 교체 판단.
7. backbone b0 교체: a_2-on-b0 가 v2s 미달 확인 → gate 안정화 매듭 후 재평가. [보류]
8. v7 재귀형 i_2(long temporal memory) — vL_rel 지각의 근본 강화(5f 가 임시해). [hypothesis]
9. legacy iy/v2 경로 정리, 규정 §1.5 reference line 해석. [정리]

---

## 9. 2026-05-26 (주간·야간 세션) — object separation 심화 · temporal ablation · bin80 · latency/backbone · SLURM

5/26 추가 세션. 큰 줄기: (1) a_2 object 인지를 FP↓·분포분리(hardneg/margin)와 gate operating-point sweep 로 매듭, (2) **temporal ablation 으로 frame 밀도(개수)가 far vL_rel std 의 진짜 lever 임을 확정**(span 확대 가설 반박), (3) far lateral 양자화 해소를 위한 **bin80(stride16 1D-merge)** 설계+구현, (4) latency/backbone(멀티프레임·EffNetV2-L) 검토로 bin80 방향을 역으로 확증, (5) GT dist 분포 confound(far=0521 편중) 발견, (6) SLURM/Singularity 클러스터 셋업.

### 9.1 a_2 object separation 심화 — hardneg / margin / both [confirmed-부분]

목적: detected-present FP 를 줄이고 obj logit 의 present/absent 분포를 분리. (in-lane 최난 negative 억제 + present-neg gap 강제.)

설계: v7b warm start, 3 변종 학습 — **hardneg**(OHEM, in-lane 최난 negative 만 골라 억제), **margin**(present−neg gap 강제), **both**(둘 다). objbin 평가 = FP@.5 / recall / far recall.

결과(objbin):

| 변종 | FP@.5 | recall | far recall | 결론 |
|---|---|---|---|---|
| hardneg | 26→**0.1%** | — | far ↓ **75%** | FP 거의 제거되나 far recall 붕괴 |
| margin | 23% | **90.5% 유지** | — | gap↑ 로 recall 보존하나 FP 여전히 높음 |
| both | hardneg 지배 | 70.8 | — | hardneg term 이 margin 압도 |

추가 진단: 5모델(v7b/baseline/hardneg/margin/both) 의 FP·FN 을 **(거리 × closing) 평면 산점도** + far-band(40-60m) FN × closing cross-tab. far 의 **61% 가 fast-approach(closing>5m/s)** 이나 TTC median 4-5s (즉 거리는 멀어도 시간 여유는 있음). status: hardneg 의 FP↓ vs far recall↓ trade-off [confirmed], 단독 채택은 §9.2 의 gate sweep 으로 보완.

### 9.2 gate operating point sweep — gate=hardneg @ logit −2.2 확정 [confirmed]

목적: hardneg 의 far recall↓ 를 gate threshold 조정으로 회복. (분포가 0/1 극단분리면 threshold 를 크게 낮춰도 FP 가 안 터짐.)

설계: hardneg gate threshold prob **0.574 → 0.10** sweep(약 8배 ↓, logit ≈ −2.2). absent 분포 p99=0.09 (극단 분리).

결과: threshold 를 8배 낮춰도 **FP 0.06 → 0.93%** (여전히 <1%), far recall **33.7 → 59.1%**, HIGH-miss **6.7 → 4.67%**. → **gate = hardneg @ logit −2.2 (prob 0.10) 확정.** closed_loop_node 의 `GATE_THRESH` 는 이미 env 화돼 있어 **코드 변경 불필요**. status [confirmed].

### 9.3 vL_rel 정확도 거리별 — 2-frame far bias 확인 [confirmed]

목적: detected-present 의 거리 bin 별 vL_rel error 진단(2f vs 5f).

결과:
- **v7b (2-frame)**: 20m+ 에서 systematic bias **+4 m/s**(closing 과소추정), far(40-60m) std **9.8**.
- **5-frame 모델들**: 전 거리 unbiased(bias ±0.5), std 만 거리 비례.

→ 2f 의 far closing 과소추정이 정량 확인됨(§1.3 의 corr 한계와 정합). status [confirmed].

### 9.4 ★ temporal ablation (span vs density) — 핵심 발견 [confirmed]

목적: vL_rel far std 를 줄이는 진짜 lever 가 **시간 span(0.5→1.0s)** 인지 **frame 밀도(개수)** 인지 분리. (사전 가설 = span 확대가 lever.)

설계: 2×2 ablation — span {0.5s, 1.0s} × frame {10, 20} + baseline(span 0.5s, 5f).

결과:

| 구성 | far vL_rel std | objF1 | 비고 |
|---|---|---|---|
| baseline (0.5s, 5f) | 9.8 | 72 | 기준 |
| span 0.5→1.0s 단독 | 3.2 → 3.0 | — | **거의 무효**(span 만 늘려선 안 됨) |
| **s10n20 (1.0s, 20f)** | 9.8 → **1.6** | 72 → **87.5** | vLrel_MAE·distMAE 반감, 전 지표 최선 |

→ **frame 밀도(개수)가 진짜 lever**(√N noise averaging). span 0.5→1.0s 단독은 무효. **사전 가설(span 확대가 lever)을 데이터로 반박.** status [confirmed], §9.5 bin80 ablation 의 frame 수 결정 근거.

### 9.5 bin80 (stride16 1D-merge) — 설계 + 구현 [confirmed 구현, ablation 학습 대기]

목적: far lateral 양자화 해소. stride32 출력은 W=40 column 이라 far 객체가 <1 column 으로 양자화됨 → stride16(W=80) feature 도입.

설계: stride32 readout(40) → interp(80) 의 **semantic** 위에, stride16(80 실측) **위치**를 **1D residual(α=0 init) merge**. α=0 시작 = interp baseline 과 동일(stride16 OFF), 학습으로 stride16 을 점진 ON. (α=0 출발이라 회귀 안전.)

구현:
- `A2ModelMultiFrameHR` (`a2_model.py`)
- `compute_i1_tok_cache.py --hr` (tokens_s16 80×160 추가 출력)
- `train_a2_multiframe.py --hr` (in-lane index 32:52, resume 시 `proj`→`proj32` rename + `lateral_pos` 40→80 interp)
- GT bin80 재생성(`prepare --n-bins 80`), cache `--hr` 재생성 완료.

검증: smoke 통과 — α=0 에서 stride16-invariant(=interp baseline), `proj16`/`alpha` grad 흐름 확인. status [confirmed 구현]. (실제 ablation 학습은 §9.9 SLURM 으로 제출 예정.)

### 9.6 멀티프레임 2/5/10 latency — sub-linear, 여유 충분 [confirmed]

목적: frame 수 증가의 latency 비용 측정(backbone amortized, fusion+head 만).

설계: `forward_tokens` batch=1, backbone 분리(KV 길이만 증가).

결과:

| frame | latency (ms) | KV 길이 배수 | latency 배수 |
|---|---|---|---|
| 2f | 0.73 | 1.0× | 1.0× |
| 5f | 1.18 | 3.9× | 1.6× |
| 10f | 2.13 | 8.8× | 2.9× |

→ KV 길이 3.9/8.8× 대비 latency 1.6/2.9× = **sub-linear**. closed-loop dt=100ms 대비 10f 도 **27~47× 여유**. → frame 수는 latency 가 아니라 **정확도/메모리 기준으로 결정**. status [confirmed].

### 9.7 EffNetV2-L 검토 — far 병목 해결 안 됨, ROI 없음 [confirmed 기각]

목적: backbone 을 키우면(v2_S → v2_L) far lateral 인지가 좋아지는가.

결과:
- latency L/S = **2.3×** (median 8.9 → 20.7ms). siamese 2회면 p99 ~100ms 로 실시간 마진 소진 → closed-loop 부적합.
- **결정적**: 두 모델 stride 동일([2,4,8,16,32]) → stride32 out (8,40) 불변 → **far lateral 병목은 v2_L 로 해결 안 됨**(키워봐야 semantic capacity 만 늘 뿐 양자화는 그대로).

→ ROI 없음. **far 는 stride16(bin80)이 정답** — backbone ablation 이 §9.5 bin80 방향을 **역으로 확증**. status [confirmed 기각].

### 9.8 GT dist 분포 발견 — far=0521 편중 confound [confirmed 데이터 한계]

목적/발견: max_dist=80m cutoff 지만 실데이터 분포가 데이터셋마다 다름.

| 데이터셋 | max dist | 50m+ 비율 |
|---|---|---|
| sangam_lead_0520 | 47.9m | **0%** (near-only) |
| sangam_lead_0521 | 60.7m | 15.2% (50-60m) |

→ far(50m+) 객체는 **0521 에만 존재**. 즉 §9.1·9.3·9.5 의 far 분석은 **0521 편중 confound** 이고, bin80 far 이점도 0521 의존. 160bin(stride8)은 60m+ 0.4% 라 검증 표본 부재 → **데이터로 검증 가능한 한계는 bin80(60m)까지**. status [confirmed 데이터 한계]. (MEMORY: feedback_experiment_design_review 의 confound 경계 사례.)

### 9.9 SLURM / Singularity 클러스터 셋업 [in-progress]

목적: bin80 ablation(80bin × {10f, 20f}) 을 클러스터로 학습.

구조:
- controller = slurm-master2, 제출/빌드 노드 = 3080×4 (166.104.167.165:2208), 실행 노드 = partition-{3090-intel, 3090-threadripper, 4090, A6000}, 공유 NFS = AILabSSD.
- 컨테이너 = Singularity sif (Docker → commit → tar → `singularity build`). `scc_train.sif` (torch2.4 cu118 + timm + numpy, `environment.yml`).
- staging = `AILabSSD/01_SlurmWorkspace/jiwon_a2bin80` (코드 + cache_hr + GT_bin80 + v7b ckpt).
- job script = `slurm/train_a2_bin80.sbatch` (partition-4090 / gpu:4090:1, `--bind AILabSSD`).

status [in-progress]: bin80 ablation(80bin × {10f, 20f}) 학습 **제출 예정**.

### 9.10 종합 — 5/26 결론

- a_2 object: **gate=hardneg @ logit −2.2** 로 FP<1% 유지하며 far recall 59% 회복(코드 변경 불필요, env). [confirmed]
- temporal: **frame 밀도가 far std lever** (span 아님). s10n20 이 전 지표 최선(far std 9.8→1.6, objF1 72→87.5). [confirmed, 가설 반박]
- far lateral: **bin80(stride16 1D-merge)** 가 정답(backbone 키우기는 stride 불변이라 무효). 구현·smoke 완료. [confirmed 구현]
- latency: 10f 도 27~47× 여유 → frame 수는 정확도/메모리로 결정. [confirmed]
- **confound 경계**: far 분석은 0521 편중, 검증 가능 한계 bin80(60m). [confirmed 데이터 한계]
- SLURM: bin80 ablation 제출 예정. [in-progress]

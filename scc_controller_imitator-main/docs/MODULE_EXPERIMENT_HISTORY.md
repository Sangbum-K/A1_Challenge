# 모듈별 실험 이력 — scc_controller_imitator

> 대상: A1 CHALLENGE 2026 SCC(Smart Cruise Control) imitation learning (카메라+ego → steer/pedal).
> 축: 파이프라인 5개 모델 모듈(seg backbone i_1 / a_2 / f_1 main / f_2 side / gate)별로 시리즈·날짜·수치를 재구성.
> 출처: `outputs_multi/**/train.log` · ckpt 파일명 · `data/ckpts/CURRENT_BEST.json` · `docs/EXPERIMENT_DIGEST_2026-05-22_26.md` · `docs/REPORT_*.md`. 본 문서는 read-only 수집물이며 새 학습/eval 을 수행하지 않았다.
> 날짜는 git commit date(YYYY-MM-DD 절대표기). 수치는 confirmed(train.log 인용) 와 hypothesis/추정 을 구분한다. 새 plot 은 만들지 않고 기존 viz png 경로만 임베드한다.

## 0. 파이프라인 (전제)

```
camera(curr,prev) → [seg backbone i_1: tf_efficientnetv2_s, fx640_v3, frozen] → 40×256 토큰
   → [a_2: 2-frame fusion, d_out=128] → i2_mem(40,128) + a2_logits(40,7: obj/vL_abs/vY_abs/vL_rel/vY_rel/prox/dist)
        ├ [f_1 main: Prof #654 SelfAttention, frozen] traj6+ego6 → steer, base_pedal (이미지 안 봄)
        └ [f_2 side: TransformerDecoder cross-attn] ego6+base_pedal + KV=i2_mem → pedal delta
   → [gate g] → final_pedal = clamp(base_pedal + g·delta, −1, 1)
```

핵심 불변식: live i2_mem == cache (seg=fx640_v3 + ego_mot=zeros + prev=200ms). 위반 시 cosine 0.80 → SCC/제동 붕괴. 종방향 lead 제동은 전적으로 f_2(image)+gate 담당(f_1 의 brake 는 traj-speed artifact).

모듈별 현행 배포 한 줄 요약(상세는 각 §):

| 모듈 | 현행 배포/best | 핵심 교훈·전환점 |
|---|---|---|
| seg backbone i_1 | fx640_v3 (v2_s, mIoU 0.921) | fx640 통합이 fx320 폐기. b0 경량화는 vehicle IoU −0.055 로 보류 |
| a_2 | v7b_d128 (2f) 배포 + 5f(a2_mf5f_acc) vrel 주입 | v5/v6 reg-loss bug(vL_rel std=0)→v7 fix. 프레임 밀도가 vL_rel lever |
| f_1 main | main_prof654.pth (frozen) | #654 frozen 이 정직 구조. #655(steer 0.0231) 채택 보류 |
| f_2 side | distbrake_v1 (예선) / predvrel_ttc(ACC, 본선후보) | closing 불신 → 거리-onset distbrake 가 신뢰가능. side 4L 우위 |
| gate | fixed_thresh(예선) → learned NN gate_head(이번) | learned_bias +3 drift→FP. soft-aggregation dead-end, 시간평활이 정답 |

---

## 1. seg backbone (i_1)

카메라 ROI(250×1280) → seg 4-class(bg/road/vehicle/wall) → features_only 토큰(40×256). a_2 의 frozen feature extractor. SAM mask pretrain GT.

| 시리즈 | 날짜 | 변경점 | 핵심 hyperparam | best 수치 | ckpt | viz |
|---|---|---|---|---|---|---|
| v2 (SAM SwinB+ensemble) | 2026-05-20 | SAM seg v2 GT 생성(SwinB+ensemble) | — | — (GT 단계) | — | — |
| v2s_full (smoke/full) | 2026-05-21 | v2_s 본학습 초기(통합 전) | tf_efficientnetv2_s | best_mIoU 0.731 (ep30, val_loss 0.456) [confirmed] | seg_pretrain_v2s_full_best_backbone.pt | viz 없음 |
| combined_v1 | 2026-05-21 | sangam+spd combined dataset | v2_s, warm-start | best_mIoU 0.892, best_val_loss 0.0970, vehicle IoU 0.76 [confirmed] | seg_pretrain_combined_v1_best_miou_backbone.pt | viz 없음 |
| fx640_v3 (배포) | 2026-05-21 | fx640 통합(speedway+sangam), fx320 폐기 | v2_s, lr 1e-4 / enc 5e-5, 13ep | best_mIoU 0.921 (ep12), val_loss 0.131, ious [bg.926/road.972/veh.864/wall.920] [confirmed] | seg_pretrain_fx640_v3_backbone.pt | ![seg fx640_v3 ep12](../outputs_multi/seg_pretrain_fx640_v3/viz_progress/ep12.png) |
| lite_b0 (경량화 분석) | 2026-05-26 | EfficientNetV2-b0 교체 검토 | b0, 15ep | best_mIoU 0.897(−0.024), vehicle IoU 0.811(−0.055), latency 2.14× (5.01→2.34ms) [confirmed] | seg_pretrain_lite_b0/best_miou_backbone.pt | viz 없음 (분석은 REPORT_seg_backbone_lite.md) |

현행 배포: fx640_v3 (v2_s, mIoU 0.921). CURRENT_BEST.json 등재.

핵심 교훈:
- fx640 통합이 fx320(MultiAgnet) 폐기 결정(2026-05-21). 학습 데이터는 fx=640 only.
- b0 경량화: i_1 토큰경로 latency 2.14× 이득이나 저하가 vehicle class(lead car 직결)에 집중(−0.055). b1/b2/b3 는 b0 에 속도-dominated 라 중단, mobilenetv3 는 last ch 960 caveat. **a_2-on-b0 ablation 결과(§2 모듈교차)가 전 지표 v2s 미달이라 교체 보류**.
- backbone 키우기(v2_L)도 stride 불변([2,4,8,16,32])이라 far lateral 양자화 해결 안 됨 → §2 bin80 으로 우회.

대표 viz: `outputs_multi/seg_pretrain_fx640_v3/viz_progress/ep00_init.png` ~ `ep13.png` (14장, epoch별 seg overlay).

---

## 2. a_2 (obj + vL_rel perception)

seg 토큰에서 per-bin [obj, vL_abs, vY_abs, vL_rel, vY_rel, prox, dist] 7채널 회귀. obj=앞차 인지, vL_rel=상대속도. 모든 학습은 prior best 를 init 으로 warm start.

### 2.1 본류 시리즈 (focal / reg-fix / fc_proj 256→128 / 분포보정)

| 시리즈 | 날짜 | 변경점 | 핵심 hyperparam | best val loss | ckpt | viz |
|---|---|---|---|---|---|---|
| v1 | 2026-05-20 | a_2 GT multi-bin(4-corner projection) + 초기 학습 | w_obj/w_reg=1, scale_v20/dist40 | 0.1199, obj_acc 0.990 [confirmed] | a2_v1_best_loss.pt | viz 없음 |
| fx640_v3 | 2026-05-21 | fx640 + close-lead 가중(close_thr30 w3) + prev_dtick20 | v1 resume, lr1e-4 | (loss scale 별도) [confirmed 존재] | (a2_fx640_v3) | viz 없음 |
| fx640_v3_local | 2026-05-21 | 로컬 캐시 경로 정합 | 동일 | best_val 0.0172, obj_acc 1.000 [confirmed] | (CURRENT_BEST a2_head) | viz 없음 |
| fx640_v4_noleadboost | 2026-05-22 | no-lead boost 제거 + lead oversample (FP↓) | v3 resume | best_val 0.0185, obj_acc 1.000 [confirmed] | a2_fx640_v4_best_loss.pt | viz 없음 |
| v5_d128 | 2026-05-25 | **fc_proj 256→128** (i2_mem 차원축소), cache-only, enc freeze | lr3e-4, d_out128, 5ep | 0.5882 (digest 인용, loss scale 변경) [confirmed-digest] | a2_v5_d128_best_loss.pt | ![a2 v5 init](../outputs_multi/a2_fx640_v5_d128/viz_progress/ep00_init.png) |
| v6 | 2026-05-25 | focal loss (obj imbalance) | focal γ2 | 0.4233 [confirmed-digest] | a2_v6_d128_focal_best_loss.pt | viz 없음 |
| v7 (reg-fix) | 2026-05-25 | **reg-loss normalize + obj_mask 누락 버그 fix** | focal | 0.2052 [confirmed-digest] | a2_v7_d128_regfix_best_loss.pt | viz 없음 |
| v7b (배포) | 2026-05-25 | focal **alpha 0.75** (≈1−pos_rate) | focal α0.75 γ2 | 0.2048 [confirmed-digest] | a2_v7b_d128_alpha075_best_loss.pt | viz 없음 |
| v7c | 2026-05-25 | focal alpha 0.9 | α0.9 | 0.2015 (실익 미미, v7b 유지) [confirmed-digest] | a2_v7c_d128_alpha09_best_loss.pt | viz 없음 |
| v8 | 2026-05-25 | Gaussian-NLL + LDS (13ch mean+σ) | NLL+LDS nbins20 | 0.2170 (이득 없음) [confirmed-digest] | a2_v8_d128_gaussNLL_lds_best_loss.pt | viz 없음 |
| v9 | 2026-05-25 | LDS only | LDS | 0.2136 (v7b 미달, dead-end) [confirmed-digest] | a2_v9_d128_lds_only_best_loss.pt | viz 없음 |

배포: v7b_d128 (2f). v8/v9 분포보정은 dead-end.

핵심 교훈(버그·전환점):
- **v5/v6 reg-loss bug**: normalize+obj_mask 누락 → vL_rel pred std=0(회귀 죽음). v7(2026-05-25)에서 fix → 회귀가 비로소 학습. MEMORY(a2_v5_v6_reg_loss_bug)·incident 등재.
- v7b 절대 corr 은 낮음(2-frame 한계) → §2.2 multi-frame 으로 근본 해결.
- yaw_shift(hand-crafted token roll) 무효: a2_2f_yaw == a2_2f_noyaw (둘 다 best 0.0843/0.0779 동급, cache 경로는 d_yaw=0 하드코딩이라 애초에 안 씀). [dead-end]

### 2.2 multi-frame — vL_rel 지각 근본 해결 [confirmed, 최대 결과]

목적: 2-frame v7b 의 vL_rel corr 0.34 한계를 프레임 수로 해결.

| 시리즈 | 날짜 | 변경점 | best val loss(자체) | vL_rel corr / corr_near | dist MAE | obj F1 |
|---|---|---|---|---|---|---|
| v7b (2f 기준) | 2026-05-25 | 2-frame | 0.2048 | 0.337 / 0.238 | 6.81 | 0.657 |
| multiframe_3f | 2026-05-26 | 3f (offset 20,40) | best 0.0838 [confirmed] | 0.801 / 0.745 | 2.92 | 0.782 |
| multiframe_4f | 2026-05-26 | 4f (10,20,40) | best 0.0776 [confirmed] | 0.807 / 0.746 | 2.78 | 0.784 |
| 5f (배포 vrel 주입) | 2026-05-26 | 5f (10,20,30,40,50) | — | **0.836 / 0.777** (sweep), 독립 3set 0.920/0.842 | 2.66 (3set 1.52) | 0.789 (3set 0.845) | a2_mf5f_acc.pt |

corr/MAE/F1 은 digest §1.3 인용([confirmed-digest], `a2/mf_sweep_eval.json`). best val loss(0.0838/0.0776)는 multiframe train.log 직접 인용[confirmed].

결론: 프레임↑ → vL_rel corr 단조 개선(0.34→0.84~0.92), dist/obj 동반 개선. 5f(span 500ms) 채택. 배포 라이브는 i2_mem/gate=v7b 2f 유지, **vrel scalar 만 5f 로 주입**(학습 분포 일치).

### 2.3 object separation / temporal ablation / bin80 (2026-05-26 세션)

| 시리즈 | 날짜 | 변경점 | 결과 | status |
|---|---|---|---|---|
| objsep hardneg/margin/both | 2026-05-26 | obj FP↓ 분포분리(OHEM/margin) | hardneg FP 26→0.1%이나 far recall ↓75%; margin recall 90.5% 유지하나 FP 23% [confirmed-digest] | confirmed-부분 |
| gate sweep | 2026-05-26 | hardneg threshold 0.574→0.10(logit −2.2) | FP 0.06→0.93%(여전히<1%), far recall 33.7→59.1% [confirmed-digest] | confirmed, GATE_THRESH env |
| tspan ablation (s/n) | 2026-05-26 | span{0.5,1.0s}×frame{10,20} | far vL_rel std: baseline 9.8 → s10n20 **1.6**, objF1 72→87.5. span 단독은 무효(3.2→3.0) [confirmed-digest] | confirmed, 가설(span) 반박 |
| bin80 s10n10 / s10n20 | 2026-05-26 | stride16(W=80) 1D-residual merge(α0 init) far lateral 양자화 해소 | far(40-60m) vL_rel std: 40bin 3.0, s10n10 1.9, **s10n20 0.96** [confirmed-digest]. train.log 자체는 done 줄 미기록(진행/미완) | in-progress |

bin80 train.log(`outputs_multi/a2_bin80/bin80_s10n10`, `bin80_s10n20`)는 args 만 있고 best done 줄 미확인 — std 수치는 digest 인용. prev_offsets s10n10=10,20..100(10f) / s10n20=5,10..100(20f).

대표 viz (분석 plot, 2026-05-26):
- ![a2 objsep gate sweep](../outputs_multi/a2_objsep/gate_sweep_hardneg.png)
- ![a2 objsep FP rate heatmap](../outputs_multi/a2_objsep/fp_rate_heatmap.png)
- ![a2 objsep vrel err by dist](../outputs_multi/a2_objsep/vrel_err_by_dist.png)
- ![a2 objsep FN dist×closing](../outputs_multi/a2_objsep/fn_dist_closing.png)
- ![a2 tspan vrel err by dist](../outputs_multi/a2_tspan/vrel_err_by_dist.png)

### 2.4 부가 (single objdist / b0 ablation)

| 시리즈 | 날짜 | 변경점 | 결과 | status |
|---|---|---|---|---|
| single_objdist_v1 | 2026-05-26 | gate 전용 경량 단일프레임 obj/dist 헤드(n_ch=2) | 학습됨, closed_loop fusion 경로와 달라 **미배포** | in-progress |
| a2-on-b0 ablation | 2026-05-26 | seg b0 위 a_2(5f) | b0 obj F1 0.812 / vLrel corr 0.888 / dist MAE 2.02 vs v2s 0.845 / 0.920 / 1.52 — 전 지표 v2s 미달 [confirmed-digest] | confirmed, backbone 교체 보류 근거 |

---

## 3. f_1 (main, steer + base_pedal)

이미지 비참조. traj6+ego6 → steer + base_pedal. controller_imitator #654 Professor SelfAttention(SelfAttentionControlPredictor_Professor) frozen 이 토대.

| 시리즈 | 날짜 | 변경점 | best 수치 | ckpt | viz |
|---|---|---|---|---|---|
| #654 Professor (frozen 토대) | (controller_imitator) | SelfAttention, frozen main | val 0.0355 (steer) [confirmed-MEMORY] | main_prof654.pth | viz 없음 |
| f1_cached_v1 | 2026-05-22~23 | cached f_1 학습(prebake) | val 0.2751 (steer 0.00° ax 2.75) [confirmed] | f1_cached_v1/best_loss.pt | viz 없음 |
| f1_cached_v2_k500p_5cyc_safe | 2026-05-23 | k500 prebake 5cycle | val ~0.343 (steer 1.92° ax 1.51) [confirmed] | f1_cached_v2_k500p_5cyc_safe_best.pt | viz 없음 |
| f1_cached_v3_k500p_prev200 | 2026-05-23 | prev=200ms align | best_val 0.3429 (steer 1.91° ax 1.52) [confirmed] | (f1_cached_v3) | viz 없음 |
| iy_spd_..._v0237_noreload (CURRENT_BEST f1) | 2026-05-23 | spd+sangam ci_aug, no-reload | val 0.145, steer MAE 0.77°, ax 0.68 [confirmed-registry] | outputs_multi/iy_spd_sangam_ciaug_v0237_noreload/best_loss.pt | viz 없음 |
| v25c main-alone finetune | 2026-05-24 | main 단독 재학습(lr3e4 decmax) | best 0.3706, val 0.3710 (steer 3.01° ax 0.70) [confirmed] | iy_v25c_best_loss.pt | viz 없음 |
| #655 finetune | 2026-05-23~24 | v0301_0303 ft654, coslinear | steer val 0.0231(#654 0.0355 개선), ax 0.1224 [confirmed-digest] | main_prof655.pth | viz 없음 |
| v25h_A image-aware | 2026-05-24 | main 에 이미지 주입 시도 | best 0.1649 (steer 0.75° ax 0.90) [confirmed] | iy_v25h_A_best_loss.pt | ![v25h_A openloop scatter](../outputs_multi/openloop_v25h_A_scatter.png) |

현행 배포: main_prof654.pth (frozen). #655(steer 0.0231 개선) 는 cascade(i2_mem/f_2) 영향 검토 후 채택 판단 — 보류.

핵심 교훈:
- f_1 의 brake 는 traj-speed artifact라 inference 에서 앞차 제동 불가 → 제동을 f_2(image)+gate 로 이관(prof635 재설계의 동기).
- v25h_A 처럼 main 에 이미지 넣으면 steer val 매우 낮으나 train/val 분포 의존 우려 → openloop scatter 만 산출, 배포 안 함.

---

## 4. f_2 (side, pedal SCC)

ego6+base_pedal + KV=i2_mem(cross-attn) → pedal delta. image 기반 lead 제동 전담. steer 는 항상 frozen main 값(3.43~3.50° MAE).

### 4.1 iy 계열 재설계 (v18 → v25) — prof635 흡수 전 과도기

| 시리즈 | 날짜 | 변경점 | best_val(자체 scale) | ckpt | viz |
|---|---|---|---|---|---|
| v18b/c sidebranch fixinit | 2026-05-23~24 | ControlNet side branch + init fix | v18c best 0.1078 [confirmed] | iy_v18c_best_loss.pt | viz 없음 |
| v19 ax-only side | 2026-05-24 | side 를 ax-only 로 | best 0.1045 [confirmed] | iy_v19_best_loss.pt | viz 없음 |
| v20 a2feat | 2026-05-24 | a_2 feature 주입 | best 0.0892 [confirmed] | iy_v20_best_loss.pt | viz 없음 |
| v21a a2 late inject | 2026-05-24 | a_2 late injection | best 0.0891 [confirmed] | iy_v21a_best_loss.pt | viz 없음 |
| v22 asym brake w3 | 2026-05-24 | asymmetric brake weight 3 | best 0.1264 / close 0.1148 [confirmed] | iy_v22_best_loss.pt | viz 없음 |
| v23 main_noimg+side cross | 2026-05-24 | main_no_image + side cross-attn redesign | close 0.4549 / far 0.4261 [confirmed] | (iy_v23) | viz 없음 |
| v23e phase1.5 fix2 | 2026-05-24 | freeze 버그 2회 fix(v23c/d/e) | best_val 0.2353 (steer 0.76° ax 1.60) [confirmed] | iy_v23e_best_loss.pt | viz 없음 |
| v24 ttc-augment | 2026-05-24 | TTC augment(in-lane bin16-25 gating) | best_val 0.2112 (steer 0.65° ax 1.47), close ax 2.20 [confirmed] | iy_v24_best_loss.pt | viz 없음 |
| v25 2phase_ttc | 2026-05-24 | 2-phase(main frozen→joint)+ttc, scheduler bug fix | best 0.2565, close 0.2701 far 0.2509 [confirmed] | (iy_v25) | viz 없음 |
| v25d side_a2_ttc | 2026-05-24 | v25c resume + a2feat + ttc side | val 0.3707 (steer 3.00° ax 0.71) [confirmed] | iy_v25d_best_loss.pt | viz 없음 |
| v25h_B / B2 (side 4L) | 2026-05-24 | side cross-attn layer 수(B0=1L…B4=4L) | B 0.6860 / B2(4L) 0.5923 [confirmed] | iy_v25h_B_best_loss.pt, iy_v25h_B2_best_loss.pt | viz 없음 |
| v25k_B4_gttc_acc | 2026-05-26 | GT inlane + ACC ttc, side 4L | best 0.5749 (ax 1.85) [confirmed] | (iy_v25k_B4) | viz 없음 |
| v25k_B4_gttc_accgate | 2026-05-26 | + ACC gate (예선 사용, val 0.5808 보고치) | train.log ep01 에서 멈춤(미완/진행), 본문 val 0.5808 은 외부보고치 [hypothesis] | iy_v25k_B4_gttc_accgate_best_loss.pt | viz 없음 |
| v26_Bminacc | 2026-05-26 | close-approach ACC 별실험 | close-approach(n741) ax_err 4.694, brake_dir 85% [confirmed] | (iy_v26) | viz 없음 |

iy 계열 best_val 은 label scale 이 버전마다 달라 cross-compare 불가(같은 label 류 내에서만 비교). B 숫자 = side cross-attn layer 수. iy 자체는 prof635 로 수렴 후 폐기 진행.

### 4.2 prof635 side (f_1 #654 frozen + Side SCC cross-attn) — 현행 구조

Prof635SideModel: q=side_state_embed(ego6)+side_pedal_embed(base_pedal), KV=img_proj(i2_mem), side_dec(TransformerDecoder 4L d128), pedal_head. loss=L1(final_pedal). trainable ≈1.08M. **val_pedal 은 라벨 분포가 달라 라벨류 내에서만 비교.**

| 시리즈 | 날짜 | 변경점(라벨/구조) | best val_pedal | ckpt | viz |
|---|---|---|---|---|---|
| v1 | 2026-05-25 | side 초기 | 0.0127 [confirmed] | (prof635_side_v1) | viz 없음 |
| v5 _ttc resgate notraj | 2026-05-25 | closing-ACC signed, residual gate, no-traj | 0.0237 (steer 3.50° frozen) [confirmed] | prof635_side_v5_best.pt | viz 없음 |
| v6 _ttc +DriverB | 2026-05-25 | + DriverB | 0.0443 [confirmed] | (prof635_side_v6) | viz 없음 |
| **distbrake_v1 (예선 배포)** | 2026-05-26 | **거리-onset, closing 비의존**, brake_only(−2·sigmoid), gate open | **0.5003** [confirmed] | prof635_side_distbrake_v1.pt | viz 없음 |
| distbrake_randbase | 2026-05-26 | base 랜덤화(over-fit 완화) | 0.5005 (영향 미미) [confirmed] | (prof635_side_distbrake_randbase) | viz 없음 |
| predvrel_ttc (본선 ACC 후보) | 2026-05-26 | 5f pred vL_rel 주입 + ttc ACC, brake_only | 0.5118 [confirmed] | prof635_side_predvrel_ttc.pt | viz 없음 |
| predvrel_ttc2 | 2026-05-26 | closing-비례 라벨 재설계 | 0.5111 [confirmed] | prof635_side_predvrel_ttc2.pt | viz 없음 |
| predvrel_signed | 2026-05-26 | signed delta | 0.0450(별scale, 비권장) [confirmed] | prof635_side_predvrel_signed.pt | viz 없음 |
| gtvrel_match_ttc / gtvrel_signed | 2026-05-26 | GT vrel 주입(상한 진단) | 0.5098 / 0.0428 [confirmed] | (prof635_side_gtvrel_*) | viz 없음 |
| gtv_1L / gtv_4L | 2026-05-26 | side layer 1L vs 4L | 0.4930 / 0.5247 (4L 우위) [confirmed] | (prof635_side_gtv_*) | viz 없음 |

현행 배포: distbrake_v1 (예선, brake_only). 본선 ACC 후보 = predvrel_ttc (5f vrel 주입).

핵심 교훈(버그·전환점):
- **closing modulation 불신**: a_2 vL_rel ↔ GT closing corr −0.279, f_2 delta ↔ a_2 vL_rel +0.023. GT vrel 주입 격리실험은 f_2 가 정확 modulate → 병목은 f_2 capacity 가 아니라 a_2 지각.
- 그래서 closing 비의존 distbrake("가까우면 제동")가 예선 배포 기본. ACC(pred-vrel)는 약/정속 구분은 되나 강접근 escalation 포화(corr 0.84 한계).
- **ttc-augment NO-OP 버그**: B 경로(i2_cache)에서 inlane 계산이 a2_cache path 에만 있어 ttc-augment 가 무효. GT inlane cache(inlane_gt_*.npz)+dataset wire 로 fix. MEMORY(ttc_augment_inlane_gt).
- side layer 4L 이 1L 우위 → 4L 채택.
- delta_mode↔ckpt 불일치(signed vs brake_only)는 치명 — hotfix 사례 있음(2026-05-25 f_2 delta_mode 미전달).

---

## 5. gate (lead-presence)

no-lead 에서 FP brake 제거(가속 불가 방지) + lead 제동 유지. g 로 f_2 delta 를 변조: final_pedal = base_pedal + g·delta.

| 시리즈 | 날짜 | 변경점 | 결과 | status |
|---|---|---|---|---|
| learned_bias | (~2026-05-25 이전) | gate_bias learned | supervision 없이 +3.02 drift → no-lead g≈0.92 → FP brake(가속 불가) [confirmed-digest] | dead-end |
| fixed_thresh (B안, 예선) | 2026-05-25 | g=sigmoid((a2 inlane obj_amax − 0.3)·5) rule | no-lead brake 24→0%, lead 제동 100% 유지, gate F1 0.728(P0.85/R0.64), 접근 recall 0.90 [confirmed-digest] | confirmed |
| gate sweep (hardneg @ logit −2.2) | 2026-05-26 | threshold 0.574→0.10 | FP 0.06→0.93%(<1%), far recall 33.7→59.1% [confirmed-digest] | confirmed (GATE_THRESH env) |
| EMA(0.7) 평활 | 2026-05-26 | 시간평활 기본 | 출발불가(pedal 깔짝거림) → EMA 복원으로 해결 [confirmed-digest] | confirmed |
| learned temporal filter (distill/beat) | 2026-05-26 | 규정-clean in-graph EMA 대체 | 평활 옵션 추가 | in-progress | learned_temporal_gate_mlp_distill.pt, learned_temporal_gate_mlp_beat.pt |
| soft-aggregation (mean/topk/logsumexp) | 2026-05-26 | 공간평균 | 실데이터 FP 92% 폭발/recall 34%(lead 가 1-bin 국소라 희석) [confirmed-digest] | dead-end |
| **learned NN gate_head MLP (이번)** | 2026-05-26 | rule gate → learned NN gate_head MLP (규정 clean) | val_pedal ≈0.49 (train.log best 0.4938, ep17~19 진행) [confirmed], 보고치 0.49 부합 | in-progress(이번 학습) | prof635_side_nngate_v1 |

핵심 교훈: learned_bias 의 supervision-free drift → FP. soft-aggregation 은 공간평균이라 1-bin 국소 lead 신호를 희석(dead-end) → **평활은 시간축(EMA/temporal MLP)이 정답**. learned NN gate_head 는 규정 clean(rule gate 잔재 제거, commit e4f3060/017e8b0 2026-05-26) 목적으로 이번 세션 도입.

note: `prof635_side_nngate_v1/train.log` 의 best 는 val_pedal 0.4938(steer 3.43° frozen)이며 gate 전용 F1/recall 별도 metric 은 train.log 에 미기록 → val_pedal 만 인용.

---

## 부록: 수치 출처·미수집 정리

- confirmed = train.log 직접 인용(seg/a_2 본류/iy/prof635/f1 cached best 줄). confirmed-digest = EXPERIMENT_DIGEST 인용(corr/F1/std 등 eval JSON 기반, 별도 eval 산출물). confirmed-registry/MEMORY = CURRENT_BEST.json / MEMORY 인용.
- **train.log 에서 직접 best 줄을 못 채운 항목**:
  - a2_bin80(s10n10/s10n20): train.log 에 args 만, done/best 줄 미기록 → std 수치는 digest 인용 (학습 진행/미완 추정).
  - iy_v25k_B4_gttc_accgate: train.log 가 ep01 에서 멈춤(미완 또는 별경로) → val 0.5808 은 task 본문 보고치(hypothesis), 자체 best 줄 없음.
  - a2_fx640_v3 / v5_d128 등 일부: train.log 에 [done] best 줄 없이 args 만(단편) — loss 는 digest/CURRENT_BEST 교차 인용.
- **viz 임베드 6장**(존재 확인): seg fx640_v3 ep12, a2 v5 init, a2 objsep 4종(gate_sweep/fp_heatmap/vrel_err/fn_dist_closing), a2 tspan vrel_err, f_1 v25h_A openloop scatter. 대부분의 iy/prof635 학습 폴더는 viz_progress png 가 없음(viz 없음 표기).
- seg fx640_v3 는 viz_progress 14장(ep00~ep13) 존재. eval_open_loop 폴더에 MultiAgnet sangam open-loop overlay 8장 + summary.json 존재(fx320 폐기 데이터라 본문 미임베드).

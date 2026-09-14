# 실수·수정 + 데이터 검증 종합 리포트 (2026-05-24 ~ 25)

SCC Controller Imitator — B 모델(이미지 기반 SCC longitudinal)이 closed-loop 에서 "앞차 접근 시 감속 안 함" 문제를 추적한 기록. **Part 1** 실수/수정, **Part 2** 데이터 검증 활동(목적·방식·결과).

---

# Part 1 — 실수 및 수정 리포트

영향 큰 순. 각 항목: 증상 → 근본원인 → 정정 → 교훈.

## 1. ⭐⭐ ttc-augment 가 B 모델에서 NO-OP (최대 근본원인)
- **증상**: B0/B1/B0ttc/B1ttc 모두 `--ttc-augment` 켰는데 brake 학습 안 됨. closed-loop 에서 앞차 접근 시 +9.79 m/s² 가속.
- **근본원인**: `scc_synth_iy_dataset.py` 의 in-lane(bin16-25) dist/closing 계산이 `if a2_cache_path:` 블록 **안에만** 존재. B 학습은 `--i2-cache` 만 쓰고 `--a2-cache` 안 씀 → `inlane_dist=inf` 유지 → `_apply_ttc_augment` 의 approach mask 항상 False → **ttc 한 번도 fire 안 함**. brake target = raw prebake (근거리+접근에서 +1.07 가속).
- **정정**: `build_inlane_gt_cache.py` 신규 — aux GT 에서 frame별 inlane_dist(GT dist)+closing(-GT vL_rel) 계산. `--inlane-gt-cache(-extra)` → dataset `inlane_gt_cache_path` (cache 무관). v25k-B4 에서 70546/72519 frame inlane populate 확인.
- **교훈**: flag 켜짐 ≠ 동작. 핵심 augment 는 fire rate 를 로그/검증으로 확인.

## 2. train_a2_v5_from_cache.py regression loss 결함 (a_2 v5/v6)
- **증상**: a_2 v5/v6 회귀채널(vL_rel 등) pred std=0 (모든 곳 0), closing recall 0%.
- **근본원인**: `loss_reg = F.l1_loss(logits[...,1:], target[...,1:])` — (1) target normalize 없음 (2) obj_mask 없음 → 90% 토큰(obj=0,reg=0) 이 "모든 곳 0 출력" 학습 강제.
- **정정**: v7 — target/scales normalize + obj_mask. pred std 0→0.3+, DriverB vL_rel corr +0.40.

## 3. --ttc-augment flag 누락 (v25h+ launcher)
- **증상**: v25d/e/f 의 6개 ttc flag 이 v25h/i/j 에서 사라짐.
- **근본원인**: v25h 부터 launcher 재작성 시 drop.
- **정정**: 복원. (단 #1 때문에 복원해도 효과 없었음 → #1 해결 후 실제 작동)

## 4. focal_alpha=0.5 무의미
- **근본원인**: `α·t+(1-α)(1-t)`, α=0.5 = pos/neg 동일가중 = 일반 BCE.
- **정정**: α=0.75(v7b)/0.9(v7c). pos rate ~5% → 적정 ≈ 1-pos_rate ≈ 0.9.

## 5. 내 eval 스케일 오류 → "GT 손상" 오판
- **실수**: GT(raw m/s)에 ×20 잘못 곱함 → range [-956,+558] (불가능) → "GT 망가짐" 결론.
- **정정**: ×20 제거. GT 정상([-38,+13]). 진짜 문제는 모델/loss.

## 6. 3D scatter 를 raw target 으로 색칠 (사용자 지적)
- **실수**: scatter 색을 augment **전** raw 로 칠함. 모델이 학습하는 건 augment **후**.
- **정정**: ttc-augment 적용 target 으로 재생성.

## 7. ttc 분석에 predicted dist 사용 (사용자 지적)
- **실수**: ttc trigger 에 a_2 추론 dist 사용. 근거리 과대평가(true 3.8m→pred 8.8m) → closing 음수 오판 → ttc miss.
- **정정**: GT aux dist 사용 + closing=-GT vL_rel 직접.

## 8. gated 재학습 시 pre-gate ckpt 미resume (사용자 지적)
- **실수**: TTC-gating 적용 후 재launch 때 B2 에서 resume (pre-gate v25k-B4 의 5ep brake 학습 버림).
- **정정**: pre-gate ckpt 복사 후 거기서 resume.

## 9. ACC ttc 원거리 과도 brake
- **증상**: 40m 에서 closing 11(TTC 3.6s) 인데 -6.6~-9.5 (여유 있는데 강함).
- **근본원인**: a_gap 의 `kclo·closing` 항이 거리/TTC 무관 상수.
- **정정**: closing 항을 TTC 로 gating (`--ttc-acc-kclo-ttc-decay`). TTC>2s 부터 감쇠, 5s 에서 0. 40m@11: -6.6→-3.0, 근거리/긴급 유지.

## 10. 안 건드리기로 한 것 (사용자 판단)
- **ax 분산(std)**: <5m std 8-13 고분산이지만 **raceline 가속 계획 포함** → override 안 함. main(Professor)이 트랙 가속, side 는 lead-brake 만.
- **a_2 perception 추가 개선**: v8(NLL)/v9(LDS) 모두 vL_rel/dist 개선 실패 → v7b 동결, 추가 투자 중단. (2-frame+perspective 한계)

## 11. 운영 실수
| 실수 | 정정 |
|---|---|
| `pkill -f "패턴"` 이 자기 bash 죽임 (exit 144) | PID 직접 kill |
| npz member loop 반복 access → 매번 재압축(수분 hang) | loop 전 1회 추출 |
| f-string backslash → SyntaxError | 변수 분리 |
| matplotlib 한글 깨짐 | 축은 영문 |
| print buffering | `python -u` |
| 전체 135K aux 매번 로드 (2-3분) | val_sample_5k.npz 사전화 (213KB, 즉시) |

## 12. 명명 규칙 변경
- B 모델 숫자 = side cross-attn layer 수. B0(1L)→**B1**, B1(4L)→**B4**.

---

# Part 2 — 데이터 검증 활동 (목적·방식·결과)

문제 추적 과정에서 수행한 검증. 각 항목: **목적** / **방식(형태)** / **결과**.

## A. a_2 perception (인지) 검증

### A-1. vL_rel head 정확도 (v6→v7)
- **목적**: a_2 가 상대속도(vL_rel)를 맞추는지. obj 인지는 되는데 속도는?
- **방식**: i_2 cache 의 a2_logits[...,3] (예측) vs aux_targets GT[...,3] 비교. corr/MAE/sign, closing detection recall. DriverB/sangam.
- **결과**: v6 pred std=0 (완전 죽음, #2 버그). v7 fix 후 DriverB corr +0.40, sign 60-74%, but 크기 압축(pred std 2 vs GT 5).

### A-2. GT vL_rel 분포 sanity
- **목적**: GT 자체가 정상인지 (내 eval 이 [-956,+558] 보임).
- **방식**: aux GT[...,3] raw 값 히스토그램/range, obj 토큰만. 3 dataset.
- **결과**: GT 정상 ([-9,+9]~[-38,+13]). 내 ×20 스케일 오류였음(#5).

### A-3. a_2 v7/v7b/v7c 3-way
- **목적**: focal α (0.5/0.75/0.9) 가 obj recall/precision 에 미치는 영향.
- **방식**: 동일 eval, α 별 obj rec/prec + vL_rel corr.
- **결과**: α↑ → recall↑(47→76%) precision↓(83→44%). v7b(63/66%) 균형 채택. vL_rel corr 은 α 무관 동일(0.40) — focal 은 obj 만 영향.

### A-4. grad-cam: vL_rel 공간 attention
- **목적**: 상대속도 계산 시 실제로 차(in-lane bin)를 보는가.
- **방식**: lead bin 의 vL_rel 출력 → 입력 token grad backprop → curr/prev bin별 grad 히트맵. 12 sample.
- **결과**: lead-bin±3 집중도 curr 47% / prev 54% → **차 보고 양 프레임 사용(메커니즘 정상)**. but 크기 예측은 tail collapse.

### A-5. grad-cam overlay: 실제 차 존재 확인
- **목적**: heatmap 의 bin 에 진짜 차가 있나 (GT projection 정확성).
- **방식**: raw CAMERA_1 ROI 이미지 + grad peak/GT lead bin overlay. 8 frame.
- **결과**: (사용자 육안) cyan(GT)에 차 있고 grad peak 일치 → **GT projection + 공간인지 정상**. 문제는 크기.

### A-6. dist channel 정확도 + 근거리 세분
- **목적**: brake 의 실제 입력인 dist 채널이 (특히 <20m) 정확한가.
- **방식**: in-lane lead dist 예측 vs GT, 거리 bucket(0-5/5-10/.../50-80) MAE/corr. v7b/v8/v9.
- **결과**: corr 0.70 (vL_rel 0.40 보다 나음). but **<5m 과대평가 (true 3.8m→pred 8.8-14m)**. v8/v9(loss tricks) 개선 실패.

### A-7. 근거리 <20m vL 정확도 (5k sample)
- **목적**: brake-critical zone 의 속도 정확도.
- **방식**: in-lane lead, 0-5/5-10/10-15/15-20m bucket corr/MAE/sign.
- **결과**: MAE 3.5m/s, corr 0.16-0.44(best 5-10m), sign 60-74%, 압축. "방향은 대충, 크기 불신".

## B. brake target (정답) 검증

### B-1. GT ax ↔ (vL_rel/dist) 상관
- **목적**: target_ax 가 시각단서(접근/거리)와 인과 있는 학습가능 신호인가.
- **방식**: prebake target_ax[0] vs aux vL_rel/dist corr. DriverB/sangam, obj-frame.
- **결과**: DriverB +0.49(시각 일치) but sangam -0.1(raceline brake, 시각무관). → 데이터 모순 의심(후에 #1 로 정정).

### B-2. brake target ttc 적용 전후
- **목적**: 근거리+접근의 실제 학습 target 이 brake 인가.
- **방식**: in-lane 근거리+접근 sample 의 raw vs ttc-augment(safe_decel) target_ax.
- **결과**: raw +1.07(가속!) → ttc -3.22. **ttc 적용 시 정상**. (단 #1 로 실제론 no-op 였음)

### B-3. 비-inlane 오염 정량
- **목적**: brake target 중 image 로 설명 불가한(in-lane lead 없는) 비율.
- **방식**: brake(ax<-1.5) sample 중 in-lane bin obj 있는 비율.
- **결과**: DriverB 44% / sangam_0521 56% 오염 (raceline/신호/회전). ttc 는 in-lane 만 고치므로 나머지 raw 오염.

### B-4. 초근거리 가속 원인 진단
- **목적**: voxel 의 <5m+접근 +6.9 가속 원인.
- **방식**: GT-true <5m+접근 234 token 의 raw ax(+) 비율, ttc fire 비율, predicted vs GT dist.
- **결과**: raw 가속 94%, ttc fire 7%, **predicted dist 8.8 vs GT 3.8 (+5 과대)** → closing -3.48 오판 → ttc miss → 가속 생존. (#7 → GT inlane 으로 정정)

## C. 분포 시각화 (scatter / voxel)

### C-1. 3D scatter (dist × bin × closing, 색=ax)
- **목적**: brake 결정이 거리/bin위치/접근속도와 어떻게 관계되나 전역 조망.
- **방식**: obj 토큰 3D 산점도, 색=target_ax (RdBu). raw → (수정) ttc-augment → (수정2) GT-ACC → (수정3) gated. 투명도/샘플수 조정. DriverB 전체(351K) 및 5k 샘플.
- **결과**: GT-ACC 적용 후 center bin+접근에 감속(파랑) 집결, 측면/비접근 가속(빨강) 확인.

### C-2. voxel: dist × closing → ax mean/std
- **목적**: 각 (거리,접근속도) 구간의 target brake 강도 + 일관성(std).
- **방식**: in-lane bin 합산, dist×closing 격자, 셀별 mean/std heatmap + 표. <20m → 0-60m 확장.
- **결과**:
  - predicted-ttc(버그): <5m+접근 +6.9 (가속), std 8-13 (비일관)
  - **GT-ACC ttc(수정)**: <5m+접근 -13.7 (강한 감속), std 0.2 (일관). closing↑→brake↑, dist↑→brake↓ 단조.
  - **gated(완화)**: 40m@11 -6.6→-3.0 (원거리 완화), 근거리/긴급 유지.

### C-3. out-lane voxel
- **목적**: 경로 밖 객체는 brake 불필요(가속/cruise) 확인.
- **방식**: out-lane bin(0-15,26-39) obj, bin×dist + 순수outlane(in-lane lead 없음) closing×dist + ax 히스토그램.
- **결과**: 순수 outlane ax +1.79 (brake 9%, 가속 64%). 근거리<10m 도 +2.97. → **측면차 헛brake 안 함 ✓**. brake 하는 경우는 in-lane lead 동반 때문.

## D. 제어(control) 검증

### D-1. ax target↔pred scatter (B1/B0ttc)
- **목적**: 모델이 brake target 을 학습했나 (부호/크기).
- **방식**: 학습 ckpt forward, ax_target vs ax_pred 산점도 + brake-event pred 평균.
- **결과**: B1(4L,non-ttc) brake event pred +0.57(target -5.94). B0ttc(1L) +9.79 (더 나쁨). → ttc no-op(#1) 때문 brake 미학습 확인.

### D-2. grad-cam: ax 출력 (v25h-B)
- **목적**: 감속 출력 시 정면(lead) bin 보는가.
- **방식**: ax[0] 출력 → i_2 grad → bin별 집중도. brake sample 15개.
- **결과**: center 집중 43%, front_obj 27%, **ax 부호 7/15 틀림** → brake 학습 자체 망가짐 (v6 cache + ttc no-op).

## E. 보조 검증

### E-1. pruning sensitivity
- **목적**: a_2 모델 압축 여지.
- **방식**: global L1 unstructured pruning 0-90%, obj/vL_rel 측정 (finetune 없이).
- **결과**: 50% sparsity 까지 obj rec 거의 유지, 90%도 붕괴 안 함 → redundancy 큼. but unstructured 는 실속도 이득 0 (2:4 structured 필요).

### E-2. inlane NO-OP 확인
- **목적**: B 학습 로그에 inlane 계산 실행됐나.
- **방식**: train 로그 "in-lane computed" 메시지 grep + launcher cache 인자 확인.
- **결과**: 메시지 없음 + a2_cache=None → #1 확정.

---

# 핵심 인과 사슬 (최종)

```
B closed-loop "감속 안 함"
 ← brake target 에 강한 감속 신호 없음
    ← ttc-augment NO-OP (#1: inlane 계산이 i2 path 에서 누락)  ← 최대원인
    ← (작동했어도) ttc flag 누락 일부버전 (#3)
    ← (작동했어도) magnitude 약함 → ACC-style (#9)
    ← (작동했어도) predicted dist 과대평가 → GT inlane (#7)
 ← (별개·부차) a_2 vL_rel 부정확 (#2 fix 후에도 perception 한계, #10)
```

**최종 fix (v25k-B4)**: GT inlane cache + ACC-style ttc(TTC-gated) + 4-layer side + a_2 v7b. main(Professor) steer 보존, side 는 GT-정확 lead-brake 학습.

# 검증 방식 요약 (재사용 가능)

| 검증 형태 | 무엇을 보나 | 도구 |
|---|---|---|
| corr/MAE/sign (bucket) | 예측 정확도, 거리/속도별 | cache vs aux GT, numpy |
| 분포 sanity | GT/예측 range·collapse | 히스토그램, std |
| grad-cam | 공간 attention (어디 보나) | output→input grad |
| grad-cam overlay | GT projection 정확성 | raw 이미지 + grad |
| scatter (target↔pred) | 제어 학습 여부, 부호 | 산점도 |
| 3D scatter | 전역 관계 조망 | dist×bin×closing, 색=ax |
| voxel mean/std | 구간별 강도+일관성 | dist×closing 격자 |
| 인과 진단 | 특정 이상치 원인 | 조건부 통계 (fire율 등) |
| pruning sweep | 모델 redundancy | sparsity vs metric |
| fast 샘플 cache | 반복 분석 가속 | 5k frame 사전 npz |

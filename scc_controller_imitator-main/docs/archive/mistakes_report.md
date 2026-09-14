# 실수 및 수정 리포트 — 타임라인 (2026-05-22 ~ 25)

IyModel(Professor main + 이미지 기반 SCC longitudinal)을 만들며 발견·수정한 실수를
**발생/발견 시간순**으로 정리. 두 부분으로 나눔:

- **Part 0 — 모델 구조 설계 시행착오 (05-22~24):** "구조를 잘못 설계하고 있던" 항목들.
  여러 아키텍처(self-attn concat → side branch → cross-attn → Professor 분리)를 거치며
  근본 설계 결함(특히 img feature 차원 불일치)을 찾아낸 과정.
- **Part 1 — B brake 디버깅 타임라인 (05-24~25):** 구조가 정리된 뒤, closed-loop 에서
  "앞차 접근 시 감속 안 함" 문제를 추적하며 잡은 실수들.

각 항목: (목적/증상) → 근본원인 → 정정. (사용자 지적) 표시 = 사용자가 먼저 발견.

> 영향도순 정리 및 데이터 검증 활동은 `data_verification_report.md` 참조.

---

# Part 0 — 모델 구조 설계 시행착오 (2026-05-22 ~ 24)

> v6~v25 가 전부 05-24 하루에 행해진 실험이라 버전 번호는 시간순이 아님(사용자 확인).
> 아래는 **구조 변경 의도** 기준으로 정리. 핵심 결함은 D8(img feature 256↔128 불일치).

### D1. 이미지를 main self-attention 에 concat — Professor steer 오염
- **원설계**: `traj + img(fi_mem_gated) + ego → self-attention → control`. 즉 이미지 토큰을
  main transformer 입력에 concat. Professor(controller_imitator) pretrain 위에 이미지를
  섞으려 한 구조.
- **문제**: 이미지 feature 가 main 에 직접 섞이며 Professor 의 깨끗한 횡방향(steer) 추종을
  오염. waypoint tracking 이 무너지고(05-23 14:21 "추종성 많이 떨어져"), 이미지 noise 가
  steer 로 새어나옴. 사용자도 의심: "logged dataset 이 steer 계산엔 안 좋은 게 아닌가".
- **정정 방향(05-24 14:15)**: main 은 `controller_imitator` 구조를 **그대로 계승, 이미지 concat 제거**
  (`main_no_image`). 이미지는 **side branch 로 분리**. → D6/D8 로 이어짐.

### D2. "복잡도 부족" 진단 → 병렬 side branch 도입 (v18, ControlNet 스타일)
- **목적(05-23 22:57~23:28)**: close-lead brake 미학습이 capacity 부족인지 feature 부족인지
  진단. 결론은 "pretrain 유지하며 복잡도만 ↑" → transformer 옆에 **병렬 모듈** 추가 + **2-phase**
  (main freeze 먼저). v18 ControlNet-style side branch.
- **시행착오**: v18 init 결함(v18b 수정), 정성 best 였던 v8 에서 resume(v18c). 즉 side branch
  자체는 맞는 방향이었으나 init/출처를 여러 번 잘못 잡음.

### D3. side_alpha 부호/스케일 — identity-start 설계 혼란
- **증상(05-24 00:30~00:39)**: side gate scale `alpha` 가 음수로 학습됨. "양수가 되게 설계한
  게 있나? layernorm/batchnorm 은?" 논쟁.
- **정리**: `side_alpha = Parameter(zeros)` 로 **α=0 (identity) 에서 시작**, main 을 해치지 않고
  점진적으로 side 기여를 키우는 설계가 맞음. 부호 자체는 자유(residual 합이라 의미 없음).
  BatchNorm 은 도입 안 함(시퀀스/배치 통계 불안정). → 설계 의도 재확인으로 마무리.

### D4. difficulty-aware sampler (v16/v17) — 샘플링으로 구조 병목 우회 시도 실패
- **목적(05-23 17:21~18:10)**: loss 상위 10%(close-lead/brake) 가 안 줄어 hard/easy 리밸런싱
  샘플러 + lr 5×.
- **결과**: 샘플링·lr 튜닝으로는 brake 학습 안 됨. **구조(이미지→ax 경로) 자체가 병목**이라는
  판단으로 이어짐 → D2(side branch)·D8(차원) 의 동기.

### D5. a_2 feature 주입 지점 혼란 (v20 풍부주입 / v21A late-inject)
- **증상**: i_g(객체 gate)가 **object presence 로만** 작동(05-24 09:11) — 접근/후퇴 구분 못 함.
- **시도**: v20 풍부 feature 주입, v21A 는 a_2 feature 를 **ax decoder 직통(late inject)**.
  "a_2 head 가 fi_mem 에 어떻게 작용하는지"(05-24 01:25) 가 불명확했음.
- **정리**: 주입 지점·방식이 그때그때 달라 ablation 이 꼬임. 최종적으로 cross-attn(D6)+i_2 cache(D8)
  로 경로를 단일화.

### D6. cross-attention 방향 오설계 → Q=traj, KV=img 정정
- **증상**: side cross-attention 의 Q/KV 방향이 초기엔 반대(이미지가 query).
- **정정(05-24 14:28)**: **Q = trajectory, KV = image**. side 는 main 과 같은 1-step 출력이되
  **acceleration 만** 내고, main ax 에 더함(residual). 횡방향은 Professor 가 전담.

### D7. 입출력 과설계 → 1.5s 입력 + 1-step 출력으로 축소
- **원설계**: 미래 long-horizon trajectory 를 입력하고 40-step 전체를 출력.
- **정정(05-24 14:15)**: 입력 trajectory 를 **미래 1.5초로 단축**, 출력은 **지금 제어할 steer/ax
  1-step** 만(`traj_zero_after`, `target_first_only`). 제어에 쓰는 건 step0 뿐인데 40-step 회귀가
  loss 를 희석시키고 있었음.

### D8. ⭐⭐ img feature 256 ↔ transformer 128 차원 불일치 (가장 근본적 설계 결함)
- **사용자 지적(05-24 23:30, 23:39)**: "이미지 인코더가 **왜 256차원으로 압축**하게 설계됐지?
  transformer 는 128차원인데. 그래서 따로 학습도 못 시키는 `img_proj` 모듈이 생긴 거 아니냐.
  no-image 플래그보다 **이 256→128 모듈이 언제부터 있었는지가 핵심** — 처음부터 설계가
  틀린 건지 중간에 바뀐 건지 판단되니까."
- **근본원인**: a_2 fusion 출력이 **256-dim** 인데 main/side transformer 는 **128-dim**. 이를 잇기
  위해 iy_model 안에 `img_proj`(Conv1d 256→128, `iy_model.py:147`)를 끼워 넣었음. 이 projection 은
  a_2 pretrain 에도 Professor pretrain 에도 속하지 않는 **고립된·따로 학습 불가능한 layer** →
  이미지 정보가 128 으로 들어오며 손실/미학습.
- **정정(05-24 23:57~00:43, a_2 v5)**: 256→128 projection 을 **a_2 내부 `fc_proj` 로 이동**
  (`a2_model.py:159`, fusion 끝에서 128 직접 출력). AuxHead 도 128 버전으로 재학습. 그 결과
  **i_2 cache 가 이미 128-dim(post-fusion)** → iy_model 의 `img_proj`/img_fusion 우회
  (`iy_model.py:225-226`), main·side 가 동일한 학습된 128 representation 을 공유.
- **의의**: "구조가 처음부터 틀렸나"의 답 — 차원 불일치 봉합용 `img_proj` 가 계속 끼어 있었고,
  a_2 v5 의 fc_proj 이전엔 이미지 표현이 일관되게 흐르지 못했음.

### D9. obj_gate = presence-only 한계 (구조적)
- i_g/obj_gate 는 "객체 있음/없음" 만 표현 → **다가오는 차와 멀어지는 차를 구별 못 함**.
  closed-loop 에서 "접근 인지하는데 계속 가속"(05-24 09:04)의 한 원인. 이후 vL_rel/closing 을
  명시적으로 쓰는 ttc-augment(Part 1) 로 보완.

### D10. side branch 1-layer 너무 단순 → 4-layer (B4)
- **사용자 지적(05-25 01:47)**: "side 구조가 너무 단순. 이미지를 보고 처리해야 하는데."
  → side cross-attn 을 1-layer(MultiheadAttention)에서 **4-layer TransformerDecoder stack**
  (`iy_model.py:190-194`)으로 확장. B 명명도 layer 수 기준으로 변경(B1=1L, B4=4L, Part 1 T20).

### D 정리 — 최종 구조
```
main:  Professor(#654, text+traj, NO image)  ──┐
                                               ├─ steer = main(고정 우수)
side:  traj(Q) × image_mem(KV, 128d i_2)       │  ax    = main_ax + α·side_ax
        4-layer cross-attn decoder ── side_ax ─┘  (α=0 identity start)
i_2 cache = a_2 fusion → fc_proj(256→128) → AuxHead(obj/vL_rel/dist...)
```
이미지·perception 은 side 로 격리, Professor 의 steer 는 보존. D1~D8 의 시행착오가 이 분리로 수렴.

---

# Part 1 — B brake 디버깅 타임라인 (2026-05-24 ~ 25)

B 모델이 closed-loop 에서 "앞차 접근 시 감속 안 함"(+9.79 m/s² 가속) 문제를 추적하며 잡은 실수.

## 배경 (이전, 2026-05-24)

### T0. build_a2_cache.py proj 누락 (이전 세션 수정됨)
- a_2 cache 빌드 시 `encoder.proj` step 누락 → ROC AUC 0.51(random). proj 추가 후 0.98.
- 이번 세션의 a_2 v5~ 는 이 fix 위에서 시작.

### T0b. DriverB prebake target=0 버그 (이전 세션 수정됨)
- prebake 가 모든 sample mask=0 → target 무효. LRU cache 로 재생성, valid 98%.

---

## Phase 0 — 시스템/데이터 정비 (세션 시작 ~03:30-04:00)

### T0c. 87GB swap + D-state pytorch 프로세스
- **증상**: bash 느려짐. swap 87GB 점유, pytorch proc(1608749) D-state(uninterruptible), kill -9 안 됨, acct_collect.
- **근본원인**: 이전 학습의 swap-in 미완 + 커널 process accounting.
- **정정**: swapoff 부분 적용으로 30GB 감소, 부모 프로세스까지 제거. 이후 reboot.

### T0d. GPU zombie 프로세스
- nvidia-smi 가 5 PID 표시하나 OS 엔 없음 → 드라이버 stale tracking. 실제 영향 없음 확인.

### T0e. a2_feat collate KeyError + a_2 v5 launch 실패
- **증상**: ci_aug 엔 a2_feat 있고 SCCSynth 엔 없어 collate KeyError. v25h-A 첫 launch crash.
- **정정**: SCCSynth `__getitem__` 에 a2_feat/i2_mem/a2_logits 기본 zeros 추가.
- 추가: a_2 v5 raw-image launch 시 sangam_fx640 raw_cache 에 prev frame 없어 pair 0 → **cache 기반 학습(v5_from_cache)** 으로 전환 (180× 빠름).

---

## Phase 1 — a_2 perception 추적 (새벽 ~04:00-05:30)

### T1. a_2 v5/v6 regression loss 결함 발견
- **증상**: a_2 v5/v6 vL_rel pred std=0 (모든 곳 0 출력), closing recall 0%.
- **근본원인**: `train_a2_v5_from_cache.py` 의 `loss_reg = F.l1_loss(logits[...,1:], target[...,1:])` — target normalize 없음 + obj_mask 없음 → 90% 토큰(obj=0,reg=0)이 "모든 곳 0" 학습 강제.
- **정정**: **v7** — target/scales normalize + obj_mask (train_a2_head.py 와 동일). pred std 0→0.3+, DriverB vL_rel corr +0.40.

### T2. eval 스케일 오류 → "GT 손상" 오판 (내 실수)
- **실수**: vL_rel eval 에서 GT(raw m/s)에 ×20 잘못 곱함 → range [-956,+558] (불가능) → "GT 망가졌다" 잘못된 결론.
- **정정**: GT 원본 분포 재확인 ([-9,+9]~[-38,+13], 정상). ×20 제거. 진짜 문제는 GT 가 아니라 모델 loss(T1).

### T3. focal_alpha=0.5 무의미 발견
- **증상**: a_2 v6 focal α=0.5 인데 obj recall 개선 미미 (sangam 3%).
- **근본원인**: `α·t+(1-α)(1-t)`, α=0.5 = pos/neg 동일가중 = 일반 BCE.
- **정정**: **v7b(α=0.75)/v7c(α=0.9)**. pos rate ~5% → 적정 ≈ 1-pos_rate. v7b(obj rec 63%/prec 66%) 균형 채택.

---

## Phase 2 — a_2 추가 개선 시도 & 한계 확인 (오전 ~06:00-08:00)

### T4. a_2 v8 (Gaussian NLL + LDS) — 개선 실패
- **시도**: interleaved mean+σ(13ch) + Gaussian NLL + inverse-density(LDS) 로 tail collapse 해소 기대.
- **결과**: vL_rel corr 오히려 하락 (0.45→0.31), dist 동일. → 효과 없음.

### T5. a_2 v9 (LDS only) — 개선 실패
- **시도**: NLL 제거, L1+LDS 만.
- **결과**: 근거리 dist 과대평가 그대로 (v7b 8.8 vs v9 14.1). → **perception 은 loss tricks 로 개선 불가** 결론. v7b 동결.

---

## Phase 3 — B 모델 brake 진단 (오전 ~08:00-09:00)

### T6. closed-loop B 테스트 — steer 정상, 감속 약함
- B 모델: steer "기깔남"(Professor 정통) but 앞차 접근 시 "찐찐막 감속" / 감속 안 함.

### T7. ax target↔pred scatter — brake 역방향
- **증상**: B1(4L) brake event 에서 pred +0.57(target -5.94), B0ttc(1L) +9.79 (가속!). 진짜 brake 상황에서 가속 출력.

### T8. GT ax↔vL_rel 상관 — 데이터 모순 의심
- DriverB corr +0.49(시각 일치) but sangam -0.1(raceline brake, 시각무관) → 데이터 오염 의심 (후에 정정).

### T9. 비-inlane 오염 정량
- brake target 의 44%(DriverB)~56%(sangam) 가 in-lane lead 없이 발생 (raceline/신호/회전).

### T10. brake target ttc 전후 확인
- 근거리+접근 raw target **+1.07(가속!)** → ttc-augment 적용 시 -3.22. ttc 적용이 핵심.

### T11. --ttc-augment flag 누락 발견 (내 실수)
- **실수**: v25d/e/f 에 있던 6개 ttc flag 이 v25h/i/j launcher 재작성 시 drop 됨.
- **정정**: 복원. (단 이때는 T16 의 NO-OP 때문에 복원해도 효과 없었음을 아직 몰랐음)

### T12. 3D scatter 를 raw target 으로 색칠 (사용자 지적)
- **실수**: scatter 색을 augment **전** raw 로 칠함. 모델이 학습하는 건 augment **후** target.
- **정정**: ttc-augment 적용 target 으로 재생성 (raw +1.15 → ttc -0.17).

### T13. ttc 분석에 predicted dist 사용 (사용자 지적)
- **실수**: ttc trigger 에 a_2 추론 dist 사용. 근거리 과대평가(true 3.8m→pred 8.8m)로 closing 음수 오판 → ttc miss.
- **정정**: GT aux dist + closing=-GT vL_rel 직접.

### T14. 초근거리 가속 원인 진단
- voxel 의 <5m+접근 +6.9 가속 원인 추적: GT-true <5m+접근 234 token 중 raw 가속 94%, ttc fire **7%**, predicted dist 8.8 vs GT 3.8 → closing 오판 → ttc miss.

---

## Phase 4 — 근본원인 확정 & 최종 fix (오전 ~09:00-09:30)

### T15. ax 분산 = 트랙 가속 계획 (사용자 지시 — 건드리지 않기)
- voxel <5m std 8-13 고분산을 override 하려 했으나 — 사용자: **raceline 가속 계획 포함된 정상 분산**. main(Professor)이 트랙 가속, side 는 lead-brake 만. → override 취소.

### T16. ⭐⭐ ttc-augment NO-OP 발견 (최대 근본원인)
- **증상**: B0/B1/B0ttc/B1ttc 모두 ttc flag 켰는데 brake 미학습 (+9.79 가속).
- **근본원인**: `scc_synth_iy_dataset.py` 의 in-lane 계산이 `if a2_cache_path:` 블록 **안에만** 존재. B 는 `--i2-cache` 만 씀 → `inlane_dist=inf` → approach mask 항상 False → **ttc 한 번도 fire 안 함**. T11 의 flag 복원도 이 때문에 무효였음.
- **정정**: `build_inlane_gt_cache.py` 신규 — GT aux 에서 inlane_dist + closing(-vL_rel) 계산. `--inlane-gt-cache` → dataset (cache 무관). v25k-B4 에서 70546 frame populate 확인.

### T17. v25k-B4 — GT inlane + ACC-style ttc 적용
- main_no_image + side 4L + ACC-style brake(근거리 강함, 긴급 -20). GT inlane 으로 ttc 실제 작동.

### T18. gated 재학습 시 pre-gate ckpt 미resume (사용자 지적)
- **실수**: TTC-gating 적용 후 재launch 때 B2 에서 resume → pre-gate v25k-B4 의 5ep brake 학습 버림.
- **정정**: pre-gate ckpt 복사 후 거기서 resume.

### T19. ACC ttc 원거리 과도 brake (사용자 지적)
- **증상**: 40m 에서 closing 11(TTC 3.6s) 여유 있는데 -6.6~-9.5 과함.
- **근본원인**: a_gap 의 `kclo·closing` 항이 거리/TTC 무관 상수.
- **정정**: closing 항을 TTC 로 gating (`--ttc-acc-kclo-ttc-decay`, TTC>2s 감쇠, 5s 에서 0). 40m@11: -6.6→-3.0, 근거리/긴급 유지.

### T20. B 명명규칙 변경 (사용자 지시)
- B 숫자 = side cross-attn layer 수. B0(1L)→**B1**, B1(4L)→**B4**.

---

## 부수 이슈 (인프라/툴)

### T21. glances GPU 웹 표시 안 됨 (모바일)
- **증상**: glances `-w` 웹 대시보드에 GPU 패널 안 보임 (특히 모바일).
- **진단**: `/api/4/gpu` 데이터 정상, JS 에 GlancesPluginGpu 존재, disable_gpu=False → 데이터/조건 OK. UI 렌더 문제(브라우저 캐시/레이아웃). nvidia-ml-py 설치로 API 확정.
- **정정**: hard refresh / custom dashboard 안내. (auto-mode 가 0.0.0.0/tailscale bind 차단 → user `!` 직접 실행)

### T22. orchestrator self-deadlock
- **증상**: 야간 chain 의 wait 스크립트가 `pgrep -f "v7c_d128"` 로 자기 자신(명령에 패턴 포함) 매칭 → 영원히 대기.
- **정정**: kill 후 pgrep 패턴이 자기 명령과 안 겹치게 / PID 직접 대기.

## 운영 실수 (세션 전반, 반복)
| 실수 | 정정 |
|---|---|
| `pkill -f "패턴"` 이 자기 bash 죽임 (exit 144) | PID 직접 kill |
| npz member loop 반복 access → 매번 재압축(수분 hang) | loop 전 1회 추출 |
| f-string backslash → SyntaxError | 변수 분리 |
| matplotlib 한글 깨짐 | 축은 영문 |
| print buffering | `python -u` |
| 전체 135K aux 매번 로드 (2-3분) | val_sample_5k.npz 사전화 (213KB, 즉시) |
| 분석 스크립트 silent crash (close20 빈 출력) | exit code + 출력 line 수 확인 |
| 중복 모니터 3개 같은 로그 추적 | TaskStop 으로 정리 |

---

## 핵심 인과 사슬 (최종)

```
B closed-loop "감속 안 함"
 ← brake target 에 강한 감속 신호 없음
    ← ttc-augment NO-OP (T16: inlane 계산이 i2 path 에서 누락)  ← 최대원인
    ← ttc flag 누락 일부버전 (T11)
    ← (작동했어도) magnitude 약함 → ACC-style (T17, T19)
    ← (작동했어도) predicted dist 과대평가 → GT inlane (T13)
 ← (별개·부차) a_2 vL_rel 부정확 (T1 fix 후에도 perception 한계, T4/T5)
```

**최종 fix (v25k-B4)**: GT inlane cache + ACC-style ttc(TTC-gated) + 4-layer side + a_2 v7b.
main(Professor) steer 보존, side 는 GT-정확 lead-brake 학습.

## 실수 패턴 회고

**구조 설계 (Part 0)**
- **차원 불일치를 projection 으로 봉합** (D8) — 256↔128 을 잇는 `img_proj` 가 따로 학습도 안 되는
  고립 layer 로 끼어 있었음. pretrain 모듈 간 차원은 **설계 시점에 맞추고**, 못 맞추면 그 projection 을
  학습 가능한 위치(여기선 a_2 내부 fc_proj)로 넣을 것.
- **관심사 분리 부재** (D1) — 잘 되는 Professor(steer)에 이미지를 섞어 둘 다 망가뜨림. 검증된
  pretrain 은 건드리지 말고 **새 능력은 residual side 로 격리**(α=0 identity start)하는 게 안전.
- **샘플링·lr 튜닝으로 구조 병목 우회 시도** (D4) — capacity/feature 병목은 sampler 로 안 풀림.
  병목의 층위(데이터 vs 구조 vs loss)를 먼저 가릴 것.
- **버전 번호 ≠ 시간순** (D 머리말) — v6~v25 가 하루에 행해져 번호로 history 추정 시 오판.
  변경 **의도** 기준으로 기록.

**디버깅/운영 (Part 1)**
- **launcher 재작성 시 flag drop** (T11) — diff 확인 습관 필요
- **배선 누락이 silent** (T16) — flag 는 켜졌으나 데이터 미공급. fire rate 로그 필수
- **분석 신호 불일치** (T12 raw vs augment, T13 predicted vs GT) — "모델이 실제 학습/사용하는 신호" 로 분석
- **resume 출처 확인** (T18) — 재학습 시 최신 ckpt 에서 이어받기

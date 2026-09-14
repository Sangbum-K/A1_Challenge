# 실수·오류·정정 기록 — 2026-05-25 세션

SCC Controller Imitator B 모델(이미지 기반 SCC brake)이 closed-loop 에서 "감속 안 함" 문제를 추적하며 발견한 버그와 정정. 시간순이 아니라 **영향 큰 순**.

---

## A. 학습 파이프라인 critical 버그

### A-1. ⭐ ttc-augment 가 B 모델에서 NO-OP (최대 근본원인)
- **증상**: B0/B1/B0ttc/B1ttc 모두 `--ttc-augment` flag 켰는데 brake 안 배움. closed-loop 에서 앞차 접근 시 +9.79 가속.
- **근본원인**: `scc_synth_iy_dataset.py` 의 in-lane(bin16-25) dist/closing 계산이 `if a2_cache_path:` 블록 **안에만** 존재. B 학습은 `--i2-cache` 만 쓰고 `--a2-cache` 안 씀 → `inlane_dist` 가 `inf` 유지 → `_apply_ttc_augment` 의 approach mask 항상 False → **ttc 한 번도 fire 안 함**. brake target = raw prebake (근거리+접근에서 +1.07 가속).
- **정정**: `build_inlane_gt_cache.py` 신규 — aux GT 에서 frame별 `inlane_dist`(GT dist) + `inlane_closing`(-GT vL_rel) 계산. dataset `--inlane-gt-cache(-extra)` 로 load (cache 종류 무관). v25k-B4 에서 적용 확인 (70546/72519 frame inlane populate).
- **교훈**: flag 켰다고 동작하는 게 아님. 핵심 augment 는 실제 fire rate/효과를 로그로 검증할 것.

### A-2. train_a2_v5_from_cache.py regression loss 결함 (a_2 v5/v6)
- **증상**: a_2 v5/v6 의 vL_rel 등 회귀 채널 pred std=0 (모든 곳 0 출력), closing 감지 recall 0%.
- **근본원인**: `loss_reg = F.l1_loss(logits[...,1:], target[...,1:])` — (1) target normalize 없음 (v 범위 ±30 vs BCE ±1), (2) obj_mask 없음 → 90% 토큰(obj=0, reg=0) 이 "모든 곳 0 출력" 학습 강제.
- **정정**: v7 — `train_a2_head.py` 처럼 `target/scales` normalize + obj_mask 만 loss. → pred std 0→0.3+, DriverB vL_rel corr +0.40.
- **교훈**: cache 기반 학습 스크립트를 새로 만들 때 원본(train_a2_head)의 loss 구조(정규화+마스크) 누락 주의.

### A-3. --ttc-augment flag 누락 (v25h+ launcher)
- **증상**: v25d/e/f 에 있던 `--ttc-augment` 6개 flag 이 v25h/i/j launcher 에서 사라짐.
- **근본원인**: v25h 부터 launcher 를 새로 작성하며 flag drop.
- **정정**: v25j-B 부터 6개 flag 복원. (단 A-1 때문에 복원해도 효과 없었음 → A-1 해결 후에야 실제 작동)
- **교훈**: launcher 재작성 시 핵심 flag diff 확인.

### A-4. focal_alpha=0.5 무의미
- **증상**: a_2 v6 focal loss α=0.5 인데 obj recall 개선 미미 (sangam 3%).
- **근본원인**: `alpha_t = α·t + (1-α)·(1-t)`, α=0.5 면 pos/neg 동일 가중 = 일반 BCE.
- **정정**: α=0.75(v7b)/0.9(v7c). pos rate ~5% → 적정 α ≈ 1-pos_rate ≈ 0.9. v7b(DriverB obj rec 63%) 채택.

---

## B. 내 분석 오류 (진단 중 실수, 사용자가 교정)

### B-1. eval 스케일 오류 → "GT 손상" 오판
- **증상**: 초기 vL_rel eval 에서 GT range [-956, +558] m/s (불가능) → "GT 망가짐" 결론.
- **실수**: GT(raw m/s)에 SCALE_V(×20) 를 잘못 곱함. 실제 GT 는 정상 ([-38,+13]).
- **정정**: ×20 제거. GT 정상 확인. 진짜 문제는 모델/loss 였음.

### B-2. 3D scatter 를 raw target 으로 색칠 (사용자 지적)
- **실수**: scatter 색을 augment **전** raw target_ax 로 칠함. 모델이 학습하는 건 ttc-augment **후** target.
- **정정**: ttc-augment 적용 target 으로 재생성. (raw +1.15 → ttc -0.17)
- **교훈**: "모델이 실제 보는 신호" 로 분석할 것.

### B-3. ttc 분석에 predicted dist 사용 (사용자 지적)
- **실수**: ttc 분석/trigger 에 a_2 추론 dist 사용. 근거리 과대평가(true 3.8m→pred 8.8m) → ttc miss.
- **정정**: GT aux dist 사용 (A-1 의 GT inlane cache). closing 도 -GT vL_rel 직접 (frame-diff 불필요).

### B-4. close ax target "망가짐" 과잉결론
- **실수**: raw target +1.07(가속) 보고 "target 망가짐" 결론.
- **정정 (사용자 Q)**: ttc 적용 시 정상 brake(-3.22). in-lane ttc target 자체는 정상, 문제는 (a) ttc no-op (A-1), (b) magnitude 약함.

---

## C. 발견했으나 "건들지 않기로" 한 것 (사용자 판단)

### C-1. ax 분산(std) = 트랙 가속 계획 포함
- voxel 분석서 <5m std 8-13 (고분산). override(replace) 하려 했으나 — **사용자: raceline 가속 계획이 포함된 정상 분산이므로 건들지 말 것**. main(Professor)이 트랙 가속 담당, side 는 lead-brake 만.

### C-2. a_2 perception 추가 개선 무의미 (실험으로 확인)
- v8(Gaussian NLL), v9(LDS) 모두 vL_rel/dist corr 개선 실패 (오히려 하락/동일). 근거리 dist 과대평가(8.8) 그대로.
- **결론**: 2-frame + perspective 의 근본 한계. v7b 동결, perception 추가 투자 중단. brake 는 dist+ttc(GT) 로.

---

## D. 운영 실수 (재발 방지용)

| 실수 | 정정 |
|---|---|
| `pkill -f "패턴"` 이 자기 bash(패턴 포함) 까지 죽임 → exit 144 | PID 직접 kill, 또는 패턴에 자기 명령 안 겹치게 |
| npz member 를 loop 에서 반복 access → 매번 재압축해제 (수 분 hang) | loop 전 `arr = z['key']` 1회 추출 |
| f-string 에 backslash (`f"{'a\\b'}"`) → SyntaxError | 변수로 분리 |
| matplotlib 한글 글자 깨짐 (DejaVu 폰트) | 데이터/축은 영문, 한글은 주석/보고로 |
| print buffering (`python ... > log`) 출력 안 보임 | `python -u` (unbuffered) |

---

## E. 명명 규칙 정정

- **B 모델 숫자 = side cross-attn layer 수** (2026-05-25 변경). B0(구 1-layer)→**B1**, B1(구 4-layer)→**B4**.

---

## 정리 — 핵심 인과 사슬

```
B closed-loop "감속 안 함"
  ← brake target 에 강한 감속 신호 없음
     ← ttc-augment NO-OP (A-1: inlane 계산이 i2 path 에서 누락)
        + ttc flag 누락 (A-3, 일부 버전)
        + (ttc 작동했어도) magnitude 약함 (kinematic) → ACC-style 로 강화
        + (작동했어도) predicted dist 과대평가 → GT inlane 으로 정확화 (B-3)
  ← (별개) a_2 vL_rel 자체 부정확 (A-2 fix 후에도 perception 한계, C-2)
```

**최종 fix (v25k-B4)**: GT inlane cache + ACC-style ttc + 4-layer side. main(Professor) steer 보존, side 는 GT-정확 lead-brake 학습.

# 현재 상태 · 함정 · Incident (에이전트 필독)

이 repo 를 만지기 전에 반드시 아는 함정과 현재 모델 상태. (2026-05-25 기준)

---

## 1. 현재 상태

- **배포 = prof635 + distbrake_v1** (`prof635_side_distbrake_v1.pt`, delta_mode=**brake_only**, distbrake 라벨 5 dataset 975K, val_pedal~0.52[no-lead 지배]). pedal label 도 `_ttc`(closing-ACC)→`_distbrake`(거리-onset) 로 전환. legacy: v5(signed/ttc, 0.0237), v6(+DriverB 0.0443).
- closed-loop: `run_closed_loop_prof635.sh <sangam|speedway> [mild|high]` (dt0p1s raceline, gate EMA=0.7). 오프라인 검증: `test_prof635_closedloop.py` PASS (양 트랙); ROS master 필요.
- 경로 portable (REPO_ROOT/SCC_DATA_DIR), 배포 weight git 동봉.
- legacy iy_model(v25h/v25k) 도 closed_loop 에서 `MODEL_TYPE=iy` 로 지원되나 점진 폐기.

---

## 2. 반드시 아는 함정 (CRITICAL)

### 2.1 closed-loop i2_mem 은 학습 캐시와 동일하게 계산
live i2_mem = `seg backbone(fx640_v3) → a_2(a2_v7b) → fusion(ego_mot=ZEROS) → fc_proj`.
- **seg backbone 은 fx640_v3 여야** (combined_v1 아님 — stale 상수 버그였음). **ego_mot 은 zeros** (build_i2_cache.py 가 0 으로 캐시 생성).
- 틀리면 live i2_mem 이 학습 분포와 **cosine 0.80** 으로 어긋나 SCC/제동 붕괴. 맞추면 cosine 1.0000.
- 검증: `verify_i2_live_vs_cache.py`. closed_loop_node.py `SEG_BACKBONE_CKPT` + line ~575 `ego_mot=zeros` 가 이 fix.
- **교훈: closed-loop 제동 이상은 모델보다 i2_mem(perception) 불일치부터 의심.**

### 2.2 f_1(base) 의 brake 는 traj-speed artifact
f_1(#654)은 이미지 안 봄. 학습 prebake 의 traj speed 채널에 감속이 박혀 있어 base 가 brake 한 것.
inference(raceline nominal speed)에선 traj-speed 안 줄어듦 → **base 는 앞차에 brake 안 함**. 종방향 lead 제동은 전적으로 **f_2(image)**가 담당. (검증: `ablate_traj_speed.py` — traj-speed flat 시 base −0.48→+0.28.)

### 2.3 gate FP brake (해결됨)
v5 학습 `gate_bias`가 +3.02 로 drift → no-lead 에서도 gate≈0.92 → f_2 delta 적용 → **앞차 없는데 FP brake**(가속 불가).
- 원인: gate 가 delta 와 redundant(delta 가 이미지로 자가게이팅) → unsupervised bias 가 압력 없이 drift. g≈1 로 수렴해 gate 무력화.
- fix: live/모델 gate 를 `fixed_thresh` = `sigmoid((a2_obj_amax − 0.3)·5)` 로 (a_2 obj: no-lead −0.46 / lead +1.41 분리). no-lead brake 24%→0%, lead 제동 100% 유지.
- 비상: `SIDE_OFF=1` (f_2 끔 = f_1 단독). gate 튜닝 `GATE_THRESH/GATE_SCALE`.
- gate F1(객체존재 기준, thr 0.3) = 0.728 (P0.85/R0.64), 접근 lead recall 0.90.

### 2.4 pedal label 은 augment 버전만 (raw 금지)
raw target_ax 기반 pedal label 은 앞차 접근 시 **가속(+) 라벨 버그**. 반드시 augment 된 `pedal_label_*_{ttc,distbrake}.npz` 사용 (`gen_ttc_pedal_labels.py`, BRAKE_MODE 로 분기). 현행 배포(distbrake_v1)는 `_distbrake`(거리-onset, closing 비의존) 라벨로 학습. `_ttc`(closing-ACC)는 legacy 이며 두 라벨은 공존(별 파일).

### 2.5 seg backbone 재학습 = 전체 cascade
seg backbone 바꾸면 모든 i1/i2 캐시 무효 → a_2/f_2 재학습. 함부로 재학습 금지.

---

## 3. ACC(상대속도 기반 제동) 약함 — v7 핵심 과제

성능 검증(`scripts/ablate_traj_speed.py`, vL_rel 상관 진단)에서:
- closing 강도별 제동 modulation 약함/역전 (접근強이 더 안 밟힘). 접근/멀어짐 구분 미미.
- **이중 병목 진단**:
  - a_2 vL_rel(ch3) ↔ GT closing corr = **−0.279** (지각 약함).
  - f_2 delta ↔ a_2 vL_rel = **+0.023** (그 약한 신호조차 f_2 가 무시).

**격리 실험 (GT vrel 주입, `--inject-gt-vrel`)**: GT 상대속도를 f_2 에 직접 주입하면
- f_2 delta 가 closing 으로 **정확히 modulate**: 멀어짐 +0.20(무제동), 접근弱 −0.17, 접근強 −0.23 (monotonic).
- **결론: f_2 capacity 는 충분. 병목은 a_2 의 vL_rel 지각.** (단 GT vrel 미정규화로 val 불안정 — 진단용)

**→ v7 처방**: 상대속도 **지각 강화**가 정답.
1. **재귀형 i_2**(curr × accumulated_prev 누적, 2-frame 대신 long temporal memory) → vL_rel 추정 강화. ← 유력.
2. f_2 에 a_2 vL_rel/dist 를 명시 입력 concat (i2_mem 만으론 추출 안 됨) + 정규화.
3. closing-sensitive aux loss / lead-upweight.

---

## 4. 데이터 무결성 교훈 (과거 incident)

- **DriverB traj 손상**: prebake traj 가 stale source(speed 25-63 고정)였음 → raw 에서 재구성(`fix_driverb_traj_prebake.py`, `_trajfix`). #654 steer corr 0.28→0.42(이동프레임 0.72).
- **a_2 cache proj 누락 버그**: build_a2_cache 가 encoder.proj 생략 → 잘못된 i2. fix 후 재생성 (docs/archive/INCIDENT_a2_cache_proj_bug.md).
- **a_2 v5/v6 reg loss 버그**: reg normalize+obj_mask 누락 → vL_rel pred std=0. v7 에서 fix.
- **velocity frame**: OBJECT velocity 는 자기 vehicle frame (world 아님). preprocess 가정 오류였음.
- pertO/pertX 88% empty key → _fixed prebake 로 100% coverage.

---

## 5. 진행 중 / 미완

- distbrake_v1 의 lead-specific 제동 품질 평가(global val_pedal 은 no-lead 지배 — gradcam/lead-TTC 평가 필요). v5/v6 는 legacy.
- v7: 재귀형 i_2 (상대속도 지각) — §3.
- f_2 가 fixed_thresh 면 no-lead(gate≈0) 무학습 → lead 샘플(~16%)만 학습. lead-upweight 검토.
- 데이터 현황 "마법사" 도구 (계획: [DATA_AUTOMATION_PLAN.md]).
- legacy iy/v2 정리.
- 경량 seg backbone 교체 (b0, tok 2.14x). 단 vehicle IoU −0.055 → a_2-on-b0 ablation 전 보류, gate 안정화 이후 판단 ([REPORT_seg_backbone_lite.md]).

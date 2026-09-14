# 세션 다이제스트 — 2026-05-26 야간 ~ 2026-05-27 새벽

> 대상: `scc_controller_imitator` (A1 CHALLENGE 2026, 한양 AutoHYU 카메라+ego → steer/pedal SCC imitation).
> 범위: 2026-05-26 23:16(d449757) ~ 2026-05-27 02:49(f815af1). git 13 commit. `docs/EXPERIMENT_DIGEST_2026-05-22_26.md` §9 이후를 이어받는다.
> 출처: `git log --since`(commit date/body) · `outputs_multi/**/train.log` · eval JSON · `outputs_multi/data_dist/`. 새 학습/eval/plot 수행 안 함(read-only 수집). confirmed(train.log/eval JSON 직접 인용) / hypothesis(보고치·관찰) 구분.
> 파이프라인 불변식: camera(curr,prev)→seg(fx640_v3 frozen)→a_2(i2_mem 40×128 + a2_logits 40×7)→{f_1 main #654 frozen, f_2 side cross-attn}→gate g→final=clamp(base+g·delta). 종방향 lead 제동은 f_2(image)+gate 담당.

---

## 0. 한 줄 요약 · 오늘의 결정

이번 세션의 큰 줄기: (1) gate 를 rule → **learned NN gate** 로 전환하다 collapse 를 만나 **BCE supervision + vrel 입력 + g=2·sigmoid** 로 단계적 교정, (2) a_2 perception 의 **bin80(stride16) vs 40bin** / **b0 vs v2s** 두 ablation 을 종결해 **40bin·v2s_n20 확정**, (3) SCC 거동을 후처리 노브가 아니라 **MPC SCC 파라미터(t_headway/s_min) 5-profile 로 데이터에 심는** 방향 전환(핵심 발견: 기존 pedal_label 은 v_scc 무시, MPC 전체 재실행 불필요), (4) gate **데이터 분포 분석**(positive 의 synthetic 과의존 / far+고closing 희소 / brakeaug 다양성 착시).

오늘의 결정:

| 항목 | 결정 | 근거(§) |
|---|---|---|
| a_2 lateral bin | **40bin 확정, bin80 폐기**(cache 삭제) | §2 |
| a_2 backbone | **v2s_n20(20f) 확정**, b0 폐기 | §3 |
| gate | **BCE supervision + vrel(closing) 입력 + g=2·sigmoid 방향**(학습은 데이터 후로 보류) | §1 |
| SCC GT | F2_DELTA_SCALE 후처리 노브 버리고 **MPC SCC 5-profile → v_scc → pedal_label**(python 재계산, synth_batch 불필요 방향) | §4 |

---

## 1. gate 진화 — rule → learned NN gate

목적: f_2 의 lead-presence gate 를 하드코딩 rule(fixed_thresh / EMA / distill / beat)에서 end-to-end 학습 MLP 로 교체(규정 clean, in-graph). 학습 forward = 추론 forward(closed_loop 는 `model.last_gate` 만 사용).

시리즈(전부 prof635_side_distbrake_v1.pt warm-start, main #654 frozen, trainable ≈1.08M, total 975,871 / train 927,078 / val 48,793):

| 단계 | commit | 변경(설계) | gate 입력 | 결과(confirmed) | status |
|---|---|---|---|---|---|
| NN gate v1 | e4f3060, 017e8b0 | rule → gate_head MLP(193 params), g=sigmoid(gate_head(obj logit 10)) | obj(10) | **collapse** — best val_pedal 0.4938. train.log 에 gate 분리 metric 미기록 | collapse |
| gate BCE | bd159de | gate target g*=(inlane lead) & (closing>tau) 에 BCE 가산(w_gate=1.0). i2_cache → mf20(v2s 20f) | obj(10) | best val_pedal **0.6217**(ep23). g|lead&closing **0.534** / g|else **0.075**, bce 0.140 | confirmed |
| BCE+vrel | 327c590, a9711ce | gate 입력에 closing(=−vL_rel) 추가 → 20dim(obj+closing). receding 닫고 approaching 만 open. gate_head 529 params | obj(10)+closing(10) | best val_pedal **0.6171**(ep16, ep17 중단), g|lead&closing **0.622** / g|else **0.071** | confirmed |
| g=2·sigmoid | ee98d36 | g∈0~2 (기존 sigmoid 천장 해소) | obj(10)+closing(10) | objvrel_g2 train.log ep03 에서 멈춤(val 0.6083). **학습 데이터 후로 보류** | in-progress(보류) |

핵심 흐름:

- **collapse(NN gate v1)**: distbrake 라벨은 no-lead 에도 brake 가 깔려 있어 gate 를 닫을 학습 유인이 없음 → g 가 1.0 부근으로 붕괴, no-lead 에서도 제동(보고치 FP-brake **77.8%** — train.log/git 직접 근거 없음, closed_loop 관찰값 [hypothesis]). best val_pedal 0.4938 은 collapse 상태의 pedal loss 일 뿐 gate 건강성과 무관.
- **gate BCE(bd159de)**: target g* = obj & closing GT(inlane_gt 의 dist finite & closing>gate_tau). BCE 가산으로 collapse 해소 — lead&접근 0.534 vs else 0.075 로 분리. val_pedal 도 0.62 대로 회복(collapse 0.49 는 라벨 분포 다른 별 scale 이라 직접 비교는 주의, 단 gate 분리 metric 으로 건강성 확정). 이 단계부터 i2_cache 가 **mf20(v2s 20f)** 로 교체(§3 결정과 정합).
- **BCE+vrel(327c590)**: obj-only gate 는 receding(멀어지는 lead)에도 obj>0 라 g 가 켜짐. closing 입력 추가로 "lead 있고 AND 다가옴"만 open. lead&closing g 가 0.534→0.622 로 더 열리고 else 는 0.071 유지 → 의도대로 approaching 분리 강화.
- **g=2·sigmoid(ee98d36)**: 기존 g=sigmoid 는 g=1(full delta)이 asymptote 라 open 에서도 g≈0.6 천장에 막혀 full 제동 불가. g=2·sigmoid 로 logit 0 에서 g=1 도달, open→g→2(full+ 제동 가능). BCE 는 logit 기반 그대로, final clamp 유지. 학습은 데이터 후로 보류.

잔여 이슈(미해결):

- **f_2 g 모호성**: f_2 가 g 를 모른 채 delta 를 학습하므로 곱셈 g·delta 의 책임 분배가 모호(같은 final 을 g·delta 여러 조합이 만듦). [hypothesis]
- closing 기반 modulation 신뢰도 약함 — corr(final, closing) ≈ −0.13 (약함). [보고치, hypothesis]

---

## 2. bin80(stride16) vs 40bin — 40bin 확정, bin80 폐기

목적: far lateral 양자화(stride32 → W=40 column, far 객체가 <1 column) 해소를 위해 도입한 stride16(W=80) 의 실익 판정.

| 구성 | far vL_rel(closing) std | 출처 |
|---|---|---|
| 40bin 기준(2f 계열) | ~3.0 | digest §9 / commit d449757 |
| bin80 stride16 1D-merge (rule gate 버전) | 3.0 → **1.9** | commit d449757 body (confirmed) |
| bin80 s10n20 far vL_rel std | **0.96** (40bin 20f 는 1.1) | 세션 보고치 [hypothesis — 수치 출처 불일치] |

수치 주의: commit d449757 은 bin80 의 far closing std 를 **3.0→1.9**(rule gate 버전)로 기록한다. 세션 보고의 **0.96(40bin 3.0 대비)** 및 **40bin 20f 도 1.1** 은 train.log/eval JSON 에서 직접 확인 못 함(bin80 학습 자체가 SLURM 제출 대기로 `outputs_multi` 에 결과 부재) → [hypothesis]. 어느 수치든 결론(이득 marginal)은 동일.

결론: bin80 의 far 이득은 **frame density 가 주 lever**(§3, s10n20 이 far std 를 9.8→1.6 으로 반감)인 상황에서 marginal(보고 기준 +0.14)에 그치고, 별도 cache(`tokens_s16`)·proj16·α merge 경로 유지 비용과 통합 리스크가 더 큼 → **40bin 확정, bin80 폐기**(cache 삭제). far 분석은 §EXPERIMENT_DIGEST §9.8 의 0521 편중 confound 한계 안에 있음(검증 가능 한계 60m).

status: 40bin 확정 [confirmed 결정]. bin80 정량 std 일부 [hypothesis].

---

## 3. b0 backbone 경량화 — v2s_n20 확정

목적: closed-loop latency 단축을 위한 seg backbone v2_s → b0 교체 가치 판정. 판단 기준은 seg mIoU 가 아니라 a_2-on-b0 품질(b0 의 vehicle IoU 저하가 obj/vrel 로 전파되는가).

a_2-on-backbone eval(동일 held-out, `a2_b0_abl/eval_*.json`, confirmed):

| 모델 | frame | obj F1 | dist MAE | vLrel MAE | vLrel corr | corr_near | flip | FP(=fp/n_pos) |
|---|---|---|---|---|---|---|---|---|
| v7b (2f 기준) | 2f | 0.721 | 4.94 | 3.34 | 0.378 | 0.306 | 0.078 | 46.4% |
| b0 5f | 5f | 0.812 | 2.02 | 1.70 | 0.888 | 0.782 | 0.123 | 27.3% |
| v2s 5f | 5f | 0.845 | 1.52 | 1.39 | 0.920 | 0.842 | 0.083 | 21.0% |
| b0 n20 | 20f | 0.876 | 0.85 | 0.75 | 0.978 | 0.929 | 0.044 | 18.0% |
| **v2s n20** | 20f | **0.898** | **0.60** | **0.57** | **0.987** | **0.949** | **0.044** | **14.8%** |

(FP% = fp/n_pos 환산; 보고의 5f "b0 21.3% / v2s 11.2%" 와는 FP 정의·집계가 다름 — 위 표는 eval JSON 의 fp/n_pos 직접 환산값 [confirmed]. 추세는 동일: 동일 frame 에서 b0 가 전 지표 v2s 미달.)

핵심:
- 5f 에서 b0 는 obj F1 0.812 / corr 0.888 로 v2s(0.845 / 0.920) 미달, flip 도 0.123 vs 0.083 으로 나쁨.
- 20f(n20)로 가면 b0 도 F1 0.876 / corr 0.978 까지 올라옴(frame density 가 backbone 저하를 상당 부분 상쇄) — 보고의 "b0 20f ≈ v2s 5f" 와 정합.
- 단 **동일 20f 면 v2s 우위**(F1 0.898, far std·flip 우수). b0 latency 이득(seg 단독 2.14×, digest §9.6)도 20f fusion 에 희석되어(보고 2.25×→1.49×) trade-off 가 약해짐.

결론: **v2s_n20 확정**. b0 폐기(cache 삭제). status [confirmed 결정].

note: 동시에 v2s 20f 학습(`mf20f_v2s_5to100`)이 완료(best val total 0.0253, ep12) — gate BCE(§1)의 i2_cache mf20 base 가 이것.

---

## 4. SCC 5-profile — 거동을 데이터에 심기

목적: 앞차 거동(공격적/보수적 cruise)을 후처리 노브(F2_DELTA_SCALE)가 아니라 **MPC SCC 파라미터(t_headway / s_min)로 GT 자체에 심어** 학습.

profile 표(확정, `run_profile_synth_set.sh`):

| profile | t_headway | s_min | 성격 |
|---|---|---|---|
| p1_aggr | 0.4 | 2.0 | 공격적(늦게·얕게 감속) |
| p2 | 0.8 | 3.0 | |
| p3 | 1.2 | 4.0 | |
| p4 | 1.6 | 5.5 | |
| p5_cons | 2.0 | 7.0 | 보수적(일찍·깊게 감속) |

핵심 발견(confirmed, commit 45b7290 / 4e22113):
- **기존 pedal_label 은 v_scc(replanned speed)를 안 쓰고 후처리로 덮어** t_headway 변화에 무반응이었음. v_scc 만 profile 을 반영하고 long_traj 의 ax 는 거의 무반응 → **v_scc 경로가 profile 차이의 유일한 통로**.
- 따라서 GT 생성은 v_scc 직접 변환: `gen_profile_pedal_labels.py` 가 v_scc 시간미분 → ax, (v_scc, ax) → PedalLabelizer → `pedal_label_<ds>_p{N}.npz`(기존 `_ttc` 와 동일 format/keys 순서). 후처리 노브(acc_safe_decel) 제거.
- sanity(viz_profile_vscc, 27 lead frame): lead 앞 감속 깊이/강도가 p1_aggr→p5_cons **단조 증가** — dv_max **3.95→9.31 m/s**, |ax_min| **1.42→7.59 m/s²**.

더 나아간 단순화 방향(보고, [hypothesis/진행 중]): SCC replanner 로직이 단순(v_safe=(gap−s_min)/t_headway)하므로 **synth_batch(C++ MPC 전체) 재실행 불필요** — 기존 scc.txt + EGO_INFO 로 python 에서 재계산 가능(ego-start feasible: trajectory v → SCC v → 차량 feasible v(ego_v0 시작, a_max=5 / a_min=−20) → ax). 단 이번 commit 들은 아직 synth_batch v_scc 경로를 거침.

버그 fix(f815af1, confirmed): vscc_to_ax 가 정지구간 cum_s 정체로 t 중복(dt=0) → np.gradient divide-by-zero. 한쪽 유효이웃 차분으로 fallback. curr_keys 빈 prebake(DriverB clip)는 sample_key suffix 제거로 복원(trajfix curr_keys 와 100% 순서일치). sangam p1(72,519 sample, miss=0) 변환 검증.

status: 변환 파이프라인·sanity [confirmed]. 5-profile pedal_label 전체 생성 + 학습 [진행 중].

---

## 5. 데이터 분포 분석 (gate positive)

목적: gate 학습에 쓰는 positive(lead & 접근) 분포의 편향 진단. `analyze_gate_data_dist.py` → `outputs_multi/data_dist/`(plot 7장 + summary_table.txt). 수치 confirmed(summary_table.txt 직접 인용).

| dataset | track | rows | unique frame | lead% | gate_pos% | far+hiClos% | closing 분포 |
|---|---|---|---|---|---|---|---|
| sangam_lead_0520 | sangam | 72,519 | **728** | 97.3 | 27.8 | 3.24 | receding 49.2 / appr_strong 18.9 |
| sangam_lead_0521 | sangam | 97,753 | **985** | 22.6 | 22.6 | **49.28** | **appr_strong 100** |
| sangam_pertO_v2 | sangam | 338,073 | 3,404 | 0.0 | 0.0 | 0.0 | no-lead |
| sangam_pertX_v2 | sangam | 332,413 | 3,343 | 0.0 | 0.0 | 0.0 | no-lead |
| DriverB_new | sangam | 135,113 | **132,838** | 44.9 | 20.4 | 10.16 | receding 49.8 / appr_strong 35.2 |

발견(confirmed 수치 + 해석):
- **gate positive 의 synthetic 과의존**: 강접근(appr_strong) 분포가 0521(approach 100%)에 쏠려 있음 — 즉 "lead & 접근" 양의 표본이 한 synthetic 셋에 편중.
- **far + 고closing 희소**: far+hiClos% 가 0521(49.3) 외에는 모두 한 자릿수 — far 거리에서 빠르게 다가오는 표본이 부족.
- **brakeaug 다양성 착시**: sangam_lead 계열은 rows 가 7만~9만이나 **unique frame 은 728 / 985**(증강 반복) — DriverB_new unique 132,838 과 대비하면 sangam 의 다양성은 착시. positive 다양성의 실질 공급원은 DriverB.
- speedway 는 위 5셋에 없음 → **f_2 학습셋에 speedway 부재**. seg backbone(fx640_v3)은 0519 sangam+speedway 만으로 학습 → **DriverB 도메인 미커버**. [hypothesis — backbone 학습셋 구성 기준]

plot: `outputs_multi/data_dist/01_frame_counts.png` ~ `07_driverB_vs_sangam.png` (7장, 신규 생성, 본문 수치는 summary_table.txt).

---

## 6. 인프라

| 항목 | 내용 | status |
|---|---|---|
| acc-test → main merge | bd236a1(2026-05-26 23:27) origin/main → acc-test merge | confirmed |
| multicar 런처 | `multicar/run_multicar.sh` + `scc_multicar.ini`(CAR_NS namespace). prefix→ckpt 매핑 검증, 죽은 GATE_A2_CKPT 제거 → ACC_A2_MF5F, GATE_THRESH 0.3 정합, cmain Car4 (03e0ac9) | confirmed |
| moraibridge catkin | 구 morai_bridge 삭제 → catkin 패키지(src/) 충돌 해소, 빌드 가능 레이아웃(e8d2940, b5088c5) | confirmed(빌드검증은 digest §6) |
| CAMERA_1 로컬 복사 | 76GB 로컬 복사, 병렬 cp 가 rsync --files-from 보다 빠름 | [hypothesis — git/log 직접 근거 없음, 인프라 작업 관찰값] |
| cache 정리 | b0 / 옛 i2 / bin80 cache 삭제(§2·§3 결정 후속) | [hypothesis — git 미기록] |
| SLURM | bin80 ablation sif/job(`slurm/train_a2_bin80.sbatch`, `train_a2_e2e.sbatch`) 준비, 제출 대기 | in-progress |

---

## 7. 교훈 (cold)

- **NN gate collapse 원인은 라벨**: distbrake 가 no-lead 에도 brake 를 깔아 gate 를 닫을 유인이 없음 → g→1 붕괴. gate 는 supervision(BCE, target=obj&closing GT) 없이 pedal loss 만으로는 못 배운다. 곱셈 gating(g·delta)에서 g 를 pedal loss 로만 학습시키는 것은 구조적 underdetermination.
- **sigmoid asymptote = g 천장**: g=sigmoid 는 full delta(g=1)가 asymptote 라 open 에서도 g≈0.6 에 막혀 full 제동 불가. g=2·sigmoid 가 logit 0 에서 g=1 도달. (range 설계가 modulation ceiling 을 결정.)
- **f_2 g 모호성**: f_2 가 g 를 모른 채 delta 학습 → g·delta 책임 분배 모호. gate 와 f_2 의 joint 학습이 같은 final 을 여러 (g,delta) 조합으로 표현 가능. [미해결]
- **synth 는 v_scc 만 profile 반영, MPC 재실행 불필요**: 기존 pedal_label 이 v_scc 무시·후처리 덮어쓰기라 t_headway 무반응이었음. profile 거동의 유일한 통로는 v_scc 이고, replanner 가 단순(v_safe=(gap−s_min)/t_headway)이라 C++ MPC 전체를 다시 돌릴 필요 없음(python 재계산 가능, [진행 중]).
- **brakeaug 다양성 착시**: rows 수(7만~9만)와 unique frame(728/985)의 괴리 — 증강 반복을 다양성으로 착각하면 안 됨. positive 실공급원은 DriverB(unique 132,838). + gate positive 의 synthetic(0521) 과의존.
- **backbone 키우기는 frame density 를 못 이김**: b0↔v2s 모두 20f 에서 격차가 좁혀짐(frame density 가 backbone 저하 상쇄). 단 동일 frame 이면 v2s 우위 + b0 latency 이득은 fusion 에 희석 → 교체 무의미.

---

## 8. 미완 / 다음

1. **gate g=2·sigmoid 학습**: SCC 5-profile 데이터 생성 후 재학습(현재 보류, objvrel_g2 는 3ep 만). [데이터 의존]
2. **SCC 5-profile pedal_label 전체 생성**: p1~p5 × 전 dataset 변환 진행 중(sangam p1 만 검증). 이후 profile-conditioned f_2 학습. [진행 중]
3. **closed_loop 정성평가 대기**: gate BCE / BCE+vrel / g2 의 실주행 FP-brake·제동 escalation 확인(현재 collapse FP-brake 77.8% 는 관찰 보고치, 정량 재산출 필요). [대기]
4. f_2 g 모호성 구조 해소(g 를 f_2 입력에 노출 또는 gate·delta 분리 학습). [hypothesis]
5. bin80 폐기 후속 cache 정리 verify, DriverB/speedway 의 gate positive 다양성 보강(데이터 수집). [정리]

---

## 부록: 수치 출처

- confirmed = train.log / eval JSON / commit body 직접 인용.
  - gate: `prof635_side_nngate_{v1,bce_v1,objvrel_v1,objvrel_g2}/train.log` (val_pedal, gate 분리 metric).
  - b0/v2s: `a2_b0_abl/eval_{b0,b0_n20,v2s_n20,v2s_3set}.json`, `mf20f_v2s_5to100/train.log` (best 0.0253).
  - SCC profile: commit 45b7290 / 4e22113 / f815af1 body, `run_profile_synth_set.sh` profile 표.
  - 데이터 분포: `outputs_multi/data_dist/summary_table.txt`.
- hypothesis(직접 근거 없음, 보고치/관찰):
  - NN gate collapse FP-brake 77.8% — train.log/git 미기록, closed_loop 관찰.
  - bin80 s10n20 far std 0.96 / 40bin 20f 1.1 — 학습 결과가 outputs_multi 에 부재(commit 은 3.0→1.9 기록). 결론(이득 marginal)은 무관하게 동일.
  - CAMERA_1 76GB 병렬 cp, cache 삭제 — 인프라 작업이라 git 미기록.
  - corr(final, closing) −0.13, FP-brake 환산 — 별도 분석 산출, 본 세션 파일에 미수집.
- 새 plot 미생성. 기존 `outputs_multi/data_dist/*.png`(7장), `outputs_multi/profile_vscc_sangam_lead_0520.png`, `outputs_multi/a2_b0_abl/*.png` 참조.

## (추가 2026-05-27 오후) closed_loop 핫픽스 + 오인지 측정 + a_2 DriverB 재학습

confirmed:
- closed_loop i2_mem mf20 핫픽스 (commit 15d4353): 추론을 2-frame A2Model → 21-frame A2ModelMultiFrame 로 정합. live vs cache cos 1.0000 (sangam/DriverB). 이전 train21f/infer2f 불일치 = 오인지 일부 원인. iy A/B 만 2f 유지.
- 오인지 측정 (`/tmp/a2_misperception_report/`, cache npz 비교·추론 불필요): presence in-dist close<30m recall@p0.9 **0.91**·ROC 0.975(충분), mid 0.30, far 0.00. vL_rel closing sangam_lead Pearson 0.80~0.85, **DriverB 0.18/MAE6.7**.
- 핵심 발견: mf20 a_2 학습셋 = 3-set(sangam_lead 0520/0521 + spd_v2), **DriverB 미포함** (train.log args). DriverB vL_rel 붕괴 = capacity 아닌 OOD.
- p3_ng cosine 140ep 완료: best val_pedal **0.0422** (프리뷰 0.0436 → 개선), commit 6197378.
- raceline: sangam mild/high 를 ay22/ay24 신규로 재생성 (P/M 252/1248, 255/1228), pit polygon vertex x-20. track/map ini 자동매핑 (sangam/speedway → location/ref).

진행/계획:
- Phase 1 (`scripts/train_a2_n20_db.sh`, 실행 중): 3-set + DriverB_new (4-set, DriverB 64%), mf20 warm, recipe 동일. ep당 ~195s, 12ep.
- Phase 2 (`scripts/run_phase2_a2db_rebuild.sh`, 대기): a_2 개선 확인 후 i2_mem 재생성 → f_2/gate/scc 재학습 → closed_loop 교체 + verify → publish.
- 미측정: sim domain gap (브릿지 필요). presence FP/FN 이 sim 현상이면 domain gap 이 주범.

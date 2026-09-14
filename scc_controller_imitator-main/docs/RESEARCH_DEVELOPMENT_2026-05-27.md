# controller_imitator — 논문화·연구 발전 방안 (냉정 평가, 2026-05-27)

리뷰어 관점: T-IV / T-VT / T-ITS / ITSC. 톤: cold, 과장 금지.
표기: [확정] 코드·문헌 확인 · [가설] 미검증 주장 · [부정] negative result(폐기/우회).
근거: 문헌 16편 웹 확인(§5) + 본 repo 코드/문서 + 이번 세션 배포 구성.

> 주의(사실 보정): 본 문서의 일부 평가는 자동 문헌조사가 읽은 `docs/ARCHITECTURE.md`(구버전)를 현행
> 배포(minimal/minimal-b emit_model_env)와 교차검증해 보정했다. 보정 지점은 ※ 로 표시.

---

## 0. 한 줄 결론

5개 후보 기여 중 **(ii) profile-conditioned longitudinal imitation** 하나만 잠재 novelty.
현 상태로는 저널 미달 → **ITSC/IV 컨퍼런스가 현실적 1차 타깃**, true input-conditioning + closed-loop
profile ablation 보강 시 **T-IV** 도전. thesis(slip-tolerant lateral) 연결은 물리축 불일치로 억지 —
"perception covariance → SMPC headway chance constraint" 한 다리만 정직, **별도 응용 논문**으로 분리 권고.

---

## 1. 시스템 요약 (평가 대상)

camera 2프레임 + ego + raceline → steer/pedal. seg→a_2(앞차 vL_rel·dist·존재)→f_1(MPC-imitation main, frozen)→f_2(pedal residual)→temporal gate→`clamp(base+g·delta)`. Expert = HPIPM SCC-MPC, (s_min,t_headway) profile label(p1~p5) imitation, vx-log 활성거리.

---

## 2. 후보 기여 5가지 — novelty 평가

| # | 기여 | 새로운 점 | 약점 / reject 사유 | 판정 |
|---|---|---|---|---|
| (i) | MPC-expert distillation을 (준)단일 그래프로 | SCC-MPC를 expert로 한 도메인 특화 IL | "단일 미분 그래프"는 과장 — f_1·seg frozen + 모듈 캐시 분리학습 = stage-wise BC. LbC/MPC-GPS/Differentiable-MPC로 방법론 포화 | 단독 불가 |
| (ii) | **profile-conditioned longitudinal IL** (s_min/headway family + vx-log 활성거리) | 물리 파라미터(s_min,t_headway) 직접 conditioning은 드묾(CIL=discrete cmd, GAIL=latent style). vx-log 활성거리 | conditioning이 **추론시 가변 입력인지** vs label 분기(=단순 multi-task)인지 불명확. vx-log heuristic 이론약함. closed-loop profile별 정량 미검증 | **유일 유망**, 보강 필요 |
| (iii) | learned temporal gate로 rule-based 대체 | lead-presence 학습형 게이팅(TemporalMLP) | ※보정: `learned_bias` 변종은 +3.02 drift로 폐기됐으나, **현행 배포는 distill(scc)·beat(qualifier) TemporalMLP gate가 실제 사용 중**(emit_model_env GATE_CKPT). 단 distill은 EMA 모방이라 "rule 대체 우위"가 약하고, learned의 이점이 정량 입증 안 됨 | 보조 ablation감, 단독 불가 |
| (iv) | 2-image 상대속도 인지(a_2) interpretable intermediate | 2프레임 vL_rel + obj/dist/vL_rel head | ※보정: vL_rel은 **in-distribution Pearson ~0.91, DriverB OOD ~0.18**(corr 0.40은 혼합/OOD 추정치). OOD 일반화·closing recall이 약점. dist(MAE~1.03m)는 양호. 단안 relative speed는 flow/depth로 포화 | 부분 성공, OOD 보강 필요 |
| (v) | modular-train → monolithic-deploy 수치 등가(1e-7) | 모듈 분리학습→단일 .pt + forward 등가 검증, 45Hz | engineering contribution. 리뷰어는 "구현 디테일" 분류 → 단독 reject | §System 1문단 |

종합: (i)(v) 포화/엔지니어링, (iii) 보조, (iv) 부분성공(OOD 약점). **(ii)만 잠재 novelty**, 현 상태 저널 미달.

---

## 3. 묶음 framing (한 편으로)

(i)+(ii) 축, (iv)/(v) 보조 강등, (iii)·(iv-OOD)는 **정직한 ablation/lesson**으로 활용(리뷰어 신뢰↑).

| 항목 | 내용 |
|---|---|
| 주제문 A (추천) | "SCC-MPC가 생성한 (s_min, t_headway) **profile-family label**을 conditioning하여, 단안 2-프레임 vision policy가 추론시 가변 ACC 거동(공격~보수)을 재현하는 **profile-conditioned longitudinal imitation**, 그리고 단안 relative-speed 인지의 OOD 한계·거리 기반 우회를 정직 분석" |
| 주제문 B | "frozen MPC-imitation main + learnable vision **residual pedal head**의 분리학습/통합배포 구조와 lead-presence gating 설계공간(learned vs rule)의 closed-loop 비교" |
| 타깃 venue | A: **IEEE ITSC / IV (컨퍼런스, 현실적 1차)** → 보강 후 **T-IV** 도전. B: ITSC/IV. **T-ITS/T-VT는 현 novelty로 무리** |

---

## 4. (ii)를 논문급으로 — 구체 보강 plan ([가설] 검증 과제)

1. true input-conditioning 확정: profile(s_min,t_headway 또는 p-index)을 **추론시 모델 입력**으로 주입(현재 label 분기면 multi-task). 단일 모델이 conditioning 벡터로 p1~p5 거동 전환됨을 보여야 함.
2. closed-loop profile ablation(정량): 동일 시나리오 lead-cut-in에서 profile별 (min headway 분포, TTC, 감속 jerk, 충돌률) 곡선. "한 perception으로 거동 family 제어" 입증.
3. baseline: rule-based ACC(고정 s_min) / CIL(discrete) / 단일 profile IL 대비.
4. vx-log 활성거리 정당화: heuristic → 간단한 안전거리 유도(반응시간·제동거리)로 형식화하거나 ablation으로 효과 분리.
5. OOD: DriverB류 분포밖 lead에서 profile conditioning이 유지되는지(현 vL_rel OOD 0.18 약점 직시).

---

## 5. 관련 문헌 (웹 확인, [확정])

| 영역 | 대표 | 출처 |
|---|---|---|
| E2E IL 조향 | PilotNet | Bojarski 2016, arXiv:1604.07316 |
| E2E IL mid-level | ChauffeurNet | Bansal 2018, arXiv:1812.03079 |
| transformer fusion | TransFuser | Chitta/Prakash, PAMI 2023, arXiv:2205.15997 |
| privileged distill | Learning by Cheating | Chen, CoRL 2019, arXiv:1912.12294 |
| MPC-guided PS | MPC-Guided Policy Search | Zhang 2016, arXiv:1509.06791 |
| differentiable MPC | Differentiable MPC | Amos, NeurIPS 2018, arXiv:1810.13400 |
| covariate shift | DAgger | Ross, AISTATS 2011 |
| vision ACC | E2E Vision ACC (DRL) | Zhu 2020, arXiv:2001.09181 |
| uncertainty→SMPC ACC | Deep-Ensemble chance-constrained ACC | Li 2024, arXiv:2403.15577 |
| command conditioning | Conditional Imitation Learning | Codevilla, ICRA 2018, arXiv:1710.02410 |
| style conditioning | Car-Following GAIL styles | Sensors 2020, 20(18):5034 |
| modular→diff graph | UniAD | Hu, CVPR 2023 best paper |
| base+residual | Residual Policy Learning | Silver 2018, arXiv:1812.06298 |

핵심 대조군: **LbC**(privileged→vision distill, (i)의 직접 선행), **CIL/GAIL-style**((ii)의 선행), **UniAD/Differentiable-MPC**("단일 그래프" 주장의 정면 선행), **Li 2024**(perception uncertainty→chance-constrained ACC, thesis 다리의 거의 동일 골격), **Residual Policy Learning**(f_2 residual의 정확한 선행).

---

## 6. 박사 thesis 축 연결 — 솔직 평가

thesis: augmented-state UKF → covariance P → SMPC chance constraint, "slip-tolerant path tracking"(lateral).

| 연결 후보 | 평가 | 판정 |
|---|---|---|
| a_2 vL_rel/dist + **uncertainty(logvar 채널) → SMPC headway chance constraint** `P(s≥s_min)≥1-ε` | thesis "covariance P → chance constraint" 패턴과 구조 동형. Li 2024가 거의 동일 골격 → 차별화 필요하나 정당 | 연결 가능(유일) |
| "slip-tolerant path tracking"(lateral/tire) ↔ 본 시스템(longitudinal ACC, f_1 frozen) | 물리축 다름(종 vs 횡). f_1 안 건드림 | 연결 금지(억지) |
| UKF augmented-state ↔ NN perception | 본 시스템 추정이론 없음 | 연결 금지 |

권고: thesis main(slip-tolerant lateral)에 끼워넣지 말 것. "perception covariance → SMPC headway chance constraint"는 thesis의 **longitudinal 확장 별책**(ITSC/T-IV 응용 논문)으로 분리. 단 그 다리도 vL_rel OOD(0.18)부터 고쳐야 성립.

---

## 7. 실행 우선순위 (연구로 발전시킬 경우)

1. [필수] (ii) conditioning을 true input-conditioning으로 확정 + closed-loop profile ablation → ITSC/IV 1편.
2. [선택] a_2 uncertainty head를 SMPC chance constraint headway에 연결(Li 2024 차별화: profile-conditioned + camera-only) → thesis 별책.
3. [폐기/보조] learned gate·vL_rel OOD는 negative/limitation 절로만.
4. T-ITS/T-VT 직행 금지(현 novelty 미달).

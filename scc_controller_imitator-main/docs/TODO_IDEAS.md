# TODO / 아이디어 정리 (2026-05-26 야간 세션 종합)

대화에서 "나중에 / 해보면 좋을" 로 나온 것 + 진행 중 + 막다른 길 정리. status: confirmed(확정) / hypothesis(가설) / in-progress / dead-end.

---

## A. 진행 중 (야간 자율, 2026-05-26)

| 작업 | 내용 | status |
|---|---|---|
| multi-frame vrel sweep | 10-tick granularity, ≤5프레임, span≤50 조합 sweep. vL_rel/dist/obj 정확·안정 평가 → 최적 조합 | in-progress (로컬 GPU) |
| SLURM 구축 준비 | SSH키·컨테이너·repo/data NFS sync·sbatch runbook. 제출은 사용자가(키 없음) | in-progress (문서) |
| 규정 준수 + 예선 branch | beat 모델 단일 모델파일(§1.1)·gate forward 통합(§1.3)·clean/minimal branch·closed-loop test | in-progress (worktree, push 안 함) |

다음 (위 결과 의존):
- vrel 신뢰(corr↑) 확인되면 → **vrel 기반 ACC pedaling 재학습** (distbrake 우회 졸업). f_2 cache 학습.
- backbone 검증 (시간 남으면): b0 등 경량 backbone 위에서 obj/vrel/dist 견디나 (a_2-on-b0 ablation).

---

## B. 결정/검증 대기 (다음 세션 우선)

- **a_2 v8 (균형 275K) cascade 채택**: v8 가 speedway obj/vrel 개선(F1+4.5, vL_rel+0.07, DriverB 무손상) 확정. → i2_mem 6개 v8 로 재생성 → f_2 재학습(spd distbrake 포함). 배포 모델 직접 개선. **단 v8 vs vrel-sweep 최적 모델 중 무엇을 cascade base 로 할지 sweep 결과 보고 결정.**
- **i2_mem 재생성 후 `verify_i2_live_vs_cache` 재확인**: fix 에이전트가 i2_mem 6개 재생성함 → 배포 불변식(seg fx640_v3 + ego_mot=0, cos 1.0) 깨지지 않았는지 확인 필요 (배포 의존).
- **gate-stability 수치 clean 재산출**: flip@0.3 14.9% 는 sangam cache 96% zero-fill artifact 였음(이제 fixed). EMA/distill/beat 의 flip 을 clean i2_mem 으로 재계산해야 진짜 이득 확정.
- **0523_validation_short 독립 held-out 화**: 현재 train/val random split → leakage 우려. 신규 raw 0523_validation_short 를 처리해 독립 평가셋으로. (방법론적 가치 큼.)
- **closed-loop beat gate 라이브 검증**: live 는 매프레임 fresh 추론(zero-fill 없음) → 원거리(40-60m) lead flicker·no-lead FP 직접 관찰.

---

## C. 신규 데이터 (raw 있으나 미처리)

| raw | 상태 | 가치 |
|---|---|---|
| sangam/speedway 0522_40m | 부분(synth/bank만) → i1/prebake/aux/i2 필요 | 40m headway 시나리오 다양성 |
| sangam/speedway 0523_pertO_agent3 | 신규 raw만 | 다양성 |
| speedway 0523_validation_short | 신규 raw만 | 독립 held-out (B 참조) |

- pertO/pertX GT 파이프라인 결함: prebake curr_key ~89% empty(SCC GT 부재) + aux 전부 0 + inlane_gt 0%. **pertO 가 진짜 무-lead 인지 vs GT-gen 실패인지 미결**(raw OBJECT_INFO 파일은 존재). 진짜 lead 있으면 GT 재생성 가치, 없으면 negative 로 충분.
- DriverB SAM mask (1094/135113 생성 후 일시정지, resume 가능) → 완성 시 seg 재학습에 logged-real 다양성 추가 → vehicle IoU 개선 가능성(hypothesis).

---

## D. 규정 (§ = A1 Challenge)

- §1.1 단일 모델파일 — seg+a2+f1+f2+gate 통합 (진행 중).
- §1.3 differentiable single chain — gate filter 를 forward 통합(numpy/deque/EMA glue 제거) (진행 중).
- §1.5 reference line — **KIAPI 제공 reference line 사용 필요**, 자가 dt0p1s raceline 추종 + mild/high 외부 switch 제거. ★해석/결정 필요.
- §1.7 gating 허용 — beat gate OK.

---

## E. backbone / seg

- b0 경량화: latency 2.14x, mIoU 0.897(−0.024), vehicle IoU 0.811(−0.055). **교체 가부 = a_2-on-b0 ablation 후 결정**(vehicle IoU 저하가 obj/gate 무너뜨리나). docs/REPORT_seg_backbone_lite.md.
- lite0(2.37x)/mobilenetv3(3.05x, 960ch caveat) 대안 후보 보존.
- 교체 시 i1/i2/a_2/f_2 full cascade 재학습 (비가역, 신중).

---

## F. 막다른 길 (재시도 금지 — 근거 있음)

- **gate soft-aggregation (mean/topk/logsumexp)**: 실데이터에서 mean/logsumexp FP 92% 폭발, topk recall 34%. lead 신호가 1-bin 국소라 공간평균이 희석. → 평활은 시간축(EMA/learned filter)이 정답. (gate soft-ramp 도 band 넓혀야 효과+FP 대가라 열등.)
- **single-frame a_2**: 2-frame 보다 flicker 더 큼(temporal context 가 안정성 도움). 
- **yaw_shift (hand-crafted token roll)**: 고-yaw subset(|dx|≥2)에서도 obj/dist/vL_rel ON==OFF (noise). 제거 가능, 규정 +. (게다가 cache 학습 경로가 d_yaw=0 하드코딩이라 v7b 는 yaw_shift 를 쓴 적도 없었음.)
- **EMA 졸업을 gate-level stateless trick 으로**: 불가. 안정성 지배변수가 EMA(temporal) 자체. → learned temporal filter(in-graph)로 대체가 정답.

---

## G. 인프라 / 데이터 매칭 교훈 (수정됨)

- a_2 학습 데이터 매칭 버그 2건 fix: pertO run-filter(B1), spd cache↔aux 날짜불일치(B2) → join 211K→275K. silent drop 가드(join/aux<0.5 fail-fast) 추가.
- i2_mem zero-fill fix: build_i2_cache 가 prebake∩cache 만 처리하던 것 → cache 전프레임 처리(6개 재생성 100%).
- ★교훈: dataset join 은 항상 매칭률 검증(silent 0% drop 위험). cache↔aux 날짜·run·key 규약 일치 assert.

---

## H. 장기 (이 프로젝트 범위 밖, 참고)

- 학습 인프라를 SLURM 로 — 본선 대량 ablation 의 표준 경로. (단일 3080Ti GPU 경합 졸업.)

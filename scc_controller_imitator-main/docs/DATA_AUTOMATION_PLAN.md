# 데이터/모델 현황 "마법사" 도구 — 설계 (계획, 미구현)

목표: 이 repo 의 **전체 상태를 한 화면**에서 — (1) 모델 스택과 각 모듈 학습 상태, (2) 데이터셋별 캐시 처리 현황,
(3) closed-loop 배포 준비도. 신규 에이전트/사람이 `python scripts/data_pipeline.py status` 한 번으로 파악.

> 현재 부분 구현: `data_pipeline.py scan` (raw 인벤토리 + 데이터셋별 단계 처리상태 표). 아래는 확장 설계.

---

## 1. 3개 뷰

### A. 모델 스택 뷰 (registry)
소스: `data/ckpts/CURRENT_BEST.json` + `data/ckpts/*.pt` + 학습 로그.
표시: 모듈(seg/a_2/f_1/f_2) × [ckpt 파일, metric(mIoU/corr/val_pedal), 학습일, frozen|trainable, 배포중 여부].
```
모듈   ckpt                          metric          상태
seg    seg_pretrain_fx640_v3         mIoU 0.921      frozen ✓배포
a_2    a2_v7b_d128_alpha075          vL_rel corr0.40 frozen ✓배포  ⚠vL_rel 약함
f_1    main_prof654                  steer corr0.88  frozen ✓배포
f_2    prof635_side_distbrake_v1     val_pedal~0.52  ✓배포(brake_only) / legacy v5(signed)0.024·v6 0.044
```

### B. 데이터 인벤토리 뷰 (현 scan 확장)
소스: raw 스캔(track/policy) + 캐시 디렉토리.
표시: 데이터셋 × 파이프라인 단계 완성도 매트릭스 + 크기 + run 수.
```
데이터셋(track_policy)      runs  manifest i1  prebake i2  inlane pedal  status
sangam_0520_..pertX        12    O        O   O       O   O      O      DONE
speedway_0523_validation   3     .        .   .       .   .      .      NEW
```
+ 집계: DONE/PARTIAL/NEW 개수, 총 frame/sample 수, 디스크 사용량.

### C. 파이프라인 / 배포 준비도 뷰
- 글로벌 단계(seg pretrain, a_2) 최신 여부 + 마지막 갱신일.
- closed-loop 준비: 배포 ckpt 4종 존재 + **i2_mem 일관성 자가검증**(seg=fx640_v3, ego_mot=0) pass/fail.
- raceline(sangam/speedway) + pedal_map 존재.

---

## 2. 데이터 소스 / 판정

| 정보 | 소스 |
|---|---|
| raw 데이터셋 목록 | `raw_root/{track}/{policy}/{run}` 스캔 (SSD 우선) |
| (track,policy)↔logical name | `manifest_i1cache_*.json` 역매핑 |
| 단계 완성도 | 캐시 파일 존재(glob): i1_tok_cache/prebake/i2_mem_cache/inlane_gt/pedal_label |
| 모델 metric | CURRENT_BEST.json + outputs_multi/*/train.log 파싱 |
| 배포 일관성 | verify_i2_live_vs_cache 로직(샘플 cosine) |

신규 판정: manifest 없음 = NEW. manifest 있고 하류 일부 결손 = PARTIAL. 전부 = DONE.

---

## 3. 출력 형태

1. **CLI 표** (`scan`/`status`) — 위 3 뷰. 색상(DONE 녹/PARTIAL 황/NEW 적).
2. **HTML 대시보드**(옵션) — `dashboard_status.py` 확장. 매트릭스 + 모델 트리 + 타임라인.
3. **JSON** (`--json`) — 다른 도구/에이전트 소비용.

---

## 4. 자동 처리 (incremental) — 신중 모드

`process` 서브커맨드(현 stub): NEW/PARTIAL 데이터셋의 **빠진 단계만** 의존성 순서로 실행(idempotent).
- 기본 `--dry-run`(계획만). 실제 실행은 GPU/NFS I/O 큼.
- manifest 없으면(control GT 결정 필요: logged vs MPC synth) 자동화 범위 밖 → 수동 manifest 후 재실행.
- seg backbone 재학습은 **자동 금지**(전체 cascade) — 명시적 글로벌 작업으로만.

처리 단계 명령(현 `_plan_stages`): compute_i1_tok_cache → prebake_iy_dataset → build_i2_cache → build_inlane_gt_cache → gen_ttc_pedal_labels.

---

## 5. 범용성 (타 머신)

- 캐시 출력 `SCC_DATA_DIR`(env) > AILabDatasetLocal > `repo/data/local`(gitignored).
- raw `SCC_RAW_ROOT`(env) > AILabSSD > AILabDataset 자동 탐지.
- AILabDatasetLocal 없는 머신도 동작 — 캐시는 repo 안 생성, git 미추적.

---

## 6. 구현 단계

1. (완료) `data_pipeline.py scan` — raw 인벤토리 + 단계 매트릭스.
2. 모델 registry 뷰 (CURRENT_BEST.json + ckpt 메트릭 파싱) — `status` 서브커맨드.
3. 배포 준비도 자가검증 (i2_mem cosine) 통합.
4. `--json` + HTML 대시보드.
5. `process` 실제 실행(현 dry-run plan) 안정화 + per-stage 로그.

우선순위: 2~3(현황 가시화)이 핵심. 4~5 는 부가. seg/cascade 는 수동 유지.

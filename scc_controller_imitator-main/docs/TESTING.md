# 테스트 / 검증 (무엇을 어떻게)

closed-loop(ROS 주행) + 오프라인 검증(ROS 없이). 에이전트가 "테스트는 뭘 하면 되는지" 보는 문서.

---

## 1. closed-loop (ROS, 실차/시뮬)

```
bash scripts/run_closed_loop_prof635.sh <sangam|speedway> [mild|high] [--viz|--viz-seg|--viz-a2|--viz-all]
# 예
bash scripts/run_closed_loop_prof635.sh speedway high           # 공격적 맵
bash scripts/run_closed_loop_prof635.sh sangam mild --viz-all   # 안정 맵 + 전체 debug 패널
# map 생략 시 mild. --rosmaster=http://<ip>:11311 로 master 지정.
```
- **python**: 런처가 `PYTHON` env > **현재 활성 conda 환경**($CONDA_PREFIX, torch 보유 시) > `ml_con`(이 머신 fallback) > `python3` 순 자동 선택 + torch 체크.
  타 머신은 `conda activate <torch env>` 후 실행하면 그 환경 사용. 강제 지정 `PYTHON=/path/to/python`.
- **TRACK×MAP**: raceline = `data/control/{track}_{map}_{P}_{M}.csv` (dt0p1s + pit polygon 6행, cutoff P/M=파일명).
- 사전: roscore(또는 시뮬/차량 master). 없으면 `ConnectionRefusedError` → `roscore &` 또는 `--rosmaster`.
- 구독: 카메라 `/app/perc/camera_1`, ego `/app/loc/vehicle_state`. publish: `command_steer`(deg), `command_pedal_accel/brake`(0~1).
- env: `MODEL_TYPE=prof635`(기본), `TRACK`, `MAP=mild|high`, `SIDE_OFF=1`(f_2 끔=f_1 단독, 비상), `GATE_THRESH=0.3 GATE_SCALE=5.0`.
- 정상 가동 시 20틱마다: `[prof635] {ms}ms i0=.. steer=.. base=.. delta=.. obj_amax=.. gate=.. -> pedal=.. (acc/brk)`.
  - no-lead: gate≈0, pedal≈base. lead 접근: gate≈1, pedal 음수(brake). i0=최근접 raceline idx(+pit 여부).
- raceline 추출: `raceline_dt0p1s.sample_traj40` — ego 최근접(pit polygon 안→[0,P] / 밖→[P,M]) 부터 40점 forward(연속 라인) + 라인끝 직선 외삽 → build_traj6.
- 코드: `scripts/closed_loop_node.py` (prof635 분기 self-contained).

---

## 2. 오프라인 검증 (ROS 불필요, 이 머신/타 머신)

| 스크립트 | 검증 내용 | 합격 기준 |
|---|---|---|
| `test_prof635_infer.py` | 모델 I/O(prebake→steer GT 일치) + raceline 파이프라인 유한성 | steer MAE<3°, corr>0.8 (현 2.61°/0.93) |
| `test_prof635_closedloop.py` | 노드 경로(raceline 추출→build_traj6→model) sangam/speedway | 양 트랙 steer/pedal 유한·정상 |
| `verify_i2_live_vs_cache.py` | live i2_mem 이 학습 캐시와 일치(cosine) | fx640_v3+ego_zeros = cos 1.0000 (다른 조합 0.80~0.93) |
| `ablate_traj_speed.py` | base=traj-speed artifact 진단 (flat 시 base brake 사라짐) + lead/no-lead loss 분해 | base −0.48→+0.28, f_2 image-driven |
| `viz_prof635_gradcam.py` | brake → image grad-cam + gate/base/delta 분해 (lead-TTC) | a_2 obj 가 lead 국소화, gate≈1 |
| `eval_a2_holdout.py` | a_2 객체 인지/거리/vL_rel held-out 평가 | obj prec, dist MAE, vL_rel corr |

실행 예:
```
python scripts/test_prof635_closedloop.py        # 양 트랙 PASS 확인
python scripts/verify_i2_live_vs_cache.py        # i2_mem 일관성 (이미지 NFS 접근 필요)
python scripts/ablate_traj_speed.py --model-ckpt data/ckpts/prof635_side_distbrake_v1.pt
```

---

## 2.5 raceline 맵 / waypoint 검수 도구 (GUI, 본인 환경)

dt0p1s 맵 제작·검수 파이프라인. cutoff P(피트끝)/M(피트머지) 와 pit polygon 을 사람이 검수해 확정.

| 스크립트 | 용도 |
|---|---|
| `pick_pit_polygon.py --track <t>` | 지도 위 클릭으로 pit 6-vertex polygon 찍기 → `pit_polygon_{track}.json` |
| `pick_raceline_index.py --track <t> --map <m>` | 클릭→최근접 raceline index 표시 (P/M 인덱스 검수) |
| `viz_pit_cutoffs.py` | P/M 후보 + pit-zoom index 라벨 (static, P/M 결정용) |
| `pick_waypoint_preview.py --track <t> --map <m> [--selftest]` | 클릭(ego 가정)→추출된 40 waypoint 표시. `--selftest`=클릭없이 static 검증 |
| `verify_pit_boundary.py` | pit 경계 off-line ego 의 in_pit/최근접 i0 할당 검증 (static) |
| `build_raceline_map.py` | dt0p1s + polygon + CUTOFFS → 통합맵 `{track}_{map}_{P}_{M}.csv` 생성 |

- cutoff `CUTOFFS` 는 `src/utils/raceline_dt0p1s.py` (또는 통합맵 파일명) 에 정의. 신규 맵: dt0p1s 받기 → polygon 찍기 → P/M 검수 → `build_raceline_map.py` → 커밋.
- 통합맵 자체완결: raceline(t;x;y;psi) + 마지막 6행 pit polygon, cutoff=파일명. closed_loop 가 glob 으로 로드.

---

## 3. 성능 진단 (ACC 품질)

`ablate_traj_speed.py` + vL_rel 상관 분석으로 접근/크루징/멀어짐 별 제동 + closing modulation 점검.
현 상태: 중거리 접근 제동 OK, **closing modulation 약함**(상대속도 지각 한계). 상세 [STATE_AND_LESSONS.md §3].

GT vrel 주입 격리 실험: `train_prof635_side.py --inject-gt-vrel` → 학습 후 delta↔closing 상관 측정. f_2 capacity vs a_2 perception 병목 구분용.

---

## 4. 학습 모니터링

- wandb: project `scc-controller-imitator`. val/pedal_l1, val/steer_mae_deg.
- 로컬 로그: `outputs_multi/{run}/train.log`, best `outputs_multi/{run}/best.pt`.
- f_2 학습은 global val_pedal 이 no-lead(84%)에 지배됨 → **lead-specific 평가 필수**(lead-TTC L1, gradcam). global loss 낮다고 제동 잘하는 것 아님.

# scc_controller_imitator 프로젝트 구조 요약

본 문서는 `controller_imitator/PROJECT_STRUCTURE_BRIEF_KR.md` 의 **이미지 확장판** 이다.
중복되는 계약은 원본을 그대로 인용하고, 추가/변경된 부분만 기술한다.
세부 설계는 `DESIGN.md` 참고.

---

## 1. 목적

기존 `controller_imitator` 가 (trajectory, ego) → (steer, accel/brake) 를 학습했다면,
본 프로젝트는 **카메라 전방 이미지 2장 (현재 + Δyaw 회전 보정된 이전)** 을 추가 입력으로 받아
선행차 인지 등의 SCC 기능을 신경망이 얻도록 한다.
**출력 채널/규약은 controller_imitator 와 완전 동일**.

---

## 2. 입력/출력 계약 차이

### 2.1 입력 (추가)
| key                       | shape              | 비고                                  |
|---------------------------|--------------------|---------------------------------------|
| `image_curr`              | `(B, 3, 250, 1280)`| ROI strip y=[235, 485), ImageNet 정규화 |
| `image_prev_rot`          | `(B, 3, 250, 1280)`| t-Δt 프레임을 Δyaw 만큼 회전 보정 후 동일 ROI |

기존 키 (controller_imitator 와 동일, **horizon 만 30**):
| key                       | shape         | 비고                              |
|---------------------------|---------------|-----------------------------------|
| `trajectory_points_local` | `(B, 30, 2 or 3)` | jiwon_e2e `wp_rel_gt` 누적 결과 (ego frame) |
| `trajectory_time`         | `(B, 30, 1)`  | `input_time_to_traj != "none"` 시   |
| `ego_info`                | `(B, 5)`      | `[vx, vy, ax, ay, yaw_rate]` (vehicle frame, SI: m/s, m/s², rad/s) |

### 2.2 출력
controller_imitator 와 **완전 동일**:
- `target = "steer" / "accel_brake" / "steer_accel_brake"` (기본: `steer_accel_brake`)
- `output_shape = "control" / "control_ego"`
- `(B, 30, D_out)` 텐서. 채널 순서 `[control...] + (옵션) [ego x, y, heading, speed]`

### 2.3 forward 시그니처
```python
output = model(
    trajectory,               # (B, 30, F_traj)
    ego,                      # (B, 6)
    image_curr=...,           # kwarg
    image_prev_rot=...,       # kwarg
)
```
- 첫 2 positional 은 controller_imitator 의 `forward(traj, ego)` 계약 유지.

---

## 3. 새 모델 추가 시 체크리스트
- [ ] `src/model/image_encoders/` 또는 `src/model/scc_predictors/` 아래 클래스 추가
- [ ] 해당 패키지 `__init__.py` 에 export
- [ ] `src/model/models.py::get_model` 분기 추가
- [ ] 출력 채널 순서 `[control...] + (옵션) [ego 4-d]` 유지
- [ ] `target × output_shape` 조합 모두 지원 (assertion 으로 단일 target 강제 금지)
- [ ] CSV 자동화 컬럼에 신규 하이퍼파라미터 표현

---

## 4. controller_imitator 와의 연결

- `train_multi_output_*.py` / `inference_multi_output.py` 를 fork 시:
  - `from src.datamodule.imitator_dataset_multi` → `from src.datamodule.scc_dataset`
  - `from src.model.models import get_model` 은 그대로 사용 (본 패키지가 같은 이름 제공)
  - dataset 이 추가로 반환하는 `batch["image_curr"]`, `batch["image_prev_rot"]` 를
    `preprocess_data_for_preload` 단계에서 모델 kwarg 로 전달
  - `tfr_seq_length=30` 으로 설정 (controller_imitator 기본 40 에서 변경)
- CSV 자동화는 `training_status/ML_Experiment_Table.csv` 에서
  `model_name ∈ {TwoImgTrajEgoController, ...}` row 만 본 프로젝트로 라우팅.

---

## 5. 데이터 소스

`jiwon_e2e` 의 자산 재활용:
- 원본: `jiwon_e2e/e2e_racing/data/raw/sangam_track/<policy>/<run_id>/`
  - `CAMERA_1/<tick>_Intensity.jpeg` — 1280×720 JPEG (전방)
  - `EGO_INFO/<tick>.txt` — yaw, vel, accel, brake, steer 포함
- 인덱스: `jiwon_e2e/e2e_racing/data/processed/index.json` + `splits.json`
- 학습에 쓰는 정책: **`MultiAgnet` (6,877 frame, 즉시 사용 가능)**,
  추후 `MultiAgentSCC` 인덱싱 후 추가

라벨 변환:
- `target_steer = steer_deg / 40.0` (controller_imitator `loss_weight_steer=40` 컨벤션)
- `target_accel_brake = accel - brake ∈ [-1, +1]` (1-d 통합 신호)

---

## 6. 위험 요소

1. **horizon 30 vs 40 불일치** — controller_imitator 기존 결과와 직접 비교하려면 양쪽 모두 동일 길이로 재학습 필요.
2. **시간 동기화** — CAMERA_1 과 EGO_INFO 의 tick 이 정확히 정렬됨을 가정. `dt_per_tick_ms` 가 run 별로 다를 수 있어 dataset 초기화 시 검증 필수.
3. **이미지 분포** — MORAI 시뮬레이션 카메라. 실차 카메라 적용 시 도메인 갭 발생 가능.
4. **메모리** — 250×1280 두 장이 들어가므로 batch_size 는 controller_imitator 대비 작아야 함 (16~32 권장).
5. **prev 부재** — 첫 프레임은 `valid_mask=0` 으로 학습에서 제외.

---

## 7. 다음 단계 우선순위
1. (DESIGN.md §11 의 구현 순서 1~9) 데이터셋 + 모델 코드 작성
2. controller_imitator 의 train script fork → image kwarg 전달
3. `MultiAgentSCC` raw 인덱싱 (`scripts/index_multiagent_scc.py`)
4. CSV 자동화 연결 (`ML_Experiment_Table.csv` 컬럼)
5. ablation: image off vs on, prev frame off vs on, ROI 크기

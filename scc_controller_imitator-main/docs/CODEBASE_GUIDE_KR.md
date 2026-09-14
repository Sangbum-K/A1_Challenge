# SCC Controller Imitator 코드베이스 가이드

> 작성 기준: 2026-09-12의 현재 소스 트리  
> 대상 경로: `scc_controller_imitator-main/`  
> 원칙: 주석이나 과거 보고서와 구현이 다르면 **실행되는 코드**를 기준으로 설명한다.

## 1. 한눈에 보는 시스템

이 저장소는 카메라와 차량 상태를 받아 다음 두 제어 출력을 만드는 MORAI용 종단간 제어 시스템이다.

- 조향: 기존 Professor controller `f₁`이 raceline trajectory와 ego state로 계산한다.
- 가감속: 영상 perception `i₁ → a₂/i₂`와 side controller `f₂`가 lead vehicle 상황을 반영해 `f₁`의 기본 pedal을 보정한다.

현재 `prof635` 운영 경로의 데이터 흐름은 다음과 같다.

```text
MORAI camera (720×1280 RGB)              vehicle state
            │                                  │
            ├─ ROI y=235:485, ImageNet norm    ├─ ego6
            │                                  └─ raceline → traj40 → traj6
            ▼
seg-pretrained EfficientNetV2-S backbone
            │ last feature ≈ (C=256,H=8,W=40)
            ▼
vertical average pool
            │ i₁ raw token: (40,256)
            ▼
wall-clock token history: current + 20 past samples
target ages = 0.05, 0.10, ..., 1.00 s
            ▼
Multi-frame a₂
  current 40 tokens = query
  20×40 history + ego token = key/value
            ├─ a₂ logits: (40,7)
            └─ i₂ memory: (40,128)
                         │
traj6 + ego6 ──► frozen f₁ (#654) ──► steer, base pedal
                         │
i₂ memory + ego6 + base pedal ──► f₂ ──► pedal delta
a₂ in-lane evidence ─────────────► gate
                         │
                         ▼
final pedal = clamp(base + gate × scale × delta)
```

여기서 `a₂`는 두 역할을 동시에 수행한다.

1. 각 lateral bin의 물체/거리/속도를 예측하는 보조 perception head
2. `f₂`가 직접 읽는 128차원 image memory `i₂` 생성

따라서 a₂의 성능은 7채널 예측 정확도뿐 아니라 최종 pedal과 폐루프 안전 지표로 평가해야 한다.

## 2. 현재 기준 핵심 tensor 계약

| 단계 | tensor | shape | 의미 |
|---|---|---:|---|
| 카메라 원본 | RGB image | `(720,1280,3)` | MORAI camera frame |
| 입력 ROI | normalized image | `(B,3,250,1280)` | `y=235:485`, ImageNet 정규화 |
| backbone last map | feature map | 보통 `(B,256,8,40)` | EfficientNetV2-S stride-32 feature |
| i₁ raw | lateral tokens | `(B,40,256)` | 세로 축을 `(1,40)`으로 pooling, projection 전 cache |
| a₂ history | token list | `21 × (B,40,256)` | current + 20 past frames |
| fused feature | fusion memory | `(B,40,256)` | temporal cross-attention 출력 |
| i₂ | image memory | `(B,40,128)` | `f₂`에 공급 |
| a₂ output | logits | `(B,40,7)` | 아래 7채널 계약 |
| trajectory | `traj40` | `(B,40,3)` | 0.1 s 간격 `[x_forward,y_left,speed]` |
| f₁ trajectory | `traj6` | `(B,6,3)` | 시각 `0.05…1.30 s`, `[-y,x,speed]` |
| ego | `ego6` | `(B,6)` | `[vx,vy,ax,ay,tire_angle,yaw_rate]` |
| controller output | steer/pedal | `(B,)`, `(B,)` | steer degree, pedal `[-1,1]` |

기본 a₂ 7채널은 다음 순서다.

| index | 이름 | 설명 |
|---:|---|---|
| 0 | `obj` | object logit |
| 1 | `vL_abs` | ego 좌표계 longitudinal absolute velocity |
| 2 | `vY_abs` | ego 좌표계 lateral absolute velocity |
| 3 | `vL_rel` | ego 대비 longitudinal relative velocity |
| 4 | `vY_rel` | ego 대비 lateral relative velocity |
| 5 | `prox` | `exp(-distance/20)` |
| 6 | `dist` | object distance, 최대 80 m 기준 |

실험용 uncertainty 모델은 object 1채널과 각 회귀값의 mean/log-variance를 interleave한 13채널을 쓴다. 7채널 checkpoint와 직접 혼용하면 안 된다.

## 3. 최상위 디렉터리 설명

| 경로 | 역할 | 현재 중요도 |
|---|---|---|
| `src/` | 학습 데이터셋, perception/controller 모델, 공통 유틸리티 | 핵심 |
| `scripts/` | 캐시 생성, 학습, 평가, 시각화, live ROS node | 핵심 |
| `cpp/` | SCC expert label/data 합성용 C++ replanner와 batch generator | 핵심 |
| `mpc/` | 횡/종방향 MPC 및 SCC 합성에 쓰인 A1 Rule fork | 핵심이지만 외부 빌드 의존성 큼 |
| `moraibridge/` | MORAI UDP ↔ ROS1 topic bridge | live 실행 핵심 |
| `multicar/` | 여러 차량의 bridge/controller와 port/namespace 관리 | 다차량 실행용 |
| `data/` | checkpoint, raceline, pedal map | 실행 자산; 소스와 함께 버전 고정 필요 |
| `config/` | 초기 설정 파일 | 일부는 현재 코드보다 오래됨 |
| `docs/` | 아키텍처/실험 보고서 | 참고용; 서로 다른 개발 시점의 문서가 공존 |
| `legacy/` | 이전 SCC/image-controller 구현 | 비교·복구용, 신규 기준으로 사용하지 않음 |
| `external/` | vendored/generated ROS message package | 인터페이스 의존성 |
| `slurm/`, `slurm_script/` | 서버 학습 job launcher | 환경 종속 |
| `.github/` | CI/workflow | 저장소 자동화 |

## 4. `src/`: 모델과 데이터 계약

### 4.1 Perception 모델

#### `src/model/image_models/seg_pretrain_model.py`

- `SegPretrainModel`은 timm `features_only` backbone의 5개 stage를 사용한다.
- decoder는 DeepLabV3+가 아니라 transpose-convolution과 skip concat을 쓰는 U-Net/FPN-lite 구조다.
- class는 background, road, vehicle, wall의 4개다.
- 학습 후 decoder는 버리고 `backbone.*` state만 downstream tokenizer로 가져간다.

#### `src/model/image_models/a2_model.py`

한 파일 안에 서로 다른 세대의 a₂가 공존한다.

- `A2Model`: current/previous 두 프레임용 구형 경로
- `A2ModelMultiFrame`: 현재 `prof635` 운영 경로의 N-frame 모델
- `A2ModelMultiFrameHR`: 80-bin stride-16 feature를 합치는 실험형
- `A2ModelSingleObjDist`: current-only, object/distance 2채널 실험형

`A2ModelMultiFrame`의 실제 동작은 다음과 같다.

1. cache의 raw i₁ `(40,C_b)`를 공유 `1×1 Conv1d`로 256차원에 projection한다.
2. 각 frame에서 lateral self-attention을 공유한다.
3. 현재 40 token을 query로, ego token과 과거 `(N-1)×40` token을 key/value로 cross-attention한다.
4. post lateral self-attention을 적용한다.
5. `256→128` projection으로 i₂를 만든다.
6. linear head가 7채널 a₂ 출력을 만든다.

운영 설정은 `N=21`, 과거 offset `5,10,…,100 tick`, 1 tick `=10 ms`다. 즉 50 ms 간격으로 1초를 본다. 학습/cache/live 모두 yaw shift off, `ego_motion=0`으로 맞춘 경로가 기본이다.

#### `src/model/image_models/prof635_side_model.py`

- `Prof635Main`: 파일/클래스 이름은 635지만 배포 checkpoint 설명상 실제 사용 main은 #654다.
- `f₁` main은 frozen이며 trajectory와 ego로 steer와 base pedal을 예측한다.
- `f₂`는 하나의 query `(ego embedding + base-pedal embedding)`가 i₂ 40 token을 cross-attention한다.
- `use_traj_in_side=False`가 의도된 기본이므로 side branch는 trajectory를 직접 보지 않는다.
- `delta_mode`는 `brake_only`, `signed`, `raw`가 있다.
- 내부 gate는 in-lane bin `16:26`의 object logit과 object-probability로 mask한 closing speed를 사용한다.

주의: 주석 일부는 gate를 `sigmoid∈(0,1)`로 설명하지만 현재 구현은 `g = 2×sigmoid(logit)`이므로 범위는 `(0,2)`다. 실험과 로그에서 `sigmoid(logit)`과 실제 제어 multiplier `g`를 구분해야 한다.

#### `src/model/a1_e2e_model.py`

분리된 backbone/a₂/f₁/f₂/temporal gate를 하나의 `state_dict`와 미분 가능한 `forward`로 포장하려는 제출용 통합 모델이다.

그러나 현재 구현은 `A2ModelMultiFrame`이 아니라 구형 `A2Model`을 생성하고 current/previous 2-frame만 처리한다. 외부 입력 `obj_window`도 caller가 별도로 만들어야 한다. 따라서 현재 21-frame `closed_loop_node.py`와 수학적으로 동일한 배포 모델이라고 볼 수 없다. 통합 모델 테스트는 “하나의 state_dict와 gradient 연결”은 검사하지만 21-frame live parity를 보장하지 않는다.

#### 기타 image 모델

- `iy_model.py`: 과거 i₂/image memory를 controller transformer 또는 side branch에 결합하던 세대
- `image_scc_models_v2.py`: raw two-image 기반의 더 오래된 end-to-end 실험형

신규 실험의 production baseline은 `prof635 + A2ModelMultiFrame(21f)`로 고정하고 이 모델들은 historical baseline으로만 다루는 편이 안전하다.

### 4.2 데이터셋

| 파일 | 역할 |
|---|---|
| `seg_pretrain_dataset.py` | image와 4-class pseudo segmentation mask 로드 |
| `a2_pair_dataset.py` | image pair와 a₂ target 조합 |
| `prof635_dataset.py` | prebake trajectory/ego, i₂ cache, pedal label, 선택적 vrel/gate target 결합 |
| `scc_synth_pair_dataset.py` | 합성된 SCC pair 학습 자료 |
| `scc_synth_iy_dataset.py` | 이전 IY 계열 합성 자료 |
| `controller_imitator_aug_dataset.py` | controller imitation augmentation 자료 |

현재 주요 training script들은 여러 dataset을 `ConcatDataset`한 뒤 **frame 단위 `random_split`**을 한다. 같은 run의 매우 인접한 frame이 train과 validation에 함께 들어갈 수 있어 validation 점수가 일반화 성능을 과대평가할 가능성이 있다.

### 4.3 공통 유틸리티

- `raceline_dt0p1s.py`, `raceline_search.py`: 차량 위치에서 0.1 s 간격 40-point trajectory 추출
- `prof635_infer.py`: `traj40→traj6`, `ego5→ego6` 변환과 추론 보조
- `pedal_map.py`, `pedal_label.py`: longitudinal label/pedal 변환
- `temporal_gate.py`: `LearnableIIR`, temporal Conv1d, temporal MLP
- `ckpt_registry.py`: checkpoint 별칭/registry 접근. 일부 script가 기대하는 `data/ckpts/CURRENT_BEST.json`은 현재 tree에 없다.

`src/model/models.py`는 존재하지 않는 active module인 `.image_models.image_scc_models`를 import한다. 이 모듈은 legacy 쪽으로 이동한 것으로 보이며, 이를 import하는 `scripts/sanity_check_env.py`도 현재 상태 그대로는 실패한다.

## 5. `scripts/`: 실제 파이프라인

120개가 넘는 script를 목적별로 묶으면 다음과 같다.

### 5.1 데이터와 GT 생성

| 단계 | 대표 script | 결과 |
|---|---|---|
| segmentation pseudo label | `generate_image_seg_masks.py` | 4-class masks |
| a₂ target | `prepare_a2_aux_targets.py` | image별 `(40,7)` target |
| pair/join manifest | `build_pair_manifest.py`, `build_meta_from_framelist.py` | dataset index |
| run bank/trajectory | `prepare_new_run_bank.py`, `prepare_logged_trajs_v2.py` | standardized run data |
| i₁ cache | `compute_i1_tok_cache.py` | raw `(40,C_b)`, 선택적 stride-16 `(80,C16)` |
| i₂ cache | `build_i2_cache_mf20.py` | mf20 `(40,128)` memory/logits |
| in-lane target | `build_inlane_gt_cache.py` | lead/gate/vrel target |
| predicted vrel | `build_pred_vrel_cache.py` | a₂ 기반 closing cache |
| profile pedal label | `gen_profile_pedal_labels*.py`, `gen_ttc_pedal_labels.py` | f₂ supervision |

`generate_image_seg_masks.py`는 현재 저장소에 없는 외부 `src.data.gt_pipeline`을 가져오므로 pseudo-mask를 처음부터 재생성하는 과정은 이 repo만으로 완결되지 않는다. 기존 mask 사용인지, 원본 GroundingDINO/SAM2 pipeline 복원인지 먼저 결정해야 한다.

### 5.2 학습

| 모델 | 대표 script | 현재 동작의 핵심 |
|---|---|---|
| segmentation | `train_seg_pretrain.py` | 4-class CE, ImageNet init, random frame split |
| multi-frame a₂ | `train_a2_multiframe.py` | cached i₁만 사용, encoder frozen, fusion/proj/head 학습 |
| 21-frame a₂ | `train_a2_n20.sh`, `train_a2_n20_db*.sh` | mf20 설정 launcher |
| single-frame a₂ | `train_a2_single_objdist.py` | obj/dist-only ablation |
| f₂ side | `train_prof635_side.py` | frozen f₁, pedal L1, 선택적 gate BCE/vrel injection |
| no-gate SCC | `train_scc_nogate.py`, `train_scc_profiles_g2.py` | profile별 controller variants |
| temporal gate | `learned_temporal_gate.py`, `train_p3_ng_long.py` | explicit `(B,K,10)` history 입력 |
| 이전 IY | `train_iy.py` | historical controller family |

### 5.3 평가와 진단

- a₂: `eval_a2_multiframe.py`, `eval_a2_holdout.py`, `eval_a2_yaw_stratified.py`
- bin resolution: `eval_bin80_quick.py`
- f₂/ACC: `eval_prof635_side_acc.py`, `test_prof635_closedloop.py`, `test_acc_closedloop.py`
- cache/live 정합: `verify_i2_live_vs_cache.py`
- gate 안정성: `exp_gate_stability.py`
- qualitative visualization: `viz_seg_inference.py`, `viz_a2_inference.py`, `viz_prof635_gradcam.py` 등

기존 evaluator가 계산하는 object F1, distance/vrel error, flip rate는 유용하지만, run-disjoint validation과 calibration, closed-loop safety까지 한 실험 ID로 연결하는 공통 evaluator는 아직 없다.

### 5.4 Live 실행

`closed_loop_node.py`가 핵심 ROS node다.

- camera: `/raw_img_app/perc/camera_1`
- state: `/app/loc/vehicle_state`
- 출력: steer, accel pedal, brake pedal, optional acceleration command
- 최대 inference rate: 20 Hz
- mf20 token history는 ROS timestamp가 아니라 `time.time()` 기반 wall clock deque다.
- 원하는 age마다 가장 가까운 token을 선택한다.
- startup 시 history가 부족하면 현재/직전 token을 반복 사용한다.
- 선택된 token의 목표 시각 오차에 상한이나 validity mask가 없다.

현재 `prof635` 분기에는 current backbone forward 외에 old previous-image backbone forward와 yaw-shift 계산도 남아 있다. mf20 경로는 이 `fp_rot`를 사용하지 않으므로 계산 낭비이며, latency 측정 전에 제거하거나 별도 baseline으로 계측해야 한다.

Launcher는 `run_closed_loop_prof635.sh`, `run_closed_loop_acc.sh`, `run_closed_loop_sim.sh` 등에 흩어져 있다. 환경 변수에 따라 model/checkpoint/gate/profile이 달라지므로, 결과에는 반드시 전체 환경 변수를 함께 저장해야 한다.

## 6. C++ expert synthesis와 MPC

### `cpp/scc_replanner`

- 차량을 path에 projection한다.
- ego lane/forward 범위에서 lead를 선택한다.
- desired headway와 minimum gap을 이용해 speed profile을 재계획한다.
- acceleration/deceleration limit을 적용한다.

### `cpp/scc_batch_synth`

- raw ego/object 기록을 읽는다.
- future trajectory와 predicted object sequence를 만든다.
- SCC replanner와 횡/종방향 controller를 실행해 expert target을 기록한다.
- random trajectory K개 생성과 shard 실행을 지원한다.

### `mpc/`

A1 Rule 계열 controller의 fork다. longitudinal MPC에서 acceleration state constraint를 약 `±5 m/s²`, jerk를 약 `±10`으로 제한한 수정이 포함된다. 이전 합성 데이터에서 비정상적으로 큰 acceleration이 발생한 이력 때문에 label 생성 버전과 MPC binary hash를 반드시 기록해야 한다.

현재 CMake/build 파일에는 원 개발 PC의 절대 경로와 외부 library 경로가 남아 있다. 새 환경에서 바로 재빌드된다고 가정하면 안 된다.

## 7. MORAI bridge와 다차량 실행

### `moraibridge/`

MORAI UDP packet을 ROS1 message/topic으로 바꾸고, controller command를 simulator로 돌려보낸다. config/launch, camera/vehicle/control UDP driver와 변환 code가 있다. 여기에 포함된 message package 대부분은 learning model이 아니라 통신 인터페이스다.

### `external/autohyu_msgs/`

차량 상태/제어 등 ROS message의 generated Python package다. 학습 로직은 없지만 live node import를 위해 필요하다.

### `multicar/`

차량별 namespace, bridge port, simulation ini와 controller 환경 변수를 생성한다. prefix에 따라 p1~p5 profile, no-gate, temporal gate, database a₂ 등 서로 다른 checkpoint를 선택할 수 있다. 멀티카 결과 비교 시 prefix가 곧 모델 정의라는 가정을 버리고 실제 resolved config를 저장해야 한다.

## 8. Active, experimental, legacy를 구분하는 기준

### Production baseline으로 볼 경로

- `closed_loop_node.py`의 기본 `MODEL_TYPE=prof635`
- seg-pretrained EfficientNetV2-S backbone
- `A2ModelMultiFrame`, 21 frames, 40 bins, `d_out=128`
- frozen Professor main + `Prof635SideModel`
- 해당 launcher가 선택한 gate/profile checkpoint

### 실험 경로

- stride-16/80-bin `A2ModelMultiFrameHR`
- 13-channel uncertainty head
- 5-frame ACC vrel injection
- single-frame obj/dist
- learned temporal gate family
- no-gate/profile별 side model

### Legacy 경로

- `legacy/` 전체
- active tree의 `IyModel`, raw two-image 계열은 production baseline이 아니라 비교 대상
- 초기 README/config에만 나타나는 구조

## 9. 현재 확인된 위험과 기술 부채

| 우선도 | 항목 | 영향 |
|---:|---|---|
| P0 | seg/a₂/f₂가 frame-random validation split 사용 | 동일 run 인접 frame 누수, 과대평가 가능 |
| P0 | integrated `A1E2EModel`은 2-frame, live는 21-frame | 제출 모델과 배포 거동 불일치 가능 |
| P0 | live history에 validity/warm-up/timestamp error guard 없음 | startup/drop/jitter 시 temporal input 의미 붕괴 |
| P0 | code tree에 Git metadata가 없음 | 정확한 재현 버전과 변경 이력 고정 어려움 |
| P1 | live mf20에서 사용하지 않는 previous backbone forward/yaw shift | GPU latency와 전력 낭비 |
| P1 | pseudo-mask 생성의 외부 `gt_pipeline` 누락 | segmentation dataset 재생성 불가 |
| P1 | absolute path가 많은 launcher/script | 다른 PC에서 재현 실패 가능 |
| P1 | checkpoint 이름/주석/모델 세대 혼재 | 잘못된 cache-checkpoint 결합 위험 |
| P1 | `CURRENT_BEST.json` 부재 | registry 의존 script 실패 가능 |
| P2 | `src/model/models.py` stale import | environment sanity script 실패 |
| P2 | 문서의 gate 범위, decoder 종류, live 최적화 설명이 코드와 다름 | 실험 해석 오류 |

## 10. 체크포인트와 캐시를 묶는 최소 manifest

실험 하나를 재현하려면 적어도 다음을 한 JSON/YAML에 묶어야 한다.

```yaml
experiment_id: a2_mf20_v2s_runholdout_seed0
source_snapshot: <git commit 또는 archive sha256>
dataset_manifest_sha256: <hash>
split_manifest_sha256: <hash>
seg_backbone:
  file: seg_pretrain_fx640_v3_backbone.pt
  sha256: <hash>
i1_cache:
  file: <path>
  sha256: <hash>
  roi: [235, 485, 0, 1280]
  normalization: imagenet
  backbone: tf_efficientnetv2_s
a2:
  architecture: A2ModelMultiFrame
  bins: 40
  prev_offsets_tick: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
  tick_seconds: 0.01
  yaw_shift: off
  ego_motion: zero
  d_token: 256
  d_out: 128
f2:
  checkpoint: <path>
  delta_mode: <raw|signed|brake_only>
  gate: <internal|none|temporal_mlp>
runtime:
  target_hz: 20
  device: <gpu>
  precision: fp32
seed: 0
```

이 manifest가 없으면 파일명만 보고 모델 세대를 복원해야 하는데, 현재 저장소에서는 그것만으로 충분하지 않다.

## 11. 읽는 순서

처음 보는 사람이 가장 빠르게 이해하려면 아래 순서가 좋다.

1. `scripts/closed_loop_node.py`: 실제 입력/출력과 선택되는 모델 확인
2. `src/model/image_models/a2_model.py`: i₁, multi-frame fusion, i₂/a₂ 계약 확인
3. `src/model/image_models/prof635_side_model.py`: f₁/f₂/gate와 최종 pedal 식 확인
4. `scripts/compute_i1_tok_cache.py`와 `scripts/build_i2_cache_mf20.py`: offline cache 계약 확인
5. `scripts/prepare_a2_aux_targets.py`: 7채널 GT 의미 확인
6. `scripts/train_a2_multiframe.py`와 `scripts/eval_a2_multiframe.py`: loss와 평가 확인
7. `scripts/train_prof635_side.py`: perception이 제어에 연결되는 방식 확인
8. `cpp/scc_batch_synth`와 `cpp/scc_replanner`: expert label의 출처 확인
9. `moraibridge/`, `multicar/`: simulator 배포 환경 확인
10. `legacy/`와 과거 문서는 마지막에 비교용으로 확인

## 12. 결론

이 코드의 핵심은 “segmentation 모델이 직접 제어한다”가 아니다. segmentation pseudo-task로 학습한 backbone을 40개의 lateral token extractor로 재사용하고, 학습형 21-frame temporal a₂가 object/kinematics와 i₂ memory를 만든 뒤, frozen trajectory controller의 pedal을 f₂가 영상 조건부로 수정하는 구조다.

다음 실험의 첫 목표는 새 모델을 더 만드는 것이 아니라, **run-disjoint split, cache/live parity, 21-frame 통합 모델 parity, timestamp validity**를 먼저 확립하는 것이다. 이 네 가지가 고정되지 않으면 이후의 mIoU, object F1, pedal MAE 개선을 신뢰하기 어렵다.

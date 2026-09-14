# 선배 SCC 코드 전체 구조와 파일 역할

## 1. 원본 위치와 사용 원칙

원본 저장소:

```text
/Users/sangbum/Workspace/A1_Challenge/scc_controller_imitator-main
```

실카메라 실험 프로젝트에는 원본 저장소에서 실제 runtime에 필요한 `a2_model.py`와 checkpoint 두 개만 `senior_code/`에 그대로 복사했다. `senior_code`는 symbolic link가 아닌 실제 디렉터리다.

```text
A1_RealCamera_A2_21FPS/senior_code
├── COPY_MANIFEST.sha256
├── src/model/image_models/a2_model.py
└── data/ckpts/          # seg backbone + mf20 a₂
```

따라서 다른 PC에는 `A1_RealCamera_A2_21FPS/`만 복사하면 된다. 실험 runner는 별도로 재작성한 model이 아니라 복사된 원본의 다음 클래스를 사용한다.

runner는 이 파일을 경로 기반으로 직접 load하고 `A2ModelMultiFrame` class를 사용한다.

새 프로젝트가 추가하는 것은 실카메라 capture와 측정 wrapper뿐이다.

## 2. 저장소 전체 지도

```text
scc_controller_imitator-main/
├── README.md, DESIGN.md, architecture.md
├── PROJECT_STRUCTURE_BRIEF_KR.md
├── requirements.txt, environment.yml
├── config/
├── src/
│   ├── datamodule/
│   ├── model/
│   │   ├── image_models/
│   │   └── utils/
│   └── utils/
├── scripts/
├── data/
│   ├── ckpts/
│   └── control/
├── cpp/
│   ├── scc_replanner/
│   └── scc_batch_synth/
├── mpc/
├── moraibridge/
├── multicar/
├── external/
├── legacy/
├── docs/
├── slurm/
├── slurm_script/
└── .github/
```

## 3. 최상위 폴더 설명

전달본에는 총 882개 파일이 있다. 그중 큰 비중은 `moraibridge/`와 `mpc/`의 ROS message/service, CMake, C++ 파일이며, perception Python core는 `src/`의 27개 파일과 `scripts/`의 학습·평가 도구에 집중되어 있다.

| 영역 | 파일 수 | 성격 |
|---|---:|---|
| root | 9 | 소개, 설계, dependency, MkDocs 설정 |
| `src/` | 27 | 현재 Python model/data/util source |
| `scripts/` | 176 | 전처리, 학습, 평가, 시각화, live launcher |
| `data/` | 38 | checkpoint와 control calibration |
| `cpp/` | 16 | SCC expert/batch synthesis C++ |
| `mpc/` | 92 | ROS/C++ MPC controller tree |
| `moraibridge/` | 295 | MORAI network bridge와 ROS interface |
| `external/` | 61 | generated ROS Python message |
| `legacy/` | 54 | 이전 세대 보존본 |
| `docs/` | 97 | 설계·실험 기록과 그림 |
| 기타 | 17 | config, multicar, Slurm, CI |

| 폴더 | 역할 | 실카메라→a₂ 실험에서 필요 여부 |
|---|---|---|
| `src/` | 실제 Python model/dataset/util | a₂ model 때문에 필요 |
| `scripts/` | 데이터 생성, 학습, 평가, live 실행 | 구조 참고 및 offline parity에 일부 필요 |
| `data/ckpts/` | seg/a₂/f₁/f₂ checkpoint | seg backbone과 mf20 a₂ 두 개 필요 |
| `data/control/` | raceline, pedal map | 불필요 |
| `cpp/` | SCC expert target 합성 | 불필요 |
| `mpc/` | lateral/longitudinal MPC와 C++ dependency | 불필요 |
| `moraibridge/` | MORAI UDP ↔ ROS bridge | 실물 OpenCV camera 실험에는 불필요 |
| `multicar/` | 여러 MORAI 차량 launcher/namespace | 불필요 |
| `external/` | generated ROS message | 불필요 |
| `legacy/` | 이전 세대 구현 | 불필요, 비교 참고만 가능 |
| `docs/` | 개발 기록과 실험 보고서 | 참고용, 시점별 불일치 주의 |
| `slurm*` | 서버 학습 launcher/environment | 재학습할 때만 필요 |
| `config/` | 초기/공통 설정 | 일부는 현재 live 코드보다 오래됨 |
| `.github/` | CI workflow | 실험 runtime에는 불필요 |

### Root 파일

| 파일 | 역할 |
|---|---|
| `README.md` | 프로젝트 소개와 기본 실행법 |
| `DESIGN.md`, `architecture.md` | 초기 시스템/모델 설계. 최신 mf20과 다른 부분이 있음 |
| `PROJECT_STRUCTURE_BRIEF_KR.md` | 기존 한국어 구조 요약 |
| `requirements.txt`, `environment.yml` | Python/Conda dependency |
| `mkdocs.yml`, `requirements-docs.txt` | 문서 사이트 생성 설정 |
| `.gitignore` | 생성물·로컬 데이터 제외 규칙 |

### 설정·운영 폴더

- `config/scc_config.ini`: SCC 실행 설정. 실카메라 a₂-only runner는 읽지 않는다.
- `slurm/`: A2 E2E/bin80 학습용 Slurm job과 환경 파일이다.
- `slurm_script/`: cache, segmentation, a₂, iᵧ/prof635 서버 작업용 runbook/launcher다.
- `.github/workflows/`: CI workflow다.

## 4. 현재 전체 시스템의 큰 흐름

원본 전체 시스템은 다음 다섯 층으로 나뉜다.

```text
1. MORAI/recorded data
       │
2. segmentation backbone → i₁ → multi-frame a₂/i₂
       │
3. Professor f₁ + image side f₂/gate
       │
4. steer/pedal ROS command
       │
5. MORAI bridge / multi-car execution
```

이번 실카메라 실험은 2번의 a₂ output에서 중단한다.

```text
physical camera → ROI → backbone → i₁ history → A2ModelMultiFrame → a₂ logits
```

## 5. `src/model/image_models/`: perception과 image model

### `a2_model.py` — 가장 중요한 원본 파일

한 파일 안에 i₁ encoder, temporal fusion, a₂ head와 여러 실험 variant가 함께 있다.

#### `LateralStripEncoder`

```text
input image: (B,3,250,1280)
  → timm EfficientNet features_only last stage
  → feature: roughly (B,256,8,40)
  → adaptive average pool to (1,40)
  → Conv1d projection
  → token: (B,40,256)
```

주의할 점은 cache/live mf20 경로가 projection 전 raw token `(B,40,C_b)`를 저장하고, `A2ModelMultiFrame.encoder.proj`가 이를 256차원으로 바꾼다는 것이다.

#### `_AttnBlock`

Multi-head attention, residual LayerNorm, FFN으로 구성된 공통 attention block이다.

#### `ImageFusionBlock`

구형 two-frame용이다. current token을 query로, ego token과 previous token을 key/value로 사용한다.

#### `_MultiFrameFusion`

현재 mf20 핵심이다.

- 각 frame에 공유 lateral self-attention
- current frame 40개 token이 query
- ego token과 과거 frame 전체 token이 key/value
- lateral position과 temporal embedding
- cross-attention 후 post self-attention

#### `A2ModelMultiFrame`

production perception model이다.

```text
21 × raw token (B,40,256)
  → shared encoder.proj
  → _MultiFrameFusion
  → fused (B,40,256)
  → fc_proj
  → i₂ (B,40,128)
  → AuxHead
  → a₂ logits (B,40,7)
```

`forward_tokens()`가 실카메라 runner에서 직접 호출되는 원본 method다.

#### `AuxHead`

기본 7채널:

| index | 의미 |
|---:|---|
| 0 | object logit |
| 1 | longitudinal absolute velocity |
| 2 | lateral absolute velocity |
| 3 | longitudinal relative velocity |
| 4 | lateral relative velocity |
| 5 | proximity |
| 6 | distance |

실험형 `predict_var=True`는 13채널 uncertainty output을 사용하므로 기본 checkpoint와 혼용하면 안 된다.

#### 기타 class

- `A2Model`: 구형 current+previous 2-frame
- `A2ModelSingleObjDist`: single-frame object/distance-only
- `A2ModelMultiFrameHR`: stride-16 feature를 합치는 80-bin 실험형

실카메라 21-frame baseline은 반드시 `A2ModelMultiFrame`이다.

### `seg_pretrain_model.py`

- `SegPretrainModel`
- timm backbone 5개 feature stage
- transpose-convolution과 skip concat을 쓰는 U-Net/FPN-lite decoder
- 4 class: background, road, vehicle, wall
- downstream에서는 decoder output을 매 frame 계산하지 않고 학습된 backbone weight만 사용

즉 실카메라 runtime은 segmentation mask를 먼저 만들고 a₂에 넣는 구조가 아니다. segmentation task로 pretrain된 **backbone feature**를 i₁으로 쓴다.

### `prof635_side_model.py`

Professor main과 f₂ pedal side controller다. 이번 실험 범위 밖이다.

### `iy_model.py`

이전 image-memory controller와 token yaw-shift helper가 같이 있다. 현재 production mf20 baseline은 yaw shift off이므로 실카메라 runner가 이 model을 사용하지 않는다.

### `image_scc_models_v2.py`

raw current/previous image를 controller에 직접 연결하던 이전 세대 실험형이다. mf20 baseline이 아니다.

## 6. `src/model/`의 나머지 파일

### `a1_e2e_model.py`

backbone, a₂, f₁, f₂, temporal gate를 하나의 `state_dict`와 differentiable graph로 묶으려는 통합 모델이다.

하지만 현재는 `A2ModelMultiFrame`이 아니라 구형 2-frame `A2Model`을 생성한다. production 21-frame과 동일한 실카메라 baseline으로 쓰면 안 된다.

### `models.py`

이전 model factory다. active tree에 없는 `.image_models.image_scc_models`를 import하는 stale 경로가 남아 있다. 실카메라 runner는 사용하지 않는다.

### `model/utils/common.py`

이전/공통 model 보조 함수다. production mf20 `forward_tokens`에는 직접 필요하지 않다.

## 7. `src/datamodule/`

| 파일 | 데이터 계약 |
|---|---|
| `seg_pretrain_dataset.py` | raw image ROI와 4-class pseudo segmentation mask |
| `a2_pair_dataset.py` | current/previous image와 `(40,7)` a₂ target |
| `prof635_dataset.py` | trajectory, ego, i₂ cache, pedal/gate label; 이번 범위 밖 |
| `scc_synth_pair_dataset.py` | C++가 합성한 SCC pair 학습 자료 |
| `scc_synth_iy_dataset.py` | 이전 IY controller 합성 자료 |
| `controller_imitator_aug_dataset.py` | controller imitation augmentation |

실카메라 FPS 측정은 dataset class를 사용하지 않는다. OpenCV frame을 곧바로 같은 ROI/normalization으로 변환한다.

## 8. `src/utils/`

| 파일 | 역할 | 이번 실험 |
|---|---|---|
| `ckpt_registry.py` | `CURRENT_BEST.json` checkpoint lookup | runner는 명시 경로를 사용하므로 불필요 |
| `raceline_dt0p1s.py` | 0.1초 간격 trajectory 40개 생성 | 불필요 |
| `raceline_search.py` | path/raceline search | 불필요 |
| `prof635_infer.py` | traj6/ego6와 controller inference 보조 | 불필요 |
| `pedal_map.py`, `pedal_label.py` | acceleration↔pedal label | 불필요 |
| `temporal_gate.py` | f₂용 temporal gate network | 불필요 |

## 9. `scripts/`: 목적별 파일 지도

`scripts/`에는 서로 다른 개발 세대의 shell/Python 파일이 같이 있다. 파일명이 최신 production을 뜻한다고 가정하지 말고 model class와 checkpoint를 확인해야 한다.

### 9.1 실시간 실행

| 파일 | 역할 |
|---|---|
| `closed_loop_node.py` | MORAI ROS camera/state → perception → f₁/f₂ → command 전체 실행 |
| `run_closed_loop*.sh` | model/profile별 environment를 설정하고 node 실행 |
| `test_prof635_closedloop.py`, `test_acc_closedloop.py` | controller closed-loop 검사 |

`closed_loop_node.py`의 perception 부분은 참고할 가치가 있지만 그대로 실카메라 21 FPS 시험에 쓰기 어렵다.

- ROS/MORAI message 필요
- f₁/f₂와 trajectory 계산 포함
- `INFER_RATE_HZ=20` cap
- mf20에서 쓰지 않는 previous backbone forward도 남아 있음

그래서 새 runner는 원본 a₂ class/checkpoint를 그대로 쓰되 입력과 timing harness만 분리한다.

### 9.2 Segmentation

| 파일 | 역할 |
|---|---|
| `generate_image_seg_masks.py` | GroundingDINO/SAM2 pseudo mask 생성; 외부 `src.data.gt_pipeline` 필요 |
| `train_seg_pretrain.py` | 4-class segmentation backbone 학습 |
| `inspect_seg_mask.py` | mask 검사 |
| `viz_seg_inference.py` | segmentation 정성 시각화 |
| `bench_seg_backbone_latency.py` | backbone latency 측정 |
| `run_seg_backbone_sweep.sh` | backbone 실험 launcher |

### 9.3 i₁/a₂ GT와 cache

| 파일 | 역할 |
|---|---|
| `prepare_a2_aux_targets.py` | object 3D/velocity를 40-bin 7채널 GT로 projection |
| `compute_i1_tok_cache.py` | seg backbone feature를 raw i₁ token cache로 저장 |
| `build_i1_tok_cache_b0.py` | B0 backbone용 cache |
| `build_i2_cache.py` | 구형 two-frame i₂ cache |
| `build_i2_cache_mf20.py` | production 21-frame a₂로 i₂/a₂ cache 생성 |
| `build_pair_manifest.py` | raw frame/data join manifest |
| `build_meta_from_framelist.py` | frame list에서 metadata 생성 |

### 9.4 a₂ 학습

| 파일 | 역할 |
|---|---|
| `train_a2_multiframe.py` | N-frame a₂ cache-only 학습 |
| `train_a2_n20.sh` | 21-frame mf20 launcher |
| `train_a2_n20_db*.sh` | database/속도 variant |
| `train_a2_single_objdist.py` | current-only object/distance |
| `train_a2_v5_from_cache.py` | 이전 cache a₂ 세대 |
| `train_a2_e2e.py` | image부터의 a₂ end-to-end 실험 |
| `train_a2_head.py` | head 중심 학습 |
| `launch_a2_*.sh`, `chain_a2_*.sh` | 실험 조합 launcher |

### 9.5 a₂ 평가

| 파일 | 역할 |
|---|---|
| `eval_a2_multiframe.py` | multi-frame object/distance/vrel/stability 평가 |
| `eval_a2_holdout.py` | holdout 평가 |
| `eval_a2_yaw_stratified.py` | yaw 구간별 평가 |
| `eval_a2_v8_vs_v7b_holdout.py` | uncertainty/new head 비교 |
| `eval_a2_single_vs_v7b.py` | single/two-frame 비교 |
| `eval_bin80_quick.py` | 40/80-bin 비교 |
| `diag_a2_obj_vs_flicker.py` | object score와 flicker 분석 |
| `verify_i2_live_vs_cache.py` | offline cache와 live-equivalent i₂ 정합 |

### 9.6 a₂ 시각화

- `viz_a2_inference.py`
- `viz_multiframe_a2_arch.py`
- `viz_upsample_exp_diagram.py`
- `viz_sam_close_leads.py`
- `viz_iy_fimem_norm.py`

### 9.7 f₂/controller 관련 — 이번 범위 밖

- `train_prof635_side.py`
- `eval_prof635_side_acc.py`
- `train_scc_nogate.py`
- `train_scc_profiles_g2.py`
- `learned_temporal_gate.py`
- `gen_profile_pedal_labels*.py`
- `gen_ttc_pedal_labels.py`
- `exp_gate_stability.py`

### 9.8 데이터/운영 도구

- `data_pipeline.py`: 데이터 처리 orchestration
- `dashboard_status.py`: 작업 상태 확인
- `dataset_dist_report.py`: dataset 분포
- `publish_watcher.sh`, `publish_test_weights.sh`: checkpoint 배포 보조
- `streaming_prebake_loop.sh`: 연속 prebake 작업
- `sanity_check_env.py`: 환경 확인. stale `models.py` import 문제 주의

## 10. `data/`

### `data/ckpts/`

다양한 세대의 `.pt/.pth`가 함께 있다. 실카메라 21-frame a₂에 직접 필요한 파일은 다음 두 개다.

```text
seg_pretrain_fx640_v3_backbone.pt
a2_mf20_v2s_n20.pt
```

현재 전달본 hash:

- seg backbone: `8b9262a5bdd214081b62223f1fee3bb96f50c55b3f3b5bcc3300a5ba30a8fd91`
- mf20 a₂: `c095b16f15cd73614f2f829bba774095e9c65626b895d4cf1558692953f44003`

나머지는 two-frame a₂, B0, database, side controller, temporal gate, profile별 checkpoint다.

### `data/control/`

raceline CSV와 pedal map이 있다. 실카메라→a₂에서는 사용하지 않는다.

## 11. `cpp/`

### `cpp/scc_replanner/`

- `scc_speed_replanner.cpp`
- path projection
- lead 선택
- headway/minimum gap 기반 speed replanning

### `cpp/scc_batch_synth/`

- `ego_parser.cpp`: ego 기록 parsing
- `object_parser.cpp`: object 기록 parsing
- `frame_loader.cpp`: frame 단위 입력 조립
- `synth_batch.cpp`: SCC/MPC expert target batch 생성
- `test/`: parser/replay tests

이 code는 a₂ runtime이 아니라 학습용 synthetic/controller label 생성에 관여한다.

## 12. `mpc/`

ROS/CMake 기반 A1 Rule controller fork다.

```text
mpc/src/app/control/
├── lateral_control/
├── longitudinal_control/
├── scc_replanner/
└── scc_batch_synth/
```

- lateral MPC: steering reference 추종
- longitudinal MPC: speed/acceleration control
- SCC replanner: lead-aware speed reference
- batch synth: expert label 생성

빌드 파일에 원 개발 장비의 library/path 가정이 남아 있어 새 환경 재빌드는 별도 작업이다. 실카메라 a₂ inference에는 필요 없다.

## 13. `moraibridge/`

MORAI simulator와 ROS1 사이의 network/interface layer다.

- `src/morai_bridge/`: camera/vehicle/control UDP driver와 launch
- `src/MORAI_CPP_Networks/`: UDP packet 구현
- `src/autohyu_msgs/`: 차량/계획/제어 ROS message
- `src/morai_msgs/`: MORAI가 제공하는 많은 message/service definition
- `src/ini_parser/`: simulation config 읽기
- `src/morai_udp_networkmodule/`: MORAI UDP network library
- `config/simulation.ini`: simulator network/port 설정

실물 UVC camera를 OpenCV로 읽는 실험에는 필요 없다.

## 14. `multicar/`

- `run_multicar.sh`: 여러 차량의 namespace, port, controller 실행
- `run_model_only.sh`: bridge와 분리해 model만 실행
- `scc_multicar.ini.example`: 차량별 설정 예시

차량 prefix에 따라 profile/checkpoint/environment가 달라지므로 MORAI 다차량용이다.

## 15. `external/`

`external/autohyu_msgs`는 generated ROS Python message다. live ROS node import에는 필요하지만 OpenCV camera runner에는 필요 없다.

## 16. `legacy/`

이전 SCC/image model, script, external fork가 보관되어 있다. active `src/`와 같은 파일명이 있을 수 있어 Python path에 동시에 넣으면 잘못된 module을 import할 수 있다.

실카메라 runner는 프로젝트 내부 `senior_code/` 하나만 `sys.path` 선두에 넣고 legacy를 import하지 않는다.

## 17. `docs/`와 root 문서

다음 문서는 서로 다른 개발 시점에 작성되었다.

- `README.md`
- `DESIGN.md`
- `architecture.md`
- `PROJECT_STRUCTURE_BRIEF_KR.md`
- `docs/ARCHITECTURE.md`
- `docs/TRAINING.md`
- `docs/MODEL_AUDIT.md`
- `docs/STATE_AND_LESSONS.md`
- `docs/EXPERIMENT_DIGEST.md`

초기 root 문서는 two-image prototype을 설명하는 경우가 많고, 현재 production mf20과 다르다. 최종 판단은 다음 순서로 한다.

1. `scripts/closed_loop_node.py`의 실제 분기
2. `a2_model.py`의 실제 class/forward
3. 사용 checkpoint state_dict
4. cache metadata
5. 문서와 주석

## 18. 실카메라 실험이 실제로 읽는 원본 파일

### 직접 import

```text
senior_code/src/model/image_models/a2_model.py
```

### 직접 load

```text
senior_code/data/ckpts/seg_pretrain_fx640_v3_backbone.pt
senior_code/data/ckpts/a2_mf20_v2s_n20.pt
```

### 새 프로젝트가 별도로 구현하는 부분

```text
scripts/live_camera_a2_21fps.py
```

- UVC/OpenCV camera capture
- latest-frame queue
- host timestamp history
- 21 FPS 판정
- latency/logging

모델 파일은 손대지 않는다.

## 19. 원본을 그대로 쓸 때의 제약

1. `A2ModelMultiFrame` 원본 constructor는 `pretrained=True`를 요청한다. 그러나 `forward_tokens()`에서는 그 내부 image backbone을 사용하지 않고 별도 seg backbone token을 받으므로, portable runner는 constructor 호출 중 다운로드만 억제한 뒤 bundled checkpoint를 load한다. 원본 source와 model architecture는 바꾸지 않는다.
2. checkpoint load 전에 architecture가 정확히 V2-S, 21-frame, `d_token=256`, `d_out=128`이어야 한다.
3. 원본 `closed_loop_node.py` 자체는 ROS/f₂/20 Hz cap 때문에 이번 실험 runner로 사용할 수 없다.
4. 실제 camera가 1280×720/FOV/pose를 재현하지 못하면 throughput은 측정할 수 있어도 accuracy 비교는 직접 성립하지 않는다.
5. 원본 repository에 Git metadata가 없어 runner 결과에는 source/checkpoint SHA-256을 기록한다.
6. `COPY_MANIFEST.sha256`과 `scripts/verify_portable_bundle.py`로 `a2_model.py`와 checkpoint 두 개의 무결성을 검사한다.

## 20. 결론

선배들의 파일을 그대로 쓰는 것은 어렵지 않다. 다만 “원본 전체 live node를 그대로 실행”하는 것과 “원본 perception model을 그대로 import”하는 것은 구분해야 한다.

- 원본 전체 live node: MORAI/ROS/f₂/20 Hz가 묶여 있어 이번 질문과 맞지 않음
- 원본 a₂ model/checkpoint: 수정 없이 그대로 사용 가능
- 새로 필요한 것: 실물 camera adapter와 측정 harness뿐

현재 프로젝트는 두 번째 방식을 사용한다.

# 실물 카메라 → 21-frame a₂, 21 FPS 실행 가능성 실험 설계

## 1. 정확한 목표

이 실험의 목표는 f₂나 차량 제어 성능을 보는 것이 아니다.

> 선배들의 시뮬레이터 입력용 perception 모델을 실제 카메라에 연결했을 때, `camera → preprocessing → backbone/i₁ → 21-frame a₂ → a₂ logits`가 실제로 지속 21 FPS로 동작하는지 확인한다.

여기에는 서로 다른 두 개의 `21`이 있다.

- **21 FPS**: 매초 a₂ output을 약 21번 생성한다. 한 frame의 시간 budget은 `1000/21 = 47.62 ms`다.
- **21-frame a₂**: 현재 token 1개와 과거 token 20개를 사용한다. 목표 history age는 `0.05, 0.10, …, 1.00 s`다.

두 조건을 모두 확인해야 한다. 21-frame tensor 한 번이 47.62 ms보다 빠르더라도 카메라 capture나 queue가 느리면 실물 시스템은 21 FPS가 아니다.

또한 upstream `closed_loop_node.py`에는 `INFER_RATE_HZ=20` cap이 있으므로 그 node를 그대로 사용하면 21 FPS를 증명할 수 없다. 본 프로젝트의 실카메라 runner는 f₁/f₂와 이 cap을 제거한 상태에서 a₂까지의 순수 output rate를 잰다.

모델 구현을 다시 작성한 것은 아니다. runner는 프로젝트 내부에 그대로 복사된 `senior_code/src/model/image_models/a2_model.py`의 `A2ModelMultiFrame`을 import하고, 함께 복사된 seg/a₂ checkpoint를 load한다. 새로 작성한 부분은 UVC camera adapter, timestamp history sampler와 계측 코드뿐이다. 실행 결과에는 모델 원본 파일과 두 checkpoint의 SHA-256이 기록된다.

## 2. 실험 범위

### 포함

- 실물 USB/UVC camera capture
- 실제 전달 해상도와 frame interval
- BGR→RGB
- 720×1280 입력과 `y=235:485` ROI
- ImageNet normalization
- segmentation-pretrained EfficientNetV2-S backbone
- i₁ raw token `(1,40,256)`
- timestamp 기반 1초 history
- `A2ModelMultiFrame`, total 21 frames
- a₂ output `(1,40,7)`
- latency, throughput, jitter, drop, history 품질
- 실카메라에서의 최소 정성/정량 domain check

### 제외

- f₁/f₂
- steer/pedal
- raceline/MPC/SCC replanner
- ROS/MORAI closed loop
- controller loss

## 3. 검증할 두 축

### 축 A — 시스템 실행 가능성

다음 질문은 GT가 없어도 답할 수 있다.

- camera가 실제 21 FPS를 공급하는가?
- backbone과 a₂가 p95 47.62 ms 이내인가?
- startup 이후 output이 21 FPS인가?
- 오래된 frame이 queue에 쌓이지 않는가?
- 20개 과거 slot이 실제 약 50 ms 간격인가?
- 5분 이상 실행해도 열/메모리/지연 악화가 없는가?

### 축 B — sim-to-real 유효성

다음 질문은 실물 촬영과 annotation이 필요하다.

- 실제 차량을 object로 검출하는가?
- no-vehicle 장면에서 false positive가 적은가?
- 거리 순서와 대략적인 scale이 맞는가?
- 접근/이탈할 때 `vL_rel` 부호가 맞는가?
- exposure, blur, 그림자, 실제 도로 texture에서도 출력이 안정적인가?

축 A가 통과해도 축 B가 실패할 수 있다. 결과 보고서에서 반드시 분리한다.

## 4. System Under Test

```text
Camera capture thread
  │ latest frame only + host timestamp
  ▼
Geometry guard
  │ expected 1280×720; resize는 throughput-only
  ▼
ROI / normalization
  │ (1,3,250,1280)
  ▼
EfficientNetV2-S last feature
  │ roughly (1,256,8,40)
  ▼
adaptive pool H→1
  │ i₁ raw (1,40,256)
  ▼
GPU token deque, timestamped
  │ current + nearest token at 0.05…1.00 s
  ▼
A2ModelMultiFrame.forward_tokens
  │ ego_motion=(0,0), yaw_shift=off
  ▼
a₂ logits (1,40,7)
  [obj, vL_abs, vY_abs, vL_rel, vY_rel, prox, dist]
```

현재 production cache와 정합하기 위해 yaw shift는 끄고 ego motion은 zero로 둔다.

## 5. 시간 정의

### Camera timestamp

OpenCV UVC 입력은 일반적으로 sensor exposure timestamp를 주지 않는다. runner의 `captured_at`은 host에서 `read()`가 끝난 시각이다.

따라서 이 실험에서 측정하는 `capture_to_a2_ms_host`는:

```text
host capture completion → a₂ output ready
```

이다. 진짜 photon/exposure-to-output latency는 camera SDK의 hardware timestamp나 LED/photodiode 방식이 필요하다.

### Steady state

21-frame a₂는 최대 1초 history가 필요하다. 시작 후 첫 1초는 같은 frame을 여러 slot에서 반복하므로 throughput 수치와 history 유효성을 steady-state와 분리한다.

- startup: history span `<1.0 s`
- steady: history span `≥1.0 s`

최종 output FPS와 latency 판정은 steady 구간을 사용한다.

## 6. 사전 조건

### Hardware 기록

- camera 제조사/모델
- connection: USB 2/3, capture card 포함
- requested/negotiated resolution과 FPS
- host CPU/GPU/RAM
- OS, camera backend
- 전원 모드와 GPU clock/power limit

### Software 기록

- Python/PyTorch/timm/OpenCV 버전
- seg checkpoint SHA-256
- a₂ checkpoint SHA-256
- source snapshot SHA-256 또는 Git commit
- precision: fp32/AMP
- display on/off

### Geometry 확인

정확도 실험 전에 아래를 확인한다.

- 실제 camera output이 1280×720인가?
- horizon과 차량 장착 높이/각도가 sim camera와 유사한가?
- `y=235:485`가 실제로 도로와 앞차를 포함하는가?
- camera horizontal FOV/focal length가 학습 camera와 얼마나 다른가?

해상도만 맞아도 camera intrinsics와 장착 자세가 다르면 a₂ lateral bin과 거리 scale이 어긋날 수 있다.

## 7. 실험 단계

### Phase 0 — Checkpoint/shape smoke test

실물 카메라 전에 확인한다.

1. seg backbone checkpoint가 EfficientNetV2-S state의 95% 이상을 load한다.
2. a₂ checkpoint에 `encoder.proj`, `fusion`, `fc_proj`, `head` key가 모두 있다.
3. 한 frame의 i₁이 `(1,40,256)`이다.
4. 21개 token 입력의 a₂ 출력이 `(1,40,7)`이다.
5. NaN/Inf가 없다.

checkpoint를 `strict=False`로 조용히 일부만 load한 상태에서 FPS를 재지 않는다.

### Phase 1 — Camera acquisition 확인

모델 실행 전 또는 runner log로 다음을 확인한다.

- negotiated resolution
- driver-reported FPS
- host-observed FPS
- frame interval p50/p95/p99
- read failure

요청을 21 FPS로 했는데 camera가 15 FPS만 제공하면 모델을 최적화해도 목표를 달성할 수 없다.

### Phase 2 — 60초 full-pipeline smoke

조건:

- display off
- fp32
- warm-up 10회
- 60초 측정
- camera 요청 1280×720 @ 21 FPS

수집:

- capture FPS
- a₂ output FPS
- preprocessing/backbone/a₂ latency
- capture-to-a₂ latency
- skipped frame
- history age error/duplicate
- NaN/Inf

### Phase 3 — 병목 분리

60초 결과가 실패하면 다음 순서로 판단한다.

1. `capture_fps <20.58`: camera/driver/USB mode 문제
2. capture는 통과하지만 backbone p95가 큼: image encoder 병목
3. backbone은 빠르지만 a₂ p95가 큼: temporal fusion 병목
4. compute는 47.62 ms 이하지만 output FPS가 낮음: capture thread, copy, display, synchronization 또는 I/O 문제
5. FPS는 통과하지만 history age error/duplicate가 큼: camera cadence와 sampler 문제

비교군:

| ID | precision | display | 목적 |
|---|---|---|---|
| R0 | fp32 | off | 정확한 baseline |
| R1 | AMP fp16 | off | CUDA 가속 효과 |
| R2 | best precision | on | 시각화 overhead |
| R3 | best precision | off | 5분 thermal test |

### Phase 4 — 5분 지속 실행

60초를 통과한 설정으로 최소 300초 실행한다.

- 1분 window별 FPS와 latency
- 첫 1분 대비 마지막 1분 latency 변화
- GPU memory 증가
- camera read failure
- process memory 증가
- thermal throttling 여부

### Phase 5 — 실카메라 출력 sanity check

고정된 안전한 장소에서 다음 장면을 촬영한다.

| Scene | 목적 |
|---|---|
| 빈 도로/주차장 | no-object false positive |
| 정면 차량 5/10/20/30/40 m | object probability와 distance order |
| 좌/중앙/우 차량 | lateral bin 방향성 |
| 차량이 천천히 접근 | `vL_rel <0` 기대 |
| 차량이 멀어짐 | `vL_rel >0` 기대 |
| 사람/벽/표지판만 존재 | vehicle confusion |
| 역광/그림자/야간 | domain robustness |
| 움직임 blur | temporal 안정성 |

첫 sanity check는 정성 평가지만, `frames.jsonl`과 영상을 frame ID로 맞춰 failure를 저장한다.

#### 실시간 A₂ 화면 규약

`--display`는 모델이 별도 이미지나 bounding box를 출력하는 것처럼 꾸미지 않는다. A₂가 보존한
공간축은 가로 40-bin뿐이므로, 실시간 RGB 화면의 모델 ROI에 각 bin을 세로 strip으로 대응시킨다.

| 화면 요소 | 원본 값 | 표시 변환 |
|---|---|---|
| 빨간 반투명 strip | a₂ channel 0 | `sigmoid` 확률을 농도로 사용 |
| `d=...m` | a₂ channel 6 | `×40`, object-positive bin에서만 표시 |
| 접근/이탈 및 `vrel` | a₂ channel 3 | `×20 m/s`; 음수=접근, 양수=이탈 |
| proximity bar | a₂ channel 5 | 화면에서만 0~1 clamp |
| 노란 중앙 영역 | 고정 bin 16~25 | 모델 출력/차선 검출이 아닌 실험 관심영역 |

하나의 차량이 여러 인접 bin에서 검출되면 이를 연속 group으로 묶고, 그 group 안의 distance와
vrel을 object probability 가중 평균해 한 번만 표시한다. raw `(40,7)` 텐서는 수정하지 않는다.
`compute_ms`는 시각화 시간을 제외하고 `stage_ms.visualization`에 따로 기록한다. 단, 화면 또는
영상 저장을 켠 run의 실제 output FPS에는 렌더링/인코딩 대기가 포함되므로 throughput baseline과
시각화 run은 분리한다.

### Phase 6 — 작은 실물 정량 set

실행 가능성이 확인된 뒤에만 만든다.

- 최소 500–1,000 labeled frame
- video/run 단위 train/validation 분리
- no-car와 distance bucket 균형
- 거리 GT: tape/rangefinder/LiDAR 등 측정 근거 기록
- relative speed GT: 고정 속도 주행 또는 timestamped position

지표:

- in-lane lead precision/recall/F1
- no-lead false positive per second
- distance MAE/bias by bucket
- relative velocity sign accuracy와 MAE
- temporal flip rate
- detection onset/release delay

## 8. 통과 기준

### A. 21 FPS 실행 가능성

60초 smoke와 300초 thermal test에서 모두 만족해야 한다.

| 항목 | 기준 |
|---|---:|
| observed camera FPS | `≥20.58` |
| steady a₂ output FPS | `≥20.58` |
| preprocessing+backbone+a₂ p95 | `≤47.62 ms` |
| NaN/Inf frame | `0` |
| unexpected resize | `false` |
| process crash/camera failure | `0` |

`20.58`은 목표 21 FPS의 98%다. 대회/제출 규칙이 정확히 21.0 이상을 요구하면 tolerance를 제거한다.

### B. Temporal input 품질

권장 초기 기준:

| 항목 | 기준 |
|---|---:|
| steady history age error p95 | `≤25 ms` |
| steady duplicate slot ratio | `≤5%` |
| history maximum span | `≥1.0 s` |

실제 camera cadence가 21 FPS이면 frame 간격은 약 47.62 ms라 50 ms target과 잘 맞지만, driver jitter를 반드시 실측한다.

### C. Sim-to-real sanity

성능 threshold는 labeled real-camera baseline을 보고 확정한다. 그 전에는 다음 guard만 쓴다.

- no-car 장면에서 object gate가 계속 켜지지 않음
- 차량 거리가 멀어질수록 예측 distance가 대체로 증가
- 접근/이탈에서 relative velocity 부호가 상식과 일치
- 출력이 한두 frame마다 무작위로 flip하지 않음

이 guard를 통과하지 못하면 FPS 최적화보다 camera geometry/domain adaptation이 먼저다.

## 9. 결과 판정표

| Camera | Compute | Output FPS | Real output | 판정 |
|---|---|---|---|---|
| fail | pass | fail | 미평가 | camera acquisition 병목 |
| pass | fail | fail | 미평가 | model compute 병목 |
| pass | pass | fail | 미평가 | pipeline/queue/I/O 병목 |
| pass | pass | pass | fail | 속도 가능, sim-to-real 실패 |
| pass | pass | pass | pass | 실카메라 a₂ 적용 가능 후보 |

## 10. 결과 파일

```text
outputs/<run_id>/
  frames.jsonl
  summary.json
  a2_overlay.mp4          # --save-overlay-video를 켰을 때
  hardware.json            # 수동 또는 다음 단계 자동화
  checkpoint_hashes.txt    # 다음 단계 자동화
  sample_video.mp4         # 정확도 audit를 할 때만
  failure_frames/
```

`frames.jsonl`에는 각 frame의 다음 값이 들어간다.

- capture/complete timestamp
- capture-to-a₂, preprocessing, backbone, a₂ latency
- skipped frame count
- 20개 temporal slot의 source sequence와 age error
- history unique source 수
- finite 여부
- 진단용 in-lane object/distance/vrel 요약
- 선택 시 raw `(40,7)`과 40-bin probability/distance/vrel/proximity 전체

## 11. 실행 명령

### fp32 baseline

```bash
python scripts/live_camera_a2_21fps.py \
  --camera 0 \
  --camera-fps 21 \
  --target-fps 21 \
  --width 1280 \
  --height 720 \
  --seg-ckpt /path/to/seg_backbone.pt \
  --a2-ckpt /path/to/a2_mf20.pt \
  --duration 60 \
  --output-dir outputs/fp32_60s
```

### CUDA AMP candidate

```bash
python scripts/live_camera_a2_21fps.py \
  --camera 0 \
  --camera-fps 21 \
  --target-fps 21 \
  --seg-ckpt /path/to/seg_backbone.pt \
  --a2-ckpt /path/to/a2_mf20.pt \
  --duration 60 \
  --amp \
  --output-dir outputs/amp_60s
```

### 5분 confirm

```bash
python scripts/live_camera_a2_21fps.py \
  --camera 0 \
  --camera-fps 21 \
  --target-fps 21 \
  --seg-ckpt /path/to/seg_backbone.pt \
  --a2-ckpt /path/to/a2_mf20.pt \
  --duration 300 \
  --amp \
  --output-dir outputs/amp_300s
```

### 실시간 화면과 A₂ overlay

```bash
python scripts/live_camera_a2_21fps.py \
  --camera 0 \
  --camera-fps 21 \
  --target-fps 21 \
  --seg-ckpt senior_code/data/ckpts/seg_pretrain_fx640_v3_backbone.pt \
  --a2-ckpt senior_code/data/ckpts/a2_mf20_v2s_n20.pt \
  --duration 60 \
  --display \
  --log-full-a2 \
  --save-overlay-video \
  --output-dir outputs/a2_visual_60s
```

## 12. 중요한 해석 주의사항

1. `--allow-resize`를 켠 결과는 throughput-only다.
2. `--display`를 켠 결과를 최종 FPS로 쓰지 않는다.
3. OpenCV timestamp로 sensor exposure latency를 주장하지 않는다.
4. startup 첫 1초의 duplicated history를 steady accuracy와 섞지 않는다.
5. checkpoint가 일부만 load된 상태로 속도만 측정하지 않는다.
6. `a₂ output FPS=21`과 `a₂ accuracy가 충분함`을 같은 결론으로 쓰지 않는다.
7. 실물 camera intrinsics/pose가 sim과 다르면 추가 calibration 또는 real-data fine-tuning이 필요하다.

## 13. 현재 상태

- 실카메라 전용 runner 구현 완료
- f₂/controller code를 experiment core에서 제외 완료
- 21-frame timestamp sampler와 startup/steady 분리 구현 완료
- stage latency/throughput/history 품질 logging 구현 완료
- 실시간 RGB+A₂ 40-bin overlay와 선택적 MP4/full-output logging 구현 완료
- Python 정적 문법 검증 가능
- 현재 workspace에는 PyTorch와 연결된 실물 camera/checkpoint가 준비되지 않아 실제 21 FPS 결과는 아직 없음

따라서 다음 실제 작업은 새 모델 학습이 아니라, 목표 hardware에서 fp32 60초 baseline을 실행해 `summary.json`을 얻는 것이다.

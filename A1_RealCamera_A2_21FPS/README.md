# A1 Real-Camera a₂ 21 FPS

실물 카메라 입력으로 아래 perception 경로가 **지속적으로 21 FPS를 낼 수 있는지** 검증하기 위한 독립 프로젝트다.

```text
physical camera
  → 720×1280 BGR/RGB 확인
  → ROI y=235:485
  → ImageNet normalization
  → segmentation-pretrained EfficientNetV2-S
  → i₁ raw lateral tokens (40,256)
  → timestamp history: current + 20 past tokens, 0.05~1.00 s
  → A2ModelMultiFrame
  → a₂ logits (40,7)
```

범위는 `a₂` 출력까지다. `f₁`, `f₂`, pedal, steering, raceline, ROS/MORAI closed-loop는 실험 대상이 아니다.

## 이 프로젝트가 답할 질문

1. 실물 카메라가 실제로 21 FPS 이상을 전달하는가?
2. `ROI preprocessing + backbone + i₁ + 21-frame a₂`의 p95 계산 시간이 47.62 ms보다 짧은가?
3. 카메라 capture부터 a₂ 출력까지 sustained output rate가 21 FPS인가?
4. 실카메라 timestamp로 구성한 20개 과거 slot이 의도한 50 ms 간격을 유지하는가?
5. 시뮬레이터로 학습한 a₂ 출력이 실카메라 영상에서도 의미가 있는가?

1~4는 실행 가능성, 5는 sim-to-real 정확도 문제다. 21 FPS를 달성했다고 perception이 정확하다고 결론 내리지 않는다.

선배들의 `closed_loop_node.py`는 `INFER_RATE_HZ=20`으로 제한되어 있어 그대로 실행하면 21 FPS 가능성을 검증할 수 없다. 새 runner는 이 제어용 20 Hz cap을 사용하지 않고 실제 camera-through-a₂ output rate를 측정한다.

## 실행 코드

[scripts/live_camera_a2_21fps.py](scripts/live_camera_a2_21fps.py)가 실카메라 전용 runner다.

특징:

- OpenCV UVC camera 입력
- capture thread는 최신 frame 하나만 유지해 오래된 frame queue 누적 방지
- camera가 제공한 실제 해상도/FPS 기록
- 21-frame history를 host timestamp 기준으로 구성
- startup 동안 반복된 history와 steady-state를 분리
- preprocessing/backbone/a₂별 p50/p95/p99 latency
- camera frame skip, history slot age error, duplicate slot 비율 기록
- a₂ in-lane object probability, distance, relative velocity를 진단용으로 기록
- 모델 ROI 위에 40개 lateral bin과 object probability를 빨간 반투명 strip으로 표시
- object-positive 연속 bin 묶음별 distance와 `vL_rel`, 40-bin proximity bar 표시
- 고정 중앙 관심영역 bin 16~25 표시(차선 검출 결과와 구분)
- 모델 계산시간과 시각화 시간을 분리해서 측정
- 선택적으로 전체 `(40,7)` 값 logging과 overlay MP4 저장
- `frames.jsonl`과 `summary.json` 생성
- `f₂`나 제어 출력은 전혀 계산하지 않음

## 선배 코드 원본을 사용하는 방식

이 프로젝트 안의 `senior_code/`는 symbolic link가 아니라 실제 디렉터리다. 실험에 필요한 선배 원본 `a2_model.py`와 checkpoint 두 개만 내용 변경 없이 복사해 넣었다. 따라서 `A1_RealCamera_A2_21FPS/` 폴더 하나만 Ubuntu PC로 옮기면 원래 저장소가 없어도 된다.

```text
senior_code/src/model/image_models/a2_model.py
  └─ A2ModelMultiFrame

senior_code/data/ckpts/
  ├─ seg_pretrain_fx640_v3_backbone.pt
  └─ a2_mf20_v2s_n20.pt
```

runner가 새로 담당하는 것은 OpenCV 실카메라 입력, timestamp history 구성, latency/FPS 기록뿐이다. 선배 source와 weight가 운반 중 바뀌거나 빠지지 않았는지는 PyTorch 설치 전에도 다음 명령으로 검사할 수 있다.

```bash
python3 scripts/verify_portable_bundle.py
```

정상이라면 `verified 3 bundled senior files`가 출력된다. 수정 복사본, f₂/controller 참고본과 학습용 파일은 실험 폴더에서 제거했다.

선배 원본 `A2ModelMultiFrame` constructor는 timm ImageNet weight 다운로드를 요청하지만, 이번 token 경로에서는 그 내부 backbone을 사용하지 않는다. 새 PC의 인터넷/cache에 의존하지 않도록 runner가 constructor 실행 중 다운로드만 막고, 즉시 bundled checkpoint를 load한다. 원본 `a2_model.py` 자체는 수정하지 않았다.

## 환경

```bash
conda env create -f environment.yml
conda activate a1-real-camera-a2
```

또는 실행 장치에 맞는 PyTorch를 먼저 설치한 뒤:

```bash
pip install -r requirements.txt
```

현재 이 workspace의 Python에는 PyTorch가 없으므로 카메라/model runtime은 아직 실행하지 못했다. Ubuntu NVIDIA GPU에서는 먼저 해당 GPU/driver에 맞는 PyTorch build가 설치됐는지 확인해야 한다.

## 필요한 checkpoint

아래 두 checkpoint가 이미 프로젝트 내부 `senior_code/data/ckpts/`에 포함되어 있다.

- segmentation backbone: 예 `seg_pretrain_fx640_v3_backbone.pt`
- 21-frame a₂: 예 `a2_mf20_v2s_n20.pt`

checkpoint 이름만 신뢰하지 말고 원본 hash와 architecture를 확인해야 한다. runner는 필요한 state_dict key가 빠지면 기본적으로 즉시 중단한다.

## Ubuntu PC로 옮기기

`A1_RealCamera_A2_21FPS/` 디렉터리 전체를 복사한다. `senior_code/`와 그 안의 약 160 MB checkpoint가 빠지면 안 된다. 복사한 PC에서:

```bash
cd A1_RealCamera_A2_21FPS
python3 scripts/verify_portable_bundle.py
conda env create -f environment.yml
conda activate a1-real-camera-a2
```

현재 runner는 OpenCV UVC 기반의 inference/timing prototype이다. 최종 `camera_aravis2 + ROS2` 입력 adapter는 아직 구현 전이므로, Ubuntu 실험 명령은 해당 adapter를 구현한 뒤 확정한다.

## 60초 1차 실험

정확한 throughput 측정에서는 `--display`를 켜지 않는다.

```bash
python scripts/live_camera_a2_21fps.py \
  --camera 0 \
  --camera-fps 21 \
  --target-fps 21 \
  --width 1280 \
  --height 720 \
  --seg-ckpt senior_code/data/ckpts/seg_pretrain_fx640_v3_backbone.pt \
  --a2-ckpt senior_code/data/ckpts/a2_mf20_v2s_n20.pt \
  --duration 60 \
  --output-dir outputs/run01_fp32
```

CUDA fp16 경로를 별도 비교할 때만 `--amp`를 추가한다.

```bash
python scripts/live_camera_a2_21fps.py \
  --camera 0 \
  --seg-ckpt senior_code/data/ckpts/seg_pretrain_fx640_v3_backbone.pt \
  --a2-ckpt senior_code/data/ckpts/a2_mf20_v2s_n20.pt \
  --duration 60 \
  --amp \
  --output-dir outputs/run02_amp
```

## 출력 해석

`summary.json`의 핵심 필드:

- `capture_fps_host_observed`: OpenCV가 실제 전달한 FPS
- `a2_output_fps_steady`: history 1초가 채워진 뒤의 a₂ output FPS
- `compute_ms.p95`: preprocessing+backbone+a₂ p95
- `capture_to_a2_ms_host.p95`: host가 frame을 받은 시점부터 a₂ 완료까지
- `capture_to_output_ms_host.p95`: a₂ decode와 중앙 lead 요약까지 완료된 시간
- `stage_ms.backbone`, `stage_ms.a2`: 병목 분리
- `stage_ms.a2_decode_and_summary`, `stage_ms.visualization`: 후처리와 화면 비용
- `history_age_error_ms.p95`: 원하는 50 ms 간격 대비 timestamp 오차
- `history_duplicate_slot_ratio`: 같은 frame이 여러 temporal slot에 반복된 비율
- `pass.sustained_a2_21fps`: 최종 시스템 실행 가능성

21 FPS의 frame budget은 `1000/21 = 47.62 ms`다.

## 정확도 확인용 화면

```bash
python scripts/live_camera_a2_21fps.py \
  --camera 0 \
  --seg-ckpt senior_code/data/ckpts/seg_pretrain_fx640_v3_backbone.pt \
  --a2-ckpt senior_code/data/ckpts/a2_mf20_v2s_n20.pt \
  --duration 60 \
  --display \
  --log-full-a2 \
  --output-dir outputs/run03_visual
```

화면 해석:

- ROI의 빨간 반투명 세로 strip: `sigmoid(a₂[:,0])` object probability
- `d=...m`: object-positive 연속 bin의 `a₂[:,6] × 40`
- `vrel=...m/s`: 같은 bin들의 `a₂[:,3] × 20`
- 빨간 왼쪽 화살표: 접근(`vL_rel < 0`), 파란 오른쪽 화살표: 이탈(`vL_rel > 0`)
- 아래 노란 막대: `a₂[:,5]` proximity를 화면 표시용으로만 0~1 clamp
- 노란 테두리 bin 16~25: 고정 중앙 관심영역이며 차선 인식 결과가 아님

한 차량이 여러 bin을 차지할 때 숫자를 반복하지 않도록 threshold를 넘는 인접 bin을 하나의
묶음으로 보고 probability 가중 평균 distance/vrel을 표시한다. 모델 원본 출력은 변경하지 않는다.

overlay 영상까지 저장하려면 `--save-overlay-video`를 추가한다. 결과 폴더에
`a2_overlay.mp4`가 생성된다.

`--display`, `--save-overlay-video`, `--log-full-a2`는 각각 렌더링·인코딩·JSON I/O 부하가
있으므로 최종 throughput 기준값은 모두 끈 별도 run으로 측정한다. `compute_ms`는 렌더링을
제외하지만 display를 켠 run의 실제 output FPS에는 이전 frame의 렌더링 대기시간이 반영된다.

## 카메라 해상도가 다를 때

정확도 실험은 반드시 학습 입력과 같은 1280×720 영상 및 `y=235:485` ROI geometry를 먼저 재현해야 한다.

`--allow-resize`는 throughput debugging에만 쓴다. resize하면 focal length, object size, horizon과 lateral bin 의미가 달라지므로 sim-to-real 정확도 결과는 무효다.

## 실험 문서

구체적인 단계, 통과 기준과 실물 촬영 시나리오는 [REAL_CAMERA_A2_21FPS_EXPERIMENT.md](docs/REAL_CAMERA_A2_21FPS_EXPERIMENT.md)에 정리했다.

원본 코드의 파일별 출처는 [UPSTREAM_MANIFEST.md](docs/UPSTREAM_MANIFEST.md)에 있다.

선배 저장소 전체의 폴더와 파일 역할은 [SENIOR_CODE_STRUCTURE_KR.md](docs/SENIOR_CODE_STRUCTURE_KR.md)에 정리했다.

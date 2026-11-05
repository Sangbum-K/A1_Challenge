# A1 Real-Camera a₂ 21 FPS

실물 카메라 입력으로 아래 perception 경로가 **지속적으로 21 FPS를 낼 수 있는지** 검증하기 위한 독립 프로젝트다.

```text
Hikrobot camera → camera_aravis2 → ROS2 sensor_msgs/Image (BayerRG8)
  → cv_bridge debayer
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

최신 실험 기록은 [2026-09-18 실외 카트 인식 시험](RESULTS_2026-09-18.md)이다.
실내 모니터 재촬영 단계의 결론은 [2026-09-14 실험 요약](RESULTS_2026-09-14.md)에 있다.

## 실행

Nuvo에서 새 터미널을 열면 프로젝트 디렉터리로 이동한다. `./run.sh`가 ROS2와
프로젝트 `.venv`를 자동으로 준비하고, `aim`·`live`·`bench`에서는 카메라 드라이버도
시작한다. 일반 실험에는 별도 `source`나 `ros2 launch`가 필요 없다.

```bash
cd /home/ailab/git/sangbum/A1_Challenge-main/A1_RealCamera_A2_21FPS
./run.sh check
./run.sh aim
```

`check`는 카메라 연결, GPU, 저장 공간과 커널을 확인한다. 이전 21 FPS 통과 기준은
`6.8.0-138-generic` 부팅이다. `6.8.1-1059-realtime`의 A1-BSW 설정에서는 약
15–16 FPS였으므로, `check`의 문제를 해결하고 `bench`로 다시 측정한다.
`aim`에서는 모델이 보는 초록 영역에 차량을 맞추고 초점·노출을 조절한다.
`[`/`]`로 노출을 바꾸고 `q`로 종료한 뒤, 화면에 표시된 노출값을 다음 명령에 넣는다.

```bash
./run.sh bench --secs 60 --exposure 2000
./run.sh live --exposure 2000 --obj-thr 0.5 --bag --out outputs/h120_d20_static
```

위 `2000` µs와 `h120_d20_static`은 예시다. 실제 노출은 `aim`에서 정하고,
결과 폴더 이름에는 장착 높이와 실제 거리 조건을 적는다. `bench`는 화면·영상·전체
A₂ 로그를 끈 **21 FPS 처리량 측정**이다. `live`는 화면, overlay MP4, 매 프레임
`(40,7)` A₂ JSON을 저장한다. `--bag`은 원본 BayerRG8 영상도 녹화하며 약
1.2 GB/분을 사용한다. 정지 카메라 시험에서는 기본 ego 입력 `[0,0]`을 유지한다.

| 동작 | 명령/키 |
|---|---|
| 모델 없이 조준·노출 확인 | `./run.sh aim` |
| 실시간 A₂ 화면 | `./run.sh live --exposure 2000` |
| 20초만 기록 | `./run.sh live --exposure 2000 --secs 20 --out outputs/test_20s` |
| 녹화 입력 재생, 표시 threshold 비교 | `./run.sh replay outputs/h120_d20_static/bag --obj-thr 0.7` |
| 카메라 기능 이름 확인 | `./run.sh features` (카메라가 다른 프로그램에서 사용 중이지 않을 때) |
| 화면에서 정상 종료 | `q` 또는 `Esc` (`aim`/`live`) |
| 터미널에서 정상 종료 | `Ctrl-C` |
| 잔류 프로세스 정리 | `./run.sh stop` |
| 전체 옵션 | `./run.sh help` |

`--obj-thr` 기본값은 `0.5`이며 차량 표시와 lead bin 선택에만 적용된다. A₂ 원본
거리·속도 값은 바뀌지 않는다. `live` 결과는 지정한 `--out` 폴더의
`frames.jsonl`, `summary.json`, `a2_overlay.mp4`에서 확인한다. 실외 20 m 거리와
종·횡 상대속도 시험 순서는 [현장 체크리스트](docs/OUTDOOR_TEST_CHECKLIST_KR.md#8-20-m-거리와-상대속도-현장-시험-2026-09-27)에 있다.

아래의 `ros2 launch`와 Python 직접 실행 예시는 개별 구성요소 점검용이다.

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

- `camera_aravis2` ROS2 BayerRG8 입력(기본)과 OpenCV UVC prototype(`--input uvc`)
- ROS2 sensor-data QoS depth 1 및 최신 frame 하나만 유지해 queue 누적 방지
- camera가 제공한 실제 해상도/FPS 기록
- 21-frame history를 ROS header timestamp 기준으로 구성(UVC는 host timestamp)
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

Nuvo에는 프로젝트 전용 `.venv`가 준비되어 있다. 새 terminal마다 ROS2를 먼저 source하는
아래 helper를 사용한다.

```bash
cd /home/ailab/git/sangbum/A1_Challenge-main/A1_RealCamera_A2_21FPS
source scripts/activate_nuvo_env.sh
```

Nuvo RTX A5000의 470.256.02 driver에 맞춰 PyTorch 2.4.1 CUDA 11.8 wheel을 사용한다.
CUDA runtime/cuDNN/cuBLAS는 전역 `/usr/local/cuda`가 아니라 `.venv` 내부에 설치된다.
전체 CUDA 11.8 compiler toolkit은 `/home/ailab/git/sangbum/cuda-11.8`에 있으며 helper가
`CUDA_HOME`, `PATH`, `LD_LIBRARY_PATH`를 설정한다. `nvcc --version`은 11.8.89다.
runner 자체는 CUDA extension을 compile하지 않고 PyTorch wheel의 runtime을 사용한다.

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
source scripts/activate_nuvo_env.sh
```

## camera_aravis2 실행

먼저 카메라를 연결하고 검색한다. 카메라가 한 대뿐이면 launch의 `guid`를 생략할 수 있다.

```bash
source scripts/activate_nuvo_env.sh
arv-tool-0.8
ros2 launch launch/hikrobot_a2_camera_1280x720.launch.py \
  guid:=192.168.1.51 exposure_us:=15000
```

현재 검증된 launch는 1920×1200 센서의 중앙에서 `1280×720`, Offset `(320,240)`을
hardware ROI로 받아 `BayerRG8 @ 21 FPS`로 전송한다. 별도 terminal에서 topic과 encoding을
확인한다.

```bash
source scripts/activate_nuvo_env.sh
ros2 topic list
ros2 topic hz /camera_driver_gv/vis/image_raw
ros2 topic echo /camera_driver_gv/vis/image_raw --once --field encoding
```

모델은 수신한 1280×720 전체를 화면에 표시하고 `y=235:485`의 1280×250 영역만 사용한다.
software resize는 없다. 다만 중앙 hardware crop의 FOV와 학습 카메라 calibration이 같다는
근거는 아직 없으므로 `geometry_valid_for_accuracy=false`다.

## 60초 처리량 실험

정확한 throughput 측정에서는 `--display`를 켜지 않는다.

```bash
python scripts/live_camera_a2_21fps.py \
  --input ros2 \
  --ros-topic /camera_driver_gv/vis/image_raw \
  --camera-fps 21 \
  --target-fps 21 \
  --width 1280 \
  --height 720 \
  --seg-ckpt senior_code/data/ckpts/seg_pretrain_fx640_v3_backbone.pt \
  --a2-ckpt senior_code/data/ckpts/a2_mf20_v2s_n20.pt \
  --duration 60 \
  --output-dir outputs/run01_fp32
```

카메라 driver는 위의 1280×720 launch로 먼저 실행해야 한다. CUDA fp16 경로를 별도 비교할
때만 `--amp`를 추가한다.

```bash
python scripts/live_camera_a2_21fps.py \
  --input ros2 \
  --ros-topic /camera_driver_gv/vis/image_raw \
  --seg-ckpt senior_code/data/ckpts/seg_pretrain_fx640_v3_backbone.pt \
  --a2-ckpt senior_code/data/ckpts/a2_mf20_v2s_n20.pt \
  --duration 60 \
  --amp \
  --output-dir outputs/run02_amp
```

## 출력 해석

`summary.json`의 핵심 필드:

- `capture_fps_host_observed`: ROS subscriber callback에서 관측한 FPS
- `a2_output_fps_steady`: history 1초가 채워진 뒤의 a₂ output FPS
- `compute_ms.p95`: preprocessing+backbone+a₂ p95
- `capture_to_a2_ms_host.p95`: host가 frame을 받은 시점부터 a₂ 완료까지
- `capture_to_a2_ms_header.p95`: 동기화된 ROS header clock 기준 sensor timestamp부터 a₂까지
- `capture_to_output_ms_host.p95`: a₂ decode와 중앙 lead 요약까지 완료된 시간
- `stage_ms.backbone`, `stage_ms.a2`: 병목 분리
- `stage_ms.a2_decode_and_summary`, `stage_ms.visualization`: 후처리와 화면 비용
- `history_age_error_ms.p95`: 원하는 50 ms 간격 대비 timestamp 오차
- `history_duplicate_slot_ratio`: 같은 frame이 여러 temporal slot에 반복된 비율
- `pass.sustained_a2_21fps`: 최종 시스템 실행 가능성

21 FPS의 frame budget은 `1000/21 = 47.62 ms`다.

### 검증된 1280×720 hardware ROI 결과

카메라 자체를 센서 중앙 `1280x720`, Offset `(320,240)`으로 설정한 60초 재실험 결과는
`outputs/real_camera_1280x720_exp15000_fp32_60s/summary.json`에 저장되어 있다.

- ROS image: 1280x720 BayerRG8 (software resize 없음)
- capture / steady a₂: 20.999 / 20.998 FPS
- compute p95: 15.214 ms
- ROS header→a₂ p95: 26.383 ms
- skipped/history-duplicate/NaN: 모두 0
- sustained 21 FPS: PASS

중앙 hardware crop은 학습 입력과 픽셀 shape은 같지만 FOV 동일성이 검증된 것은 아니므로
여전히 `geometry_valid_for_accuracy=false`다.

화면 녹화까지 포함한 25000 µs 결과와 두 최종 실험의 비교는
[2026-09-14 실험 요약](RESULTS_2026-09-14.md)에 정리되어 있다.

## 정확도 확인용 화면

```bash
./run.sh live --exposure 2000
```

노출은 화면에서 `[`와 `]`로 실시간 조절되고, 모델 band의 포화/암부 비율이 함께 표시된다.
실내 모니터 재촬영을 재현하려면 그때 값을 그대로 준다.

```bash
./run.sh live --exposure 25000 --secs 60
```

`scripts/run_live_visualization.sh`는 2026-09-14 모니터 run을 그대로 재현하기 위해
보존한 아카이브 스크립트다. 새 run에는 사용하지 않는다.

화면 해석:

- ROI의 빨간 반투명 세로 strip: `sigmoid(a₂[:,0])` object probability
- `d=...m`: object-positive 연속 bin의 `a₂[:,6] × 40`
- `vL=...`, `vT=...m/s`: 같은 bin들의 종/횡 상대속도(`a₂[:,3:5] × 20`)
- 화면 상단 `PEAK ... RAW a2`: 선택된 bin의 원본 7채널. 다음 줄은 7채널을 확률·m/s·m로 변환한 값
- `frames.jsonl`의 `a2_bins.raw_40x7`: 매 프레임 전체 40×7 원본값 (`./run.sh live` 기본 기록)
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

실외 카트 시험 절차와 주의점은 [OUTDOOR_TEST_CHECKLIST_KR.md](docs/OUTDOOR_TEST_CHECKLIST_KR.md)에 있다.

구체적인 단계, 통과 기준과 실물 촬영 시나리오는 [REAL_CAMERA_A2_21FPS_EXPERIMENT.md](docs/REAL_CAMERA_A2_21FPS_EXPERIMENT.md)에 정리했다.

원본 코드의 파일별 출처는 [UPSTREAM_MANIFEST.md](docs/UPSTREAM_MANIFEST.md)에 있다.

선배 저장소 전체의 폴더와 파일 역할은 [SENIOR_CODE_STRUCTURE_KR.md](docs/SENIOR_CODE_STRUCTURE_KR.md)에 정리했다.

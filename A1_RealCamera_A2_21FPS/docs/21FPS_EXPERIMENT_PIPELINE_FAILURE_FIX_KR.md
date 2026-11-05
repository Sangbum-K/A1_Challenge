# 실물 카메라 A2 21 FPS 실험 총정리

작성일: 2026-09-22  
기준 실험: 2026-09-14 실내 검증, 2026-09-18 실외 카트 시험

## 1. 결론부터

이번 프로젝트가 검증하는 경로는 아래 하나다.

```text
Hikrobot 카메라 1대
→ camera_aravis2
→ ROS2 BayerRG8
→ debayer / crop / normalize
→ segmentation-pretrained EfficientNetV2-S
→ i₁ token
→ current + 과거 20개 token
→ A2ModelMultiFrame
→ a₂ (40×7)
```

- `21 FPS`는 초당 a₂ output 약 21개를 뜻하며, 한 frame 예산은 `47.62 ms`다.
- `21-frame`은 현재 token 1개와 과거 token 20개를 함께 쓴다는 뜻이다.
- 2026-09-18 실외 본 시험은 **16.45 FPS로 실패**했다.
- 카메라는 실외에서도 약 20.99 FPS로 정상 수신됐다. 카메라 대역폭이 실패 원인은 아니다.
- 같은 날, 같은 코드와 모델은 실내에서 약 21.00 FPS, compute p95 14.4 ms로 통과했다.
- 실외에서는 preprocess, backbone, A2가 함께 약 2.2~3.8배 느려졌다. 확인된 직접 병목은
  `backbone + A2 compute`의 예산 초과이며, 근본 원인은 아직 확정되지 않은 **PC의 전력·클럭·실행 환경 상태**다.
- live 화면과 MP4 기록 구간은 실패의 근본 원인은 아니지만, 느려진 상태에서 frame당 약
  14.7 ms를 더 썼다. full JSON 기록도 같은 단일 loop에서 수행돼 실제 출력률을 약 20 FPS
  수준에서 16.5 FPS까지 추가로 낮추는 데 영향을 줬다.
- 따라서 **코드를 대규모로 다시 짜거나 모델을 줄일 단계는 아니다.** 먼저 카트 전원 상태의
  순수 bench와 GPU/CPU 클럭을 함께 측정해야 한다.
- 노출, white balance, 정상 종료 문제는 바로 개선할 수 있다. 하지만 플랫폼 저하의 근본 원인을
  측정 없이 특정 설정 하나로 바로 고칠 수 있다고 단정할 수는 없다.

처리량 통과와 인식 정확도는 별도 문제다. 현재 geometry는 calibration 근거가 없어
`geometry_valid_for_accuracy=false`이고, 실외 영상은 과노출과 temporal history 훼손까지
겹쳤다. 21 FPS를 복구해도 거리·속도 정확도가 자동으로 보장되지는 않는다.

## 2. 실험 범위와 성공 기준

### 포함

- 물리 Hikrobot 카메라 1대
- `camera_aravis2`와 ROS2 image subscriber
- BayerRG8 debayer와 모델 입력 전처리
- 선배 segmentation-pretrained backbone과 i₁
- timestamp 기반 21-frame A2
- a₂ `(1,40,7)` 출력
- 처리량, latency, frame drop, history 품질, NaN/Inf 측정

### 제외

- f₁, f₂
- pedal, steering, controller
- raceline, MPC
- MORAI closed loop
- 실제 차량 제어 성능

### 처리량 합격 기준

| 항목 | 기준 |
|---|---:|
| 카메라 수신률 | 20.58 FPS 이상 (`21 × 0.98`) |
| steady a₂ 출력률 | 20.58 FPS 이상 |
| model compute p95 | 47.62 ms 이하 |
| NaN/Inf frame | 0 |
| resize | accuracy 시험에서는 불허 |

최종 `sustained_a2_21fps`는 카메라, a₂ 출력률, compute budget, finite output이 모두
통과해야 true가 된다.

## 3. 실험 환경

### 하드웨어

| 항목 | 내용 |
|---|---|
| 카메라 | Hikrobot MV-CS023-10GC, `DA8931779`, `192.168.1.51` |
| 센서 | 1920×1200, GigE Vision |
| 공식 전송 ROI | 1280×720, sensor offset `(320,240)` |
| PC | Nuvo |
| GPU | NVIDIA RTX A5000 24 GB |
| 카메라 NIC | `enp5s0`, `192.168.1.100/24`, MTU 1500, gateway 없음 |

`camlink`와 `camlink10g`가 같은 subnet/address를 쓰므로 동시에 연결하면 안 된다.
`./run.sh check`가 이 조건을 검사한다.

### 소프트웨어

| 항목 | 내용 |
|---|---|
| OS kernel | `6.8.0-138-generic` |
| CPU governor | `powersave` |
| GPU persistence | Disabled |
| NVIDIA driver | 470.256.02 |
| ROS2 | Humble |
| camera driver | `camera_aravis2`, aravis 0.8.20 |
| DDS | Fast DDS, 64 MB large-SHM profile |
| Python | 3.10.12, 프로젝트 `.venv` |
| PyTorch | 2.4.1+cu118, cuDNN 9.1.0 |
| OpenCV / timm | 4.10.0 / 1.0.20 |

실제 추론은 프로젝트 `.venv`의 CUDA runtime을 쓴다. 로컬 CUDA 11.8 toolkit 설치 자체가
매 frame 추론 경로에 들어가는 것은 아니다.

### 카메라 설정

```text
PixelFormat              BayerRG8
Width × Height           1280 × 720
OffsetX, OffsetY         320, 240
AcquisitionFrameRate     21.0 Hz
GevSCPSPacketSize        1500
Gain                     0 dB (고정 노출 run)
```

9월 18일 실외 run은 `ExposureAuto=Off`, 20000 또는 25000 µs를 사용했다. 이 값은 실내용
값이어서 주간 실외에서는 모델 band가 크게 포화됐다.

## 4. 코드와 파일 역할

| 파일 | 역할 |
|---|---|
| `run.sh` | `check`, `aim`, `live`, `bench`, `replay`, `stop` 단일 진입점 |
| `scripts/live_camera_a2_21fps.py` | ROS2 수신, 전처리, backbone, history, A2, 계측, 로그/화면 |
| `launch/hikrobot_a2_camera_1280x720.launch.py` | camera_aravis2와 GenICam 설정 실행 |
| `scripts/preflight.sh` | bundle, GPU, 디스크, NIC, 카메라, 잔류 process 점검 |
| `scripts/aim_camera.py` | 모델 없이 조준, 노출, 포화, 초점 확인 |
| `scripts/camera_params.py` | 실행 중 exposure parameter 변경 |
| `configs/real_camera_mf20.yaml` | 입력 계약과 acceptance 기준 |
| `configs/fastdds_large_shm.xml` | ROS2 image용 Fast DDS SHM 설정 |
| `configs/bag_qos_override.yaml` | best-effort image bag 기록 QoS |
| `senior_code/src/model/image_models/a2_model.py` | 수정하지 않은 선배 원본 모델 |
| `senior_code/data/ckpts/seg_pretrain_fx640_v3_backbone.pt` | segmentation-pretrained backbone |
| `senior_code/data/ckpts/a2_mf20_v2s_n20.pt` | 21-frame A2 checkpoint |

`senior_code`의 원본과 checkpoint는 수정 대상이 아니다. adapter와 runner만 프로젝트 쪽에서
수정한다. 현재 bundle은 아래 명령으로 3개 핵심 파일의 hash 검증을 통과한다.

```bash
python3 scripts/verify_portable_bundle.py
```

원본 `closed_loop_node.py`를 쓰지 않는 이유는 ROS1/MORAI, f₁/f₂/controller에 결합돼 있고
`INFER_RATE_HZ=20` cap이 있어 이번 camera→a₂ 21 FPS 시험을 그대로 측정할 수 없기 때문이다.

## 5. 입력부터 a₂까지의 파이프라인

### 5.1 카메라와 ROS2

1. 카메라가 1280×720 BayerRG8을 21 Hz로 전송한다.
2. `camera_aravis2`가 `/camera_driver_gv/vis/image_raw`에 `sensor_msgs/Image`를 publish한다.
3. runner는 sensor-data QoS, best effort, keep-last depth 1로 subscribe한다.
4. callback에서 `cv_bridge`로 BayerRG8을 BGR8 1280×720으로 debayer한다.
5. runner는 queue를 쌓지 않고 항상 최신 frame만 처리한다. 추론이 느리면 오래된 frame을
   처리하는 대신 중간 frame을 버린다.

이 정책은 latency 누적을 막지만, compute가 21 Hz를 못 따라가면 `skipped_camera_frames`가
증가한다.

### 5.2 영상 전처리

```text
BGR uint8 (720,1280,3)
→ y=235:485 crop
→ BGR to RGB
→ float32 / 255
→ ImageNet mean/std normalization
→ CHW + batch
→ tensor (1,3,250,1280)
→ GPU
```

기본 실행에서는 입력 shape가 다르면 오류를 내며, 임의 resize는 하지 않는다.

공식 1280×720 ROI의 sensor OffsetY가 240이므로 모델이 실제로 보는 sensor row는
`240 + 235 = 475`부터 `725`까지다.

### 5.3 backbone과 i₁

```text
(1,3,250,1280)
→ tf_efficientnetv2_s final feature
→ 약 (1,256,8,40)
→ adaptive average pool, output (height=1, width=40)
→ squeeze / transpose
→ raw i₁ token (1,40,256)
```

현재 frame의 backbone은 한 번만 실행한다. 과거 20장의 이미지를 매 output마다 다시
backbone에 넣지 않고, GPU에 보관한 과거 i₁ token을 재사용한다.

### 5.4 21-frame history 구성

원래 학습 데이터는 100 Hz이며 과거 offset `[5,10,...,100]` tick을 사용했다. 실시간 runner는
동일한 시간 간격을 ROS header timestamp 기준으로 재현한다.

```text
current token                         1개
target age 0.05, 0.10, ... 1.00 s   20개
합계                                 21개
```

각 목표 age마다 buffer에서 timestamp가 가장 가까운 token을 선택한다. startup에 과거가
부족하거나 중간 frame이 많이 빠지면 같은 source token이 여러 slot에 재사용된다.
`history_unique_sources`와 `history_duplicate_slot_ratio`가 이 훼손을 측정한다.

### 5.5 A2ModelMultiFrame

각 raw token `(B,40,256)`은 다음 순서로 처리된다.

```text
21개 raw i₁ token
→ shared Conv1d projection
→ lateral position + temporal embedding
→ current token을 query로 사용
→ [ego token, 과거 20 frame token]을 key/value로 사용
→ temporal cross-attention + post self-attention
→ fc projection, d_out=128
→ 7-channel linear head
→ a₂ (B,40,7)
```

`ego_motion=(B,2)`의 `[vx, wz]`는 실제 attention 입력이다. 정지 또는 매우 느린 카트에서는
`[0,0]`이 맞지만, 움직이는 차량에서 계속 `[0,0]`을 넣으면 특히 속도 channel에
train/deploy mismatch가 생긴다. 현재 `--ego-vx`, `--ego-wz`는 run 전체에서 고정값이며
CAN 실시간 연동은 아직 없다.

## 6. a₂ 값의 의미와 decode

a₂는 40개 가로 bin 각각에 7개 값을 출력한다.

| channel | raw 의미 | runner decode |
|---:|---|---|
| 0 | object logit | `sigmoid(raw)` → object probability |
| 1 | longitudinal absolute velocity | `raw × 20` m/s |
| 2 | lateral absolute velocity | `raw × 20` m/s |
| 3 | longitudinal relative velocity | `raw × 20` m/s |
| 4 | lateral relative velocity | `raw × 20` m/s |
| 5 | proximity | raw 값, 표시에서만 0~1 clamp |
| 6 | planar distance | `raw × 40` m |

학습 label의 proximity는 `exp(-distance/20)` 형태다. 그러나 출력 head에서 proximity와
distance는 독립 channel이므로 두 값이 수학적으로 항상 일치하도록 강제되지는 않는다.

40 bin은 영상 가로 1280 px를 32 px씩 나눈 구조다. 중앙 관심영역 bin 16~25는 화면 표시와
요약을 위한 고정 범위일 뿐 차선 검출 결과나 차량 bounding box가 아니다.

모델이 수치를 출력한다는 사실만으로 실제 거리와 상대속도가 정확하다고 볼 수 없다. 정확도
판정을 위해서는 카메라 FOV/intrinsic, optical center, 장착 높이·pitch·yaw와 실제 GT가 필요하다.

## 7. 결과 비교

### 대표 결과

| 조건 | capture FPS | a₂ FPS | compute | 결과 |
|---|---:|---:|---:|---|
| 9/14, 1280×720, FP32, 60초 bench | 20.999 | 20.998 | p95 15.214 ms | PASS |
| 9/18 실내 live run 1 | 20.997 | 20.997 | p50 13.719 / p95 14.359 ms | PASS |
| 9/18 실내→실외 live run 4 | 20.989 | 16.497 | p50 49.559 / p95 51.225 ms | FAIL |
| 9/18 실외 본 run 5 | 약 20.99 | 16.453 | p50 약 49.5 / p95 57.3 ms | FAIL |

### 단계별 중앙값

| 단계 | 9/18 실내 | 9/18 실외 | 변화 |
|---|---:|---:|---:|
| preprocess | 3.8 ms | 8.3 ms | 2.2× |
| backbone | 6.0 ms | 22.5 ms | 3.8× |
| A2 | 4.6 ms | 17.7 ms | 3.8× |
| compute 합 | 13.7 ms | 49.5 ms | 3.6× |
| visualization | 약 8.9 ms | 약 14.7 ms | 추가 부담 |

`compute_ms`는 preprocess 시작부터 A2 완료까지이며, A2 decode와 시각화는 제외된다. 반면
실제 steady output FPS는 같은 단일 처리 loop의 이전 frame 시각화, window 처리, MP4 write
시간 영향을 받는다.

실외 상태의 compute p50 49.5 ms만으로도 47.62 ms 예산을 넘는다. 여기에 live 부가 작업이
더해져 output interval p50이 약 64.8 ms, 즉 약 16.5 FPS가 됐다.

### frame drop과 history 손상

실외 본 run에서는 카메라가 20.99 FPS로 계속 들어왔지만 runner가 따라가지 못했다.

- 수신 frame 중 약 21.7%가 최신-frame 정책에서 건너뛰어졌다.
- 과거 20 slot의 고유 source는 중앙값 16개로 줄었다.
- frame의 약 83~93%가 과거 고유 source 20개를 모두 확보하지 못했다.
- 같은 token이 서로 다른 age slot에 반복되면서 상대속도 추정에 필요한 시간 정보가 손상됐다.

따라서 낮은 인식률을 순수 sim-to-real 문제 하나로 해석할 수 없다. 과노출과 temporal input
손상이 동시에 존재했다.

## 8. 21 FPS가 실패한 이유

### 확인된 사실

1. **카메라는 원인이 아니다.** 실외에서도 약 20.99 FPS로 정상 수신됐다.
2. **실외의 model compute가 budget을 넘었다.** p50 약 49.5 ms, p95 51~57 ms였다.
3. **CPU와 GPU 단계가 함께 느려졌다.** backbone과 A2만이 아니라 preprocess도 느려졌다.
4. **live 부가 작업이 추가 하락을 만들었다.** 시각화 p50 약 14.7 ms가 같은 loop에 포함됐다.
5. **첫 frame부터 느렸고 시간에 따라 단조 악화하지 않았다.** run 중간에는 30 ms대로
   회복됐다가 다시 49 ms대로 돌아왔다.
6. **동일 모델은 같은 날 실내에서 충분히 빨랐다.** 따라서 모델 크기 자체가 21 FPS를
   원천적으로 막는다는 결론은 성립하지 않는다.

### 현재 가장 강한 원인 가설

실내 책상 환경에서 카트 전원/실외 환경으로 이동할 때 바뀐 **GPU·CPU clock, power state,
전원 공급 또는 display/compositor 경합**이다. CPU와 GPU가 함께 느려졌고 발열 누적 형태가
아니기 때문에 이 가설이 가장 설명력이 높다.

단, 당시 GPU clock, power draw, throttle reason, CPU frequency를 함께 저장하지 않았으므로
“전원 부족”이나 “GPU P-state” 중 하나로 아직 확정할 수 없다.

### 원인이 아닌 것 또는 우선순위가 낮은 것

- **카메라 FPS/네트워크:** 수신률이 정상이다.
- **21장을 매번 backbone 처리:** 실제 코드는 현재 frame 1장만 처리하고 과거 token을 재사용한다.
- **센서 hardware crop 부족:** backbone 입력은 이미 250×1280이다. crop을 줄여도 backbone과
  A2의 40.2 ms는 변하지 않는다.
- **누적 발열:** 첫 frame부터 느리고 중간 회복 구간이 있어 단조 thermal throttling과 맞지 않는다.
- **checkpoint 손상:** 동일 checkpoint가 실내에서 통과했고 portable hash 검증도 통과한다.

## 9. 바로 고칠 수 있는가

### 판단표

| 항목 | 지금 바로 가능? | 조치 |
|---|---|---|
| live 부가 부하 제거 | 예 | 처리량 판정은 `bench`로 수행; 필요 시 `--no-display --no-video --no-full-log` |
| 실외 과노출 | 예 | `aim`으로 clipped < 5% 조정 또는 auto exposure 사용 |
| 녹황색 color cast | 예 | auto exposure 경로의 continuous white balance 사용/현장 확인 |
| SIGTERM 시 summary/MP4 유실 | 예, 코드 수정 필요 | runner에 SIGINT/SIGTERM graceful shutdown handler 추가 |
| GPU/CPU power state | 조건부 | 계측으로 확인 후 performance governor/persistence/power 공급 조치 |
| hardware 1280×250 crop | 가능하지만 후순위 | debayer/전송 절감용이며 현재 21 FPS 실패의 근본 해결책은 아님 |
| 실제 거리·속도 정확도 | 즉시 불가 | calibration과 GT 기반 별도 검증 필요 |
| 움직이는 차량의 ego input | 즉시 불가 | CAN 등에서 실시간 `vx`, `wz` 연동 필요 |

즉, **실험을 다시 통과시킬 가능성은 높다.** 이미 동일 장비와 모델이 여러 번 21 FPS를
통과했다. 다만 가장 중요한 플랫폼 저하는 다음 측정으로 원인을 확정한 뒤 고쳐야 한다.

## 10. 가장 짧은 수정·재검증 절차

### 1단계: 카트 전원 상태에서 순수 bench와 GPU 상태 동시 기록

```bash
cd /home/ailab/git/sangbum/A1_Challenge-main/A1_RealCamera_A2_21FPS
./run.sh check

nvidia-smi \
  --query-gpu=timestamp,pstate,clocks.sm,clocks.gr,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active \
  --format=csv -l 1 > /tmp/a2_gpu_before_fix.csv &
GPU_LOG_PID=$!

./run.sh bench --secs 60
kill -INT "${GPU_LOG_PID}"
```

같은 시험을 실내 상시 전원과 카트 전원에서 각각 한 번씩 수행한다. 인용할 처리량 결과는
화면과 영상 기록을 켠 `live`가 아니라 이 `bench` 결과로 정한다.

### 2단계: 결과에 따라 분기

#### bench가 다시 14~20 ms대면

모델과 전원은 정상이고 display/compositor 또는 live 기록 경합 가능성이 크다.

- 처리량 검증은 bench 결과로 완료한다.
- 실제 인식 run에서는 불필요한 MP4/full JSON을 끈다.
- 원본 보존이 필요하면 raw bag을 기록하고 offline replay로 overlay를 만든다.

#### bench도 49 ms대면

전원·클럭 상태 문제를 우선 처리한다.

```bash
sudo cpupower frequency-set -g performance
sudo nvidia-smi -pm 1
./run.sh bench --secs 60
```

설정 전후의 GPU log와 summary를 비교한다. persistence와 governor로 회복되지 않으면 카트의
인버터/배터리 출력, GPU power limit, 케이블, host 전원 profile을 확인한다. 설정을 바꾸기
전에 반드시 baseline log를 남겨야 어떤 조치가 효과가 있었는지 알 수 있다.

### 3단계: 실외 영상 품질 수정

```bash
./run.sh aim
```

- 모델 band의 `clipped < 5%`를 목표로 한다.
- 그늘과 양지를 오가면 고정 20000~25000 µs를 사용하지 않는다.
- auto 경로 예시:

```bash
./run.sh live --exposure auto --exposure-max 2000 --bag
```

auto exposure와 continuous white balance의 실제 적용 상태를 화면과 bag으로 확인한다.

### 4단계: 종료 안정성 보완

현재 runner는 `KeyboardInterrupt`는 처리하지만 SIGTERM을 별도로 graceful-stop으로
변환하지 않는다. 다음 현장 run 전에 SIGINT/SIGTERM 모두 main loop 종료 후 아래 작업을
거치도록 수정한다.

```text
frame log flush/close
→ VideoWriter.release()
→ camera.stop()
→ summary.json 작성
```

수정 전에는 `kill -9`를 쓰지 말고 화면의 `q`/Esc, Ctrl-C 또는 `./run.sh stop`을 사용한다.

### 5단계: 합격 후 정확도 시험 분리

21 FPS가 복구된 뒤 아래 조건을 별도 충족해야 정확도를 논할 수 있다.

- 학습 카메라와 실제 카메라의 FOV/intrinsic 대응 확인
- optical center와 장착 pose 확인
- 이동 차량이면 실시간 ego motion 입력
- 실제 차량 거리와 상대속도 GT
- 적정 exposure/white balance
- history unique source 20/20에 가까운 상태

## 11. hardware crop을 적용한다면

현재 모델이 보는 sensor 영역과 픽셀 단위로 같은 1280×250 입력은 다음과 같다.

```text
Width=1280
Height=250
OffsetX=320
OffsetY=475
runner ROI y=0:250
```

이 변경은 1280×720 전체 debayer와 전송량을 약 65% 줄일 수 있다. 하지만 backbone input과
A2 token 수는 그대로이므로 실외에서 느려진 GPU 연산을 복구하지 못한다. 또한 전체 frame을
보며 조준하기 어려워지는 trade-off가 있다. 플랫폼 문제 해결 후 여유 최적화로만 검토한다.

## 12. 현재 남은 기술 부채

- SIGTERM graceful shutdown 미구현
- 카트/차량 실시간 CAN ego motion 미연동
- FOV/intrinsic 및 장착 pose calibration 미완료
- 실외 주간 최적 노출 범위 미확정
- run 5는 강제 종료로 `summary.json`과 정상 MP4 index가 없고 `frames.jsonl`만 유효
- 카트 전원 상태의 GPU/CPU clock·power 동시 로그가 없음

## 13. 근거 파일

- 전체 인수인계: [`../PROJECT_CONTEXT.md`](../PROJECT_CONTEXT.md)
- 9월 18일 실외 결과: [`../RESULTS_2026-09-18.md`](../RESULTS_2026-09-18.md)
- 9월 14일 실내 결과: [`../RESULTS_2026-09-14.md`](../RESULTS_2026-09-14.md)
- 실험 설계: [`REAL_CAMERA_A2_21FPS_EXPERIMENT.md`](REAL_CAMERA_A2_21FPS_EXPERIMENT.md)
- 현장 절차: [`OUTDOOR_TEST_CHECKLIST_KR.md`](OUTDOOR_TEST_CHECKLIST_KR.md)
- 선배 코드 구조: [`SENIOR_CODE_STRUCTURE_KR.md`](SENIOR_CODE_STRUCTURE_KR.md)
- 실내 60초 bench 결과: [`../outputs/real_camera_1280x720_exp15000_fp32_60s/RESULT.md`](../outputs/real_camera_1280x720_exp15000_fp32_60s/RESULT.md)
- 실내 visualization 결과: [`../outputs/real_camera_1280x720_exp25000_visual_60s/RESULT.md`](../outputs/real_camera_1280x720_exp25000_visual_60s/RESULT.md)

## 14. 최종 요약

```text
카메라 21 FPS 수신             정상
동일 모델의 21 FPS 가능성       이미 실내에서 확인
실외 카트 run의 model compute   47.62 ms 예산 초과
live 시각화/기록                추가 FPS 하락 요인
가장 유력한 근본 원인           전력·클럭·실행 환경 상태
실외 낮은 인식률의 주원인       과노출 + history 손상
hardware crop                   보조 최적화, 근본 해결책 아님
즉시 할 일                      카트 전원 bench + clock/power log
그다음                          노출/WB 수정, graceful shutdown, 재시험
```

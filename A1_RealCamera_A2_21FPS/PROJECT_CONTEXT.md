# A1 실물 카메라 a₂ 실험 — 세션 간 인수인계

마지막 갱신: 2026-09-18 (Asia/Seoul)

이 문서는 새 Codex 세션에서 기존 대화의 목표와 결정사항을 다시 파악하기 위한 기준 문서다. 새 세션은 작업 전에 이 파일을 먼저 읽는다.

## 1. 사용자의 실제 목표

검증 대상은 f₂ 전체 출력이나 차량 제어가 아니다.

> Hikrobot 실물 카메라 한 대의 입력을 `camera_aravis2 + ROS2`로 받아, 선배들이 만든 perception 경로를 그대로 사용했을 때 `segmentation-pretrained backbone/i₁ → 21-frame A2ModelMultiFrame → a₂`가 지속 21 FPS로 실행되는지 검증한다.

범위:

```text
Hikrobot camera 1대
→ camera_aravis2
→ ROS2 sensor_msgs/Image (BayerRG8 raw)
→ debayer/RGB 및 A2 입력 geometry 변환
→ ImageNet normalization
→ segmentation-pretrained EfficientNetV2-S backbone
→ raw i₁ token
→ timestamp 기반 current + past 20 token
→ A2ModelMultiFrame
→ a₂ logits (1,40,7)
```

범위 밖:

- f₁/f₂
- pedal/steering
- raceline/MPC
- MORAI closed-loop
- controller 성능

`21 FPS`와 `21-frame`은 다른 의미다.

- 21 FPS: 초당 약 21개 a₂ output, frame budget 약 47.62 ms
- 21-frame: 현재 1개 + 과거 20개, 목표 age 0.05, 0.10, …, 1.00초

## 2. 카메라 검증에 대한 현재 전제

workspace의 `MULTICAM_VERIFICATION_260907.md`에는 Hikrobot 3대 동시 시험이 기록되어 있다.

- 1920×1120 BayerRG8
- 문서상 25.00 FPS, 3대 동시
- 60초 총 유실 0
- 10GbE uplink 점유 12.90%
- CPU debayer + 2.5배 등방 축소 p95 최대 6.70 ms
- camera_aravis2 1.1.0, ROS2 Humble, Fast DDS

사용자는 별도 시험에서 1920×1120 40 FPS 수신도 잘 된다고 명시했다. 따라서 현재 프로젝트에서는 카메라 acquisition/bandwidth를 다시 핵심 검증으로 삼지 않는다. 최종 방향은 카메라 1대만 사용하는 것이다.

## 3. 선배 시뮬레이터 입력 계약

원본 data path:

```text
CAMERA_1/{tick}_Intensity.jpeg
```

영상 입력:

- MORAI에서 저장된 RGB JPEG
- decode 후 NumPy `uint8`, shape `(720,1280,3)`, 값 0–255
- 세로 ROI `y=235:485`, 가로 전체 유지
- ROI shape `(250,1280,3)`
- `/255.0`으로 RGB를 float32 0–1로 변환
- ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`
- channel-first 및 batch 추가 후 backbone input `(1,3,250,1280)`

JPEG 자체가 모델 입력인 것은 아니다. 시뮬과 실물 모두 최종적으로 같은 normalized RGB tensor를 backbone에 주면 코드상 호환된다.

```text
시뮬: JPEG decode → RGB uint8 → normalize → tensor
실물: BayerRG8 → debayer → RGB uint8 → normalize → tensor
```

JPEG compression artifact와 sensor/debayer/color response 차이는 실행 형식 문제가 아니라 sim-to-real domain 차이다.

## 4. backbone과 mf20 a₂ 입력

```text
normalized image (1,3,250,1280)
→ tf_efficientnetv2_s last feature
→ adaptive average pool to height 1, width 40
→ raw i₁ token (1,40,256)
```

a₂ 학습은 raw image를 매번 처리하지 않고 미리 계산한 i₁ cache를 사용했다. cache token은 `(40,256)` float16으로 저장되고 학습 시 float32로 읽었다.

mf20 입력:

- current token 1개
- past token 20개
- source recording tick: 1 tick = 10 ms, 100 Hz
- offsets `[5,10,15,…,100]` tick = 0.05–1.00초
- total 21 × `(1,40,256)` raw token
- `ego_motion=(1,2)`는 `[0,0]`
- yaw shift off
- `forward_tokens()` output: a₂ `(1,40,7)`, optional i₂ `(1,40,128)`

매 output마다 과거 이미지 20장을 backbone에 다시 넣지 않는다. 현재 이미지 backbone 1회 후 과거 raw token을 buffer에서 재사용한다.

## 5. a₂ label/output 의미

선배들은 MORAI `EGO_INFO`와 `OBJECT_INFO`로 `(40,7)` GT를 만들었다. 이 정보는 a₂ 입력이 아니라 학습 label 생성에만 사용됐다.

| channel | 의미 |
|---:|---|
| 0 | object logit/presence |
| 1 | longitudinal absolute velocity |
| 2 | lateral absolute velocity |
| 3 | longitudinal relative velocity; 음수는 ego가 따라잡는 방향 |
| 4 | lateral relative velocity |
| 5 | proximity, `exp(-distance/20)` |
| 6 | planar distance |

배포 mf20 학습 설정은 width 1280, fx 640, cx 640, lateral bin 40개, bin width 32 px, `scale_v=20`, `scale_dist=40`을 사용했다. 계산상 horizontal FOV 기준은 약 90도다. 실제 카메라 입력을 맞출 때 shape뿐 아니라 FOV/intrinsic, optical center, mounting height/pitch/yaw를 고려해야 한다.

## 6. 실물 hardware ROI에 대한 결론

카메라가 지원하면 센서/GenICam 단계에서 `Width=1280`, `Height=250`, 적절한 `OffsetX/OffsetY`를 지정해 `1280×250 BayerRG8`만 전송하는 것이 가능하다.

장점:

- 1920×1120 대비 pixel/전송량 약 85% 감소
- debayer 대상 감소
- software resize 제거 가능

그러나 crop은 resize가 아니다. 1920 폭의 중앙 1280만 자르면 좌우 FOV가 줄어든다. hardware ROI가 선배 학습 입력의 전체 수평 시야와 약 90도 FOV를 재현하는 경우에만 바로 `1280×250`을 쓰는 것이 타당하다. Bayer phase를 위해 offset increment/짝수 정렬도 실제 카메라 capability에서 확인한다.

FOV가 맞지 않으면 원본 수평 시야를 유지하는 ROI(예: 1920×375)를 hardware에서 받고 1280×250으로 등방 축소하는 대안이 있다. 정확한 ROI/offset은 물리 카메라 calibration과 장착 자세를 확인한 후 확정해야 한다.

## 7. 실물 검증 단계에 대한 합의

모니터에 앞차가 접근/이탈하는 영상을 띄우고 실물 카메라로 촬영하는 시험은 1차 smoke/sanity test로 유용하다.

성공 수준을 분리한다.

1. 기술적 성공: camera→a₂가 sustained 21 FPS, NaN/오류/queue 누적 없음
2. 반응성 성공: 접근 시 distance 감소, 이탈 시 증가, vL_rel 방향 변화, object probability 반응
3. 실제 성능 성공: 실제 도로/차량과 거리·속도 GT로 정확도 검증

모니터 재촬영 결과만으로 실제 거리 m/s 정확성이나 전체 sim-to-real 성공을 주장하지 않는다. f₂ 숫자가 나온다는 사실도 이번 범위의 성공 기준이 아니다.

## 8. 원본 선배 코드와 portable 프로젝트

원본 저장소:

```text
/Users/sangbum/Workspace/A1_Challenge/scc_controller_imitator-main
```

portable 실험 프로젝트:

```text
/Users/sangbum/Workspace/A1_Challenge/A1_RealCamera_A2_21FPS
```

`A1_RealCamera_A2_21FPS/senior_code/`는 symlink가 아닌 실제 복사본이다. 실험 폴더 하나만 Ubuntu PC로 옮길 수 있도록 구성했다.

포함된 byte-identical 선배 파일:

- `src/model/image_models/a2_model.py`
- `seg_pretrain_fx640_v3_backbone.pt` 약 77 MB
- `a2_mf20_v2s_n20.pt` 약 83 MB
- `COPY_MANIFEST.sha256`

핵심 SHA-256:

- `a2_model.py`: `6763cfe442ec7be32eb22d0873b2d8b8577073b3d1f435fb6878d99337c2f5a0`
- seg checkpoint: `8b9262a5bdd214081b62223f1fee3bb96f50c55b3f3b5bcc3300a5ba30a8fd91`
- mf20 checkpoint: `c095b16f15cd73614f2f829bba774095e9c65626b895d4cf1558692953f44003`

검증:

```bash
python3 A1_RealCamera_A2_21FPS/scripts/verify_portable_bundle.py
```

정리 후 검증 기준은 `verified 3 bundled senior files`이며 세 runtime 파일은 원본과 byte-identical이다. project 안에는 symlink가 없다.

선배 파일과 checkpoint는 수정하지 않는다. 원본 `A2ModelMultiFrame` constructor가 timm pretrained download를 요청하는 문제는 portable runner가 constructor 호출 중 download flag만 억제하고 bundled checkpoint를 즉시 load하도록 처리했다.

## 9. 현재 구현 상태와 중요한 미완료점

현재 runner:

```text
A1_RealCamera_A2_21FPS/scripts/live_camera_a2_21fps.py
```

현재 구현된 기능:

- camera_aravis2 ROS2 `sensor_msgs/Image` subscriber가 기본 입력
- sensor-data QoS depth 1 및 latest-frame-only 처리
- `cv_bridge`의 BayerRG8→BGR8 debayer 후 기존 RGB normalization 경로 연결
- ROS header timestamp 기반 history와 header/host-arrival latency 분리 기록
- OpenCV UVC capture는 `--input uvc` fallback prototype으로 유지
- 1280×720 → y=235:485
- BGR→RGB, ImageNet normalization
- exact senior backbone/a₂ checkpoint load
- timestamp current + past 20 token selection
- stage p50/p95/p99, output FPS, capture-to-a₂ latency, drop/history 품질
- 실시간 모델 ROI 위 A₂ 40-bin object probability strip, 거리/vL_rel annotation,
  proximity bar, 고정 중앙 관심영역 bin 16~25 overlay
- 화면 상단은 software resize 없는 실제 1280×720 수신 frame 전체이며, 모델이 사용하는
  `x=0:1280, y=235:485` crop은 초록 경계로 표시한다. 노란색은 카메라 영상 위에서
  제거했고 하단 패널의 lead-search bin 16~25 표시에만 사용한다. 이는 crop/차선/차량
  bounding box가 아니다.
- 인접 object-positive bin group 단위 표시와 선택적 raw `(40,7)` JSON logging/overlay MP4 저장
- 모델 compute와 A₂ decode/시각화 시간을 별도 계측
- `frames.jsonl`, `summary.json`
- f₂/controller 미사용

2026-09-14 Nuvo에서 ROS2 adapter와 실제 camera run을 검증했다. 처음에는 `enp5s0`의
NetworkManager `camlink`가 DHCP로 바뀌어 주소를 못 받았고 discovery가 실패했다.
`enp5s0=192.168.1.100/24`, MTU 1500, no gateway로 복구한 뒤 실제 연결된 카메라
`Hikrobot-MV-CS023-10GC-DA8931779 (192.168.1.51)`를 Aravis와 camera_finder 양쪽에서 찾았다.

초기 1920×1120 smoke/resize run으로 실제 BayerRG8 receipt, debayer, QoS와 CUDA 경로를
검증한 뒤, 아래의 1280×720 hardware-ROI 공식 결과로 대체했다. 중복되는 예비 산출물은
2026-09-14 정리 과정에서 제거했다. Fast DDS large-SHM profile은 driver와 runner 양쪽에
계속 적용한다.

사용자 요청으로 camera input 자체를 중앙 hardware ROI 1280×720으로 변경하고 노출을
15000 µs로 올린 뒤 전체 실험을 다시 수행했다. 센서 1920×1200에서 Offset=(320,240),
BayerRG8, packet 1500, 21 FPS이며 software resize는 사용하지 않았다. 결과 파일은
`outputs/real_camera_1280x720_exp15000_fp32_60s/summary.json`이다.

- ROS image 1280×720 BayerRG8, exposure 15000 µs
- measured 1261 frames, steady 1240 frames
- capture 20.999 FPS, steady a₂ 20.998 FPS
- compute p95 15.214 ms, ROS callback arrival→a₂ p95 15.867 ms
- ROS header→a₂ p95 26.383 ms
- skipped frame 0, history duplicate 0, NaN/Inf 0
- `sustained_a2_21fps=true`, `resized_any_frame=false`

shape은 선배 입력 1280×720과 같지만 1920 센서에서 중앙 crop한 것이므로 수평 FOV가
학습 카메라와 같다는 calibration 근거는 아직 없다. 따라서 명시적으로
`geometry_valid_for_accuracy=false`이며 이 결과는 throughput 성공이다.

같은 결과를 사람이 읽기 쉽게 정리한 보고서는
`outputs/real_camera_1280x720_exp15000_fp32_60s/RESULT.md`에 있다.

사용자와 함께 모니터 주행 영상을 촬영한 25000 µs 정식 시각화 60초 run도 수행했다.
결과와 직접 화면 검토는
`outputs/real_camera_1280x720_exp25000_visual_60s/RESULT.md`에 정리했다.

- 화면 표시 + overlay MP4 + full raw `(40,7)` A₂ JSON을 동시에 기록
- measured 1260, steady 1239 frame
- capture 20.997 FPS, steady a₂ 20.996 FPS, compute p95 15.595 ms
- skipped 0, history duplicate 0, NaN/Inf 0, sustained 21 FPS PASS
- 0–10초 비주행/전환 화면의 중앙 lead 검출률 0%
- 10–60초 주행 영상의 중앙 lead 검출률 53.67%
- lead-positive run median 2 frame, 최장 약 1.90초로 반응이 분절됨
- 기술 처리량은 PASS지만 화면 반응성은 PARTIAL이며 거리·상대속도 정확도 성공이 아님

Nuvo runtime은 `A1_RealCamera_A2_21FPS/.venv`에 구성한다. RTX A5000 driver 470.256.02와
호환되는 PyTorch 2.4.1 CUDA 11.8 조합이며 CUDA runtime/cuDNN은 `.venv` 안에 격리한다.
full CUDA 11.8 toolkit도 `/home/ailab/git/sangbum/cuda-11.8`에 로컬 설치되어 있고
`nvcc V11.8.89`를 확인했다. runner는 CUDA extension을 compile하지 않으므로 실제 inference는
PyTorch wheel의 격리된 runtime을 사용한다.

원본 `closed_loop_node.py`를 그대로 쓰면 안 되는 이유:

- ROS1/MORAI topic과 message에 결합
- f₁/f₂/controller 포함
- `INFER_RATE_HZ=20` cap
- 이번 목표인 ROS2 physical-camera→a₂ 21 FPS를 그대로 측정할 수 없음

Nuvo `.venv`의 torch/cv2/timm, 실제 RTX A5000, ROS2 BayerRG8 camera runtime을 검증했다.

## 9-1. 2026-09-18 실외 카트 인식 시험 준비 변경

### 시험 형태 (사용자 확인)

카트에 본체와 모니터를 올려 실외로 나가서 실제 차량 인식 여부를 **화면으로 실시간
확인**하는 시험이다. 60 km/h 주행이 아니다. 따라서:

- 노출은 auto가 아니라 **상수**로 충분하다. 단 실내용 15000~25000 µs는 주간 실외에서
  포화되므로 값 자체는 현장에서 맞춰야 한다.
- 카트는 정지에 가까우므로 `ego_motion=[0,0]`이 맞다. `--ego-vx`는 구현해뒀지만
  이 시험에서는 기본값 0을 쓴다.
- 원본 bag 녹화는 필수가 아니다(재현이 쉬움). `--bag` opt-in으로 두었다.

모니터 재촬영에서 실외로 넘어가기 위해 아래를 수정했다. 선배 vendored
파일과 checkpoint는 건드리지 않았다.

실행 경로를 `./run.sh` 하나로 통합했다.

```text
./run.sh check | features | aim | live | bench | replay <bag> | stop
```

`launch/hikrobot_a2_camera_1280x720.launch.py`를 `OpaqueFunction` 기반으로 바꾸고
guid/ROI/offset/fps/노출/게인/화이트밸런스를 전부 launch argument로 뺐다. **기본값은
2026-09-14 검증 설정과 완전히 동일**하다(ExposureAuto=Off, 15000 µs, Gain 0,
BalanceWhiteAuto=Off, Offset (320,240), packet 1500). 파라미터 생성 결과를 검증 설정과
assert로 비교해 동일함을 확인했다.

`scripts/live_camera_a2_21fps.py` 변경:

- `--ego-vx/--ego-wz` 추가. `A2ModelMultiFrame`의 `ego_motion (B,2)`는 `ego_proj`를 거쳐
  cross-attention KV에 들어가는 **실제 모델 입력**이다. 정지 카메라에서는 `[0,0]`이 맞지만
  움직이는 차량에서 `[0,0]`은 train/deploy mismatch다. `summary.json`에 기록한다.
- `--duration 0` = q/Esc/Ctrl-C까지 무제한. 주행 run 길이를 미리 모를 때 쓴다.
  Ctrl-C에서도 `summary.json`이 정상 기록되는 것을 확인했다.
- `--first-frame-timeout` 추가. 기존 5초 고정값은 bag replay에서 model load와 bag 재생
  시작이 경주해 실패할 수 있었다.
- `DISPLAY`가 없는데 `--display`를 주면 경고 후 자동으로 끈다.
- ROI 높이가 학습값 250과 다르면 경고한다.
- `summary.json`에 `roi_y`, `ego_motion_input` 추가.
- `--tune-exposure`: display 중 `[`/`]` 키로 노출을 실시간 변경하고, 모델 band의
  clipped/crushed 비율을 화면에 표시한다. `./run.sh live`가 기본으로 켠다.
  `configs/camera_dynamic_parameters.yaml`의 `ExposureTime`이 ROS 파라미터로 노출되는
  것을 이용한다. 파라미터 이름은 추측하지 않고 `ros2 param list`로 런타임에 찾는다.
  못 찾으면 화면에 `exposure not tunable`을 띄우고 나머지는 그대로 동작한다.
- ego tensor를 루프 밖으로 hoist해 frame당 H2D 복사를 제거했다.

신규 파일:

- `run.sh`: 단일 진입점
- `scripts/camera_params.py`: camera_aravis2 dynamic parameter 실시간 setter
- `scripts/preflight.sh`: bundle/GPU/커널/governor/디스크/NIC/카메라/잔류 프로세스 점검
- `scripts/aim_camera.py`: 모델 없이 즉시 뜨는 조준 창. 모델 band, 40 bin 경계,
  clipped/crushed 비율, Laplacian 초점 지표, 판정 메시지
- `configs/bag_qos_override.yaml`: `ros2 bag record`가 best-effort publisher를 받도록
- `docs/OUTDOOR_TEST_CHECKLIST_KR.md`: 현장 절차

`scripts/run_live_visualization.sh`는 2026-09-14 모니터 run 재현용 아카이브로만 남겼다.

합성 publisher로 전체 경로를 실행해 패치를 검증했다(capture 21.000 FPS, a₂ 출력
20.99 FPS, NaN 0, ego/roi 기록 확인). record→replay도 합성 스트림으로 확인했다.
best-effort publisher에서 11.76초 동안 248 message를 유실 없이 녹화했고(218 MiB,
약 1.1 GB/분), 그 bag을 `./run.sh replay`로 다시 돌려 21.0 FPS a₂ 출력을 얻었다.

`pgrep -f 'camera_driver_gv'`는 `tail -f ...camera_driver.log`처럼 문자열만 포함한
다른 터미널까지 오탐했다. driver 실행 파일 경로로 매칭하도록 고쳤다.

### 실물 카메라 검증 (2026-09-18, Hikrobot-MV-CS023-10GC-DA8931779)

세션 도중 카메라가 연결되어 전 경로를 실물로 확인했다.

- `./run.sh check`: NIC static, 단일 subnet, ping, aravis discovery 전부 OK
- `./run.sh bench --secs 60 --exposure 2000`: 1280×720 BayerRG8, capture 20.996 FPS,
  a₂ 20.995 FPS, **compute p95 19.77 ms**, header→a₂ p95 31.04 ms, skipped 0,
  NaN 0, resize 없음, **sustained 21 FPS PASS**
- `./run.sh live --exposure 8000`: display + overlay MP4 + full log 동시에 켜고 PASS.
  화면 상단에 `exposure 8000 us | band clipped 0.0% crushed 97.6% | [ / ] exposure
  << UNDEREXPOSED`가 정상 표시됨(어두운 실내라 예상된 결과)
- `[`/`]` 실시간 노출: **동작 확인**. 3000→20000→1000 µs로 바꾸며 mean brightness가
  0.15 → 7.37 → 0.00으로 변했다. 단 `ros2 param get`은 launch 당시 값을 계속 돌려준다
  (set은 적용되는데 readback만 stale). 그래서 코드는 startup에 한 번만 읽고 이후에는
  로컬에서 값을 추적한다. 이걸 polling으로 "고치면" 화면 숫자가 틀리게 된다.
- `--exposure auto` 경로: `AutoExposureTimeLowerLimit/UpperLimit`, `AutoGainUpperLimit`
  이름이 이 카메라에서 accept됐다. unknown-feature 경고 없이 20.997 FPS PASS.

### 미해결로 남긴 것

- `--ego-vx`는 상수다. CAN(can0/can1 존재)에서 실시간 ego 속도를 읽는 경로는 없다.
  카트 시험에서는 문제되지 않지만, 나중에 실제 주행 차량에 올리면 필요하다.
- 실외 주간 조도에서 적정 노출값은 아직 모른다. 현장에서 `./run.sh aim`으로 정한다.
- `kill -9`로 driver를 죽이면 노드가 카메라를 잡은 채 남아 다음 run이 실패한다.
  `./run.sh stop`을 쓴다.
- **커널이 처리량을 좌우한다.** generic(`6.8.0-138-generic`, 옵션 없음)에서는 compute p95 14~20 ms로
  PASS다(2026-09-14 기준선도 generic). GRUB 항목 "A1-BSW 운용"(`6.8.1-1059-realtime` +
  `isolcpus=8-15 idle=poll intel_pstate=disable`)에서는 P-core가 2.1 GHz로 묶여 compute p50 약 49.7 ms,
  약 15~16 FPS로 FAIL이다. 실내·실외 모두 재현된다. 21 FPS가 필요한 run은 generic으로 부팅한다.
  근거: `RESULTS_2026-09-18.md` 11절.
  초기 측정에서 p95 48 ms가 나온 적이 있는데 GPU가 idle clock에서 올라오는 중이었고
  데스크톱 앱이 돌고 있던 탓이다. 인용할 숫자는 항상 `./run.sh bench`로 새로 잰다.

## 10. 주요 문서

- `RESULTS_2026-09-18.md`: 실외 카트 인식 시험 전체 기록(환경·5개 run·원인 분석·다음 할 일)
- `RESULTS_2026-09-14.md`: 실내 모니터 단계 두 실험의 짧은 요약과 바로가기
- `README.md`: portable 설치와 실행
- `docs/REAL_CAMERA_A2_21FPS_EXPERIMENT.md`: 현재 실험 설계
- `docs/SENIOR_CODE_STRUCTURE_KR.md`: 선배 저장소 전체 구조 설명
- `docs/UPSTREAM_MANIFEST.md`: 복사 파일/해시/출처
- `senior_code/COPY_MANIFEST.sha256`: portable source/checkpoint 무결성
- `outputs/real_camera_1280x720_exp15000_fp32_60s/RESULT.md`: 1280×720, 15000 µs 60초 결과 보고서
- `outputs/real_camera_1280x720_exp25000_visual_60s/RESULT.md`: 25000 µs 모니터 재촬영 60초 화면/반응성 보고서
- `docs/OUTDOOR_TEST_CHECKLIST_KR.md`: 실외 카트 인식 시험 현장 절차

## 9-2. 2026-09-18 실외 시험 결과 요약

실외 카트 run 5회를 수행했다. 전체 기록은 `RESULTS_2026-09-18.md`에 있다. 핵심만:

- 21 FPS **실패**(a₂ 16.45 FPS, compute p95 51~57 ms, 카메라 frame 21.7 % drop).
- 그러나 같은 날 같은 코드가 **실내에서는 compute p95 14.4 ms로 PASS**했다. 실외에서
  preprocess(CPU)·backbone(GPU)·a₂(GPU)가 **모두 균일하게 3.6배** 느려졌고, 첫 프레임부터
  평탄하며 중간에 30 ms대로 회복됐다가 되돌아갔다. 발열 누적이 아니라 플랫폼 클럭/전력
  상태 문제로 보인다. **원인 미규명이며 이것이 최우선 과제다.**
- 인식률 7.67 %(실내 모니터 53.67 % 대비). 주원인은 노출이다. 실내값 20000~25000 µs를
  실외에서 그대로 써서 모델 band의 22 %가 백색 포화됐다(`<< OVEREXPOSED` 경고는 정상 표시됨).
- frame drop으로 21-frame history의 고유 source가 중앙값 16/20으로 떨어졌다. 속도 채널
  입력 자체가 왜곡된 상태였다.
- **hardware crop은 해결책이 아니다.** backbone 입력은 이미 250×1280이라 49.5 ms 중
  40.2 ms(backbone+a₂)는 센서 crop과 무관하다. 다만 debayer/전송량 절감 효과는 있고,
  `Offset=(320,475) Size=1280×250`이면 현재와 픽셀 단위로 동일한 geometry다.
- run 5(61분)는 SIGTERM 종료로 `summary.json`과 `a2_overlay.mp4`(moov atom 없음)를 잃었다.
  `frames.jsonl`만 살아남았다. runner에 SIGTERM 핸들러가 필요하다.

## 11. 다음 세션이 지켜야 할 방향

0. **최우선**: 실외/카트에서의 3.6배 플랫폼 저하 원인을 규명한다. `./run.sh bench --secs 60`을
   카트 전원 상태에서 돌리고 nvidia-smi 클럭을 같이 기록한다. bench가 빠르면 display 경합,
   bench도 느리면 전원/클럭 문제다. 이걸 풀기 전에 모델·geometry를 건드리지 않는다.
1. 실외 run 전에 `./run.sh check`를 돌리고, 인용할 처리량 숫자가 필요하면
   `./run.sh bench`로 따로 잰다. 화면/영상/로그를 켠 run의 FPS를 기준값으로 쓰지 않는다.
2. hardware ROI가 확정되지 않았으면 FOV/calibration 없이 임의로 `1280×250` 중앙 crop을
   정확도-valid 경로라고 단정하지 않는다. `geometry_valid_for_accuracy`는 계속 false다.
3. acquisition은 이미 검증됐으므로 측정 초점은 preprocessing, CPU→GPU transfer, backbone,
   21-frame A2, end-to-end output rate다.
4. throughput 21 FPS 통과와 a₂ 실제 정확도를 별도 결과로 보고한다. 주행 run에서 숫자가
   나온다는 사실은 거리·속도 정확도 성공이 아니다.
5. 선배 원본 vendored files는 절대 직접 수정하지 않고 wrapper/adapter만 새 프로젝트
   영역에서 수정한다.
6. 나중에 분석할 여지가 있는 run은 `--bag`으로 원본을 남긴다. `./run.sh replay`로 같은
   입력을 몇 번이든 다시 돌릴 수 있다.
7. 카트가 아니라 실제 주행 차량에 올리는 단계가 오면 그때 `--ego-vx`와 auto exposure
   (`--exposure auto`)를 다시 검토한다. 둘 다 구현은 돼 있다.

## 12. 2026-09-27 20 m 장착·종횡 상대속도 시험 준비

사용자는 실차를 카메라에서 약 20 m에 두고 카메라 높이를 바꾸며 A₂ 거리/속도를
확인할 계획이다. 20 m 한 점에 예측을 맞추는 것만으로 geometry/거리 정확도는 검증되지
않는다. 렌즈 FOV/intrinsic·왜곡, 장착 높이/pitch/yaw, 센서 ROI, 실제 거리 기준점을
기록하고 10/20/30 m 같은 여러 거리에서 비교한다. 현재
`configs/camera_info_1280x720_uncalibrated.yaml`은 실측 calibration이 아니다.
정지 카메라/정지 차량이면 종·횡 상대속도 GT는 0에 가깝다. 움직이는 카메라는 상수
`--ego-vx`로 정확한 동적 속도 GT 시험을 할 수 없으며 실시간 ego 입력이 필요하다.

`run.sh`에 `--obj-thr` 옵션(기본 0.5)과 run_config 기록을 추가했다. threshold는
object-positive bin/lead 선택에만 쓰며 원본 A₂ 회귀 출력은 바꾸지 않는다.
`live_camera_a2_21fps.py` 화면은 선택된 peak bin의 원본 7채널 및 단위 변환값을
모두 표시한다. `frames.jsonl`의 `a2_bins.raw_40x7`은 기존대로 전체 원본을 저장하며
디코드 절대/상대 종·횡 속도도 전부 저장한다. group과 lead 요약에도
`v_lat_rel_mps`를 추가했다. SIGTERM을 정상 중단 요청으로 받아 frame log close,
MP4 release, `summary.json` 저장 경로를 거치게 했다.

오늘 현장 절차는 `docs/OUTDOOR_TEST_CHECKLIST_KR.md` 8절에 기록했다.
`preflight.sh`의 잘못된 기준 커널(realtime)을 실측 21 FPS 통과 설정인
`6.8.0-138-generic`으로 수정하고, A1-BSW realtime 설정의 기존 15–16 FPS
실패를 명시적으로 경고하도록 했다. 문법 검사, portable bundle 검증
(`verified 3 bundled senior files`), 합성 A₂ 7채널 디코드/표시 시험은 통과했다.
2026-09-27 `./run.sh check` 현재 상태는 `6.8.1-1059-realtime`, 카메라 NIC
`enp5s0` DOWN/주소 없음, 카메라 미발견으로 5 FAIL/4 WARN이다. 따라서 이날 새 코드를
연결된 실물 카메라로 재검증하지 못했고 오늘 현장 run의 21 FPS·실차 거리/속도 정확도는
미검증이다. 이전 2026-09-18 실물 ROS2/BayerRG8 검증 사실과 구분한다.

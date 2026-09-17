# A1 실물 카메라 a₂ 실험 — 세션 간 인수인계

마지막 갱신: 2026-09-14 (Asia/Seoul)

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

## 10. 주요 문서

- `RESULTS_2026-09-14.md`: 오늘 최종 보존한 두 실험의 짧은 요약과 바로가기
- `README.md`: portable 설치와 실행
- `docs/REAL_CAMERA_A2_21FPS_EXPERIMENT.md`: 현재 실험 설계
- `docs/SENIOR_CODE_STRUCTURE_KR.md`: 선배 저장소 전체 구조 설명
- `docs/UPSTREAM_MANIFEST.md`: 복사 파일/해시/출처
- `senior_code/COPY_MANIFEST.sha256`: portable source/checkpoint 무결성
- `outputs/real_camera_1280x720_exp15000_fp32_60s/RESULT.md`: 1280×720, 15000 µs 60초 결과 보고서
- `outputs/real_camera_1280x720_exp25000_visual_60s/RESULT.md`: 25000 µs 모니터 재촬영 60초 화면/반응성 보고서

## 11. 다음 세션이 지켜야 할 방향

1. 최종 1280×720 hardware-ROI 60초 baseline은 통과했다. 다음 throughput 단계는 같은 경로의 300초 thermal run이다.
2. hardware ROI가 확정되지 않았으면 FOV/calibration 없이 임의로 `1280×250` 중앙 crop을 정확도-valid 경로라고 단정하지 않는다.
3. acquisition은 이미 검증됐으므로 측정 초점은 preprocessing, CPU→GPU transfer, backbone, 21-frame A2, end-to-end output rate다.
4. throughput 21 FPS 통과와 a₂ 실제 정확도를 별도 결과로 보고한다.
5. 선배 원본 vendored files는 절대 직접 수정하지 않고 wrapper/adapter만 새 프로젝트 영역에서 수정한다.

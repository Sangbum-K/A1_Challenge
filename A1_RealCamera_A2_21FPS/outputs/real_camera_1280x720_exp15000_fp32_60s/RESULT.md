# 실물 카메라 A₂ 21 FPS 검증 결과

실험일: 2026-09-14 (Asia/Seoul)  
결과: **PASS — 실제 `camera_aravis2 → ROS2 → preprocessing → i₁ → 21-frame A₂ → a₂` 경로가 60초 동안 sustained 21 FPS를 달성했다.**

## 1. 검증 범위

이번 실험은 아래 경로의 실행 가능성과 처리량을 검증했다.

```text
Hikrobot MV-CS023-10GC
→ camera_aravis2
→ ROS2 sensor_msgs/Image (BayerRG8)
→ cv_bridge debayer (BGR8)
→ RGB 변환 및 y=235:485 ROI
→ ImageNet normalization
→ segmentation-pretrained EfficientNetV2-S
→ raw i₁ token (1,40,256)
→ current + 과거 20개 timestamp token
→ A2ModelMultiFrame
→ a₂ logits (1,40,7)
```

`f₁`, `f₂`, 차량 제어, pedal/steering 및 MORAI closed-loop는 포함하지 않았다.

## 2. 실험 설정

| 항목 | 설정 |
|---|---|
| 카메라 | Hikrobot MV-CS023-10GC, `192.168.1.51` |
| 센서 전체 크기 | 1920×1200 |
| 카메라 hardware ROI | **1280×720** |
| Hardware offset | `(320,240)` 중앙 crop |
| ROS encoding | `bayer_rggb8` |
| Software resize | **사용 안 함** |
| 모델 ROI | `y=235:485`, 최종 `(1,3,250,1280)` |
| 노출 | **15000 µs (15 ms)**, auto off |
| 카메라/목표 FPS | 21 FPS |
| GEV packet | 1500 bytes |
| 모델 precision | FP32 |
| 장치 | NVIDIA RTX A5000, CUDA 11.8 |
| 측정 시간 | 60초 |
| 화면/영상 저장 | off |

`21 FPS`는 초당 a₂ 출력 개수이고, `21-frame`은 현재 token 1개와 목표 age
0.05–1.00초의 과거 token 20개를 뜻한다.

## 3. 핵심 결과

| 판정 항목 | 기준 | 측정 | 결과 |
|---|---:|---:|---|
| 카메라 수신률 | ≥ 20.58 FPS | **20.999 FPS** | PASS |
| steady a₂ 출력률 | ≥ 20.58 FPS | **20.998 FPS** | PASS |
| compute p95 | ≤ 47.619 ms | **15.214 ms** | PASS |
| NaN/Inf frame | 0 | **0** | PASS |
| 종합 sustained 21 FPS | 모두 충족 | **true** | **PASS** |

- 전체 측정 frame: 1261
- history 1초가 채워진 steady frame: 1240
- frame budget 대비 compute p95 여유: 약 32.41 ms
- 확인된 병목: 없음

## 4. 지연시간

| 구간 | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|
| 전체 compute | 14.393 | **15.214** | 15.697 |
| preprocessing | 3.492 | 3.804 | 3.970 |
| backbone | 6.446 | 7.294 | 7.522 |
| A₂ temporal model | 4.467 | 4.955 | 5.207 |
| A₂ decode/summary | 0.190 | 0.568 | 0.755 |
| ROS callback arrival → a₂ | 15.003 | 15.867 | 16.304 |
| ROS header timestamp → a₂ | 25.489 | **26.383** | 26.943 |

ROS header 기준 지연은 카메라/PTP와 host clock이 동기화됐다는 전제에서 해석한다.

## 5. 21-frame history 품질

| 항목 | 결과 |
|---|---:|
| skipped camera frames | **0** |
| drop ratio | **0.0** |
| 중복 history slot 비율 | **0.0** |
| history age error p50 | 11.918 ms |
| history age error p95 | 21.893 ms |
| history age error max | 23.946 ms |

21 FPS의 frame 간격은 약 47.62 ms이므로, 50 ms 단위 목표 age에 가장 가까운 실제 frame을
선택할 때 최대 약 반 frame의 양자화 오차가 발생할 수 있다. 이번 결과에서는 과거 20개
slot이 모두 서로 다른 source frame으로 구성됐다.

## 6. 모델 및 원본 무결성

- segmentation backbone checkpoint coverage: **100%** (`774` keys, missing 0)
- A₂ checkpoint: `820` keys, required missing 0
- a₂ 출력 shape: `(1,40,7)`
- senior `a2_model.py` SHA-256: `6763cfe442ec7be32eb22d0873b2d8b8577073b3d1f435fb6878d99337c2f5a0`
- segmentation checkpoint SHA-256: `8b9262a5bdd214081b62223f1fee3bb96f50c55b3f3b5bcc3300a5ba30a8fd91`
- A₂ checkpoint SHA-256: `c095b16f15cd73614f2f829bba774095e9c65626b895d4cf1558692953f44003`

## 7. 결론과 제한사항

기술적 처리량 검증은 성공했다. 카메라가 직접 전달한 1280×720 BayerRG8 frame에서 software
resize 없이 preprocessing, backbone/i₁, 21-frame A₂와 a₂ 출력까지 60초 동안 21 FPS를
유지했다.

하지만 이 결과는 **sim-to-real 출력 정확도 성공을 의미하지 않는다.** 현재 1280×720 입력은
1920×1200 센서의 중앙 hardware crop이다. 학습 카메라와 pixel shape은 같지만 수평 FOV,
intrinsic, optical center 및 장착 자세가 같다는 calibration 근거는 아직 없다. 따라서 결과의
`geometry_valid_for_accuracy`는 의도적으로 `false`다. 실제 거리·속도 정확도는 별도 GT
실험으로 평가해야 한다.

## 8. 원본 산출물

- [summary.json](summary.json): 전체 집계 및 PASS/FAIL
- [frames.jsonl](frames.jsonl): frame별 timing/history/A₂ 요약

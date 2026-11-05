# 기존 리포를 활용하는 설계

조사일: 2026-10-12. 기존 rosbag 결과는 자체 지면 제거·군집화·Kalman 추적 + P5 계산이다. 카메라 융합이나 아래 모델의 실행 결과가 아니다. 아래는 도입 후보이며 소스·가중치 설치 및 해당 차량 성능 검증은 미실시다.

## 권장 순서

| 경로 | 기존 구현에서 가져올 부분 | 직접 작성할 부분 | 판단 |
|---|---|---|---|
| A: LiDAR 기준선 | CenterPoint 3D 검출 + Autoware multi_object_tracker | Hesai 입력 변환, INS 시간/좌표 어댑터, 선행차 선택, P5 연결 | 우선 도입 |
| B: A + 현재 A₂ | 동일 검출·추적, 복사한 A₂ 가중치 | 40개 가로 bin과 3D 객체의 영상 투영 구간 대응 | 기존 학습 자산 보존, 보조 증거로 비교 |
| C: BEVFusion | Autoware BEVFusion 또는 NVIDIA CUDA-BEVFusion | 1카메라 구성에 맞춘 학습/내보내기, 센서 보정, 동일 추적/P5 연결 | 추가 이득이 실측될 때 도입 |

### A. LiDAR 검출 + 다중 객체 추적

[Autoware CenterPoint](https://autowarefoundation.github.io/autoware_universe/latest/perception/autoware_lidar_centerpoint/)는 PointCloud2를 받아 학습 기반 3D 객체 검출을 수행하는 TensorRT 구현이다. 자체 군집만으로 차량을 판별하던 부분을 교체한다. **공개 가중치가 Hesai 32채널·현재 트랙에서 잘 동작한다는 보장은 없다.** 좌표, intensity 범위, point feature, voxel 크기, 학습 범위와 실제 차량 크기를 맞춰 평가한다. 검출기 학습 입력이 전체 점군이면 기존 분석용 지면 제거를 무조건 앞에 넣지 않는다.

[Autoware multi_object_tracker](https://autowarefoundation.github.io/autoware_universe/latest/perception/autoware_multi_object_tracker/)를 통해 다중 객체의 ID·운동 상태를 관리한다. detector 출력 속도의 좌표·신뢰도를 확인하고 tracker 속도와 혼용하지 않는다. 선행차 선택은 추적과 별도의 모듈이다. 화면 오른쪽 물체를 지정했던 오프라인 seed는 자동 선행차 판단 규칙이 아니다.

예상 이득은 차량 모양/문맥을 사용한 재검출과 다중 가설 관리다. 실제 점이 거의 없는 구간에서 관측을 복원하거나 동일 ID를 보장하지는 않는다. 정답 라벨과 비교해야 한다.

### B. 현재 A₂를 살리는 약한 후단 융합

A₂ 출력은 bbox가 아니라 가로 40칸의 존재 정보다. 이를 가짜 bbox로 만들어 표준 IoU 융합에 넣지 않는다. LiDAR 3D box를 보정된 영상으로 투영하고, 영상 전처리의 crop/resize를 정확히 반영한 가로 구간과 bin 확률 분포를 비교한다. 같은 bin의 여러 차량은 거리·운동·경로 연속성으로 구분하며 모호하면 카메라 가중치를 낮춘다.

카메라 미탐지/ROI 이탈/미보정은 LiDAR 차량을 삭제할 근거가 아니다. 정상 대응될 때만 차량 분류 및 선행차 후보 점수를 보강한다. 카메라에서 거리·상대속도·자차속도는 가져오지 않는다. 이 어댑터는 현재 A₂ 형식에 맞춘 자체 코드가 필요하다.

### C. 특징 수준 융합 비교

[Autoware BEVFusion](https://autowarefoundation.github.io/autoware_universe/latest/perception/autoware_bevfusion/)은 카메라·LiDAR 특징을 결합하며 LiDAR 단독 모드도 문서화한다. 단독/융합 모델 자산과 설정을 구별해야 한다. 기본 카메라 수는 6개다. **Hikrobot 1대에 맞춰 숫자만 1로 바꾸면 공개 다중 카메라 가중치가 그대로 동작한다고 가정하지 않는다.** 네트워크 입력 형상, 시야, 보정, 학습 분포를 확인해 단일 전방 카메라 학습/미세조정과 ONNX 내보내기가 필요할 수 있다.

[NVIDIA Lidar_AI_Solution](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution)은 CUDA CenterPoint/BEVFusion 등 GPU 구현을 제공한다. [CUDA-BEVFusion 문서](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/blob/master/CUDA-BEVFusion/README.md)의 Orin 성능 수치는 우리 차량의 동시 실행 성능이 아니다. GPU/드라이버/CUDA/TensorRT 호환성부터 별도로 확인한다. 이번 로컬 `nvidia-smi` 조회는 드라이버와 통신하지 못했으므로 현재 GPU 실행 검증은 하지 못했다.

BEVFusion에 A₂의 40칸 결과를 그대로 입력할 수는 없다. 이 경로는 이미지 backbone을 포함한 다른 모델이며 현재 A₂를 그대로 유지하는 B와 구별한다. 카메라 외부·내부 보정과 시간 동기화는 필수다.

[TransFusion](https://github.com/XuyangBai/TransFusion)도 비교 후보지만 연구 코드를 ROS 2 실시간으로 옮기는 비용을 별도 평가한다. CenterPoint/BEVFusion은 활용 가능한 검증된 계열이지 2026년 모든 벤치마크의 최신 1위라는 뜻은 아니다. 최신 논문의 순위만으로 이 센서 구성에 우월하다고 판단하지 않는다.

## 버전과 독립성

전체 Autoware 스택을 실행하지 않고 인지 모듈과 필요한 dependency closure만 별도 workspace에 도입한다. 현재 ROS 2 Humble과 호환되는 릴리스/commit을 실제 빌드로 선택하고, 모델 checksum 및 TensorRT engine 생성 환경을 기록한다. 최신 main이 Humble 호환이라고 가정하지 않는다. 이 단계가 끝나기 전에는 lockfile에 임의 SHA를 쓰지 않는다.

소스는 이 프로젝트 `third_party/`, 모델은 `models/`에 실제 복사한다. 외부 프로젝트 경로를 참조하는 symlink, PYTHONPATH, sibling install overlay를 쓰지 않는다. ROS/CUDA 같은 시스템 설치 의존성과 실행 중 토픽 구독은 코드 경로 의존성과 구별한다.

## 선택을 바꾸는 증거

같은 3~12분 데이터와 분리된 평가 주행에서 A/B/C를 비교한다. 5:59 부근 복구율, 가시 구간 차량 recall, ID 교체, 잘못된 선행차 선택, 유효 거리/속도 비율, P5 변동, end-to-end 지연 p50/p95/p99, 로컬라이제이션 동시 실행 지연을 측정한다. 카메라를 추가해 이득이 없거나 지연이 커지면 A를 유지한다. 기존 prototype와 비교할 때 수동 seed/자동 초기화 조건도 맞춘다.

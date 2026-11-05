# C++ A₂ GPU 실행

CPU 실행본은 `install/`, GPU 실행본은 `install_gpu/`로 분리한다.
GPU 라이브러리는 PyTorch 공식 LibTorch 2.4.1 CUDA 11.8 **cxx11 ABI** 배포본을
`vendor/libtorch/`에 풀어 사용한다. Python wheel의 구 ABI LibTorch를 ROS에 직접 링크하지 않는다.
CUDA toolkit은 현재 `../../cuda-11.8`이며 다른 위치에서는 `CUDA_HOME`으로 지정한다.
드라이버를 변경하지 않고 RTX A5000에서 검증한다.

공식 다운로드:
https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcu118.zip

모델은 `torch 2.4.1+cu118`, `timm 1.0.20` 환경에서 변환한다.
원본 모델이 device를 사용하는 상수 텐서를 생성하므로 **CUDA에서 trace**해야 한다.
CPU trace 파일의 map_location만 바꾸는 것으로는 충분하지 않다.
TF32는 변환과 C++ 실행 양쪽에서 끄고 FP32 수치 비교를 수행한다.
원본 체크포인트 및 reference 파일은 수정하지 않는다.

```bash
# 프로젝트 루트에서, 해당 버전 torch/timm가 설치된 Python으로:
../A1_FSDS_F2_Validation/.venv/bin/python -B cpp/tools/export_a2.py \
  --device cuda --out cpp/models_gpu
bash run.sh build-gpu
bash run.sh camera-gpu

# 실행 중인 노드와 분리한 ROS 도메인에서, 각 시험은 자체 프로세스를 종료함:
ROS_DOMAIN_ID=91 ROS_LOCALHOST_ONLY=1 bash run.sh benchmark-gpu \
  --synthetic --seconds 60 --out outputs/gpu_synthetic
ROS_DOMAIN_ID=91 ROS_LOCALHOST_ONLY=1 bash run.sh benchmark-gpu \
  --seconds 60 --out outputs/gpu_live
```

측정은 모델 warmup 이후 시작하며 초기 5초를 제외한다. `benchmark.json`에 실제 영상 Hz,
A₂ Hz, 입력 header부터 관측까지 지연 p50/p95/p99, history_ready 비율 및 원시 시각을 저장한다.
21Hz 판정은 실측 A₂ ≥20.9Hz(측정 허용차), history_ready ≥99%, 모든 출력 유한값이다.
합성 영상 시험은 계산/ROS 처리량, 실카메라 시험은 센서부터 출력까지의 처리량이다.
둘 다 거리·상대속도 정확도나 차량 제동 검증을 의미하지 않는다.

장착 방향: 사용자가 LiDAR -Y를 차량/카메라 전방이라고 확인했다.
LiDAR +Z가 위라고 가정한 yaw +π/2를 shadow.yaml에 반영했다.
변환식은 base X=-lidar Y, base Y=lidar X, base Z=lidar Z이다.
평행이동·roll/pitch·미세 yaw 및 카메라 extrinsics는 아직 측정값이 없으므로
extrinsics_verified는 false이며 시간 보정 완료로 간주하지 않는다.

실행 시 CUDA toolkit `lib64`도 라이브러리 경로에 추가한다. 공식 LibTorch의
NVRTC 내부 라이브러리 검색을 위해 필요하며 `run.sh camera-gpu/benchmark-gpu`가 설정한다.
21프레임 head의 과도한 초기 specialization을 피하도록 TorchScript profiling executor를
변환 검증과 C++ 양쪽에서 비활성화했다. 일반 graph executor로 동일 가중치를 실행한다.
노드는 CUDA/cuDNN 워밍업을 마친 뒤 영상 구독을 시작한다.

전체 센서와 함께 shadow 실행을 점검하려면 `bash run.sh sensor-check-gpu --seconds 60`을 쓴다.
이는 GPU A₂와 기존 LiDAR/GNSS 드라이버를 실행하고 자체 시작한 프로세스를 종료한다.
IMU 미수신과 보정 미완료에 따른 P5 invalid 상태는 GPU 처리량과 별도로 유지한다.

현재 카메라의 GevTimestampTickFrequency는 100,000,000Hz로 읽혔다.
새 launch는 GevSCPD 2000 ticks(20μs)를 사용한다(기존 관측 400 ticks).
원본 launch는 보존하고 cpp/launch/hikrobot_gpu.launch.py에서 설정한다.
C++ 영상 구독 큐는 4프레임으로 제한하며 OpenCV CPU 스레드는 1개다.

## ROS 영상 전달 설정

원시 Aravis 수신은 20초 동안 421장, 20.997Hz, frame ID 누락/실패/버퍼 underrun 0이었다.
기본 DDS 설정에서는 ROS 전달 중 영상 누락이 있었고, 큰 공유메모리 transport를 지정한
합성 시험에서 630장 입력/630장 출력, 20.999Hz, history_ready 100%로 개선됐다.
`cpp/config/fastdds_large_images.xml`은 SHM segment 64MiB, 단일 메시지 최대 4MiB,
UDP 송수신 버퍼 16MiB를 지정한다. 설정 구조는
[Fast DDS 2.6 공식 문서](https://fast-dds.docs.eprosima.com/en/v2.6.12/fastdds/xml_configuration/transports.html)를 따른다.
OS 전역 설정이나 센서 IP, NIC MTU는 변경하지 않았다.

**영상 발행 드라이버와 A₂ 양쪽에 같은 DDS 설정을 적용해야 한다.** GPU 실행 명령은
FASTRTPS_DEFAULT_PROFILES_FILE을 자동 설정한다(사용자가 이미 지정한 값은 유지).

```bash
# 터미널 1
bash run.sh camera-driver-gpu
# 터미널 2
bash run.sh camera-gpu
```

외부에서 이미 실행 중인 드라이버는 자동으로 설정이 바뀌지 않는다.
시험 도구는 새로 시작한 자식 프로세스에만 설정을 적용하며 다른 프로세스를 종료하지 않는다.

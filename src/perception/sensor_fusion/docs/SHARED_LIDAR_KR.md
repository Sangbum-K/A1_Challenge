> **최신 구현:** source/sink를 독립 C++ 프로세스로 분리하고 익명 Unix socketpair로 연결. 단일 프로세스 두 context 방식은 폐기. [실측 기록](network/camera_restore_20261104/README.md).

> **실제 연결 확인:** [2026-11-04 실행 기록](network/current_before_shared_start/README.md). source context는 공유 발견 허용, destination context는 localhost로 제한하도록 수정. 아래 초기 실센서 미검증 기록은 이전 상태.

# 공용 LiDAR 입력 — 2026-11-04

우리 UI 실행기는 Hesai 드라이버를 시작하거나 UDP2368을 열지 않는다. localization/공용 실행기가 드라이버를 한 번만 실행하며, 우리 프로그램은 C++ 단방향 도메인 브리지로 점군만 구독한다. 팀원 저장소와 NetworkManager 설정은 변경하지 않았다.

```text
LiDAR UDP2368 → localization 소유 Hesai 드라이버
                       ├─ localization 점군 구독
                       └─ /lidar_points (source domain)
                           → C++ lidar_domain_bridge
                           → /lead_fusion/input/lidar_points (domain107)
                           → association → 동일 대상 P5 shadow
```

## 사용

localization 쪽에서 평소 사용하는 LiDAR 실행 명령으로 드라이버를 먼저 실행한다. localization 저장소의 토픽은 `/lidar_points`, 프레임은 `hesai_lidar`다. 소스 도메인 고정값은 저장소에서 확인되지 않아 **ROS 기본값0을 기본 설정**으로 사용한다. 운용 도메인이 다른 경우 아래 환경변수로 반드시 지정한다. 자동으로 여러 도메인을 탐색하거나 팀원 환경을 상속하지 않는다.

```bash
cd /home/ailab/git/sangbum
./run_sensor_fusion.sh lidar-check  # 입력만 검사, 카메라/UI/센서 드라이버 기동 없음
./run_sensor_fusion.sh             # 공용 입력 확인 후 카메라·인지·UI·브리지 실행
./run_sensor_fusion.sh stop        # 우리가 시작한 프로세스만 종료
```

도메인/토픽이 다른 경우(숫자와 토픽은 실제 localization 설정으로 대체):

```bash
A1_LIDAR_SOURCE_DOMAIN=0 A1_LIDAR_SOURCE_TOPIC=/lidar_points ./run_sensor_fusion.sh lidar-check
A1_LIDAR_SOURCE_DOMAIN=0 A1_LIDAR_SOURCE_TOPIC=/lidar_points ./run_sensor_fusion.sh
```

- 이 연결은 **같은 PC**의 localization을 대상으로 한다. localhost 제한 유지. 원격 PC 공유는 별도 설계가 필요하다.
- source와 target 도메인은 달라야 한다. target107은 기존처럼 인지 전용으로 유지한다.
- 시작 전6초 안에 발행자1개 및 타임스탬프가 증가하는 유효한 PointCloud2 3개를 요구한다. 발행자2개 이상, frame불일치, 잘못된 데이터 크기, 중복·역행·0시각, 수신 없음은 통과하지 않는다.
- 검사 성공은 ROS 점군 수신 확인이다. 실제 차량/센서 정확도나 재생 데이터가 아닌 실센서임을 인증하지 않는다.
- 입력 부재 시 원커맨드는 실패 메시지와 함께 종료한다. 드라이버 자동 시작/네트워크 자동 변경 없음.
- 실행 중 입력 단절 또는 다중 발행 시 새 점군 전달을 중단한다. bridge 로그에 WAITING_OR_STALE, 기존 UI에 stale 표시. 오래된 점군을 새 시각으로 재발행하지 않는다.
- source의 데이터와 header/fields/frame/stamp는 수정하지 않는다. 회전·지면제거는 우리 association에서만 수행한다. TF/GNSS/제어 토픽은 브리지하지 않는다.
- best-effort / volatile / depth2 사용: reliable 및 best-effort source에 구독 가능. 전달 지연이 누적되는 큐를 만들지 않는다.
- 우리 종료 동작은 자체 브리지·카메라·인지·UI만 대상으로 한다. 외부 LiDAR 드라이버와 localization은 소유하지 않는다.
- 신규 실행부터 `association.csv`가 자동 저장된다. 기존 geometry/거리별 단계 진단을 실차에서 기록할 수 있다.

## 검증

`cpp/evidence/shared_lidar/`에 로그 보존.

- C++ bridge 빌드 성공.
- 격리 도메인115→116 합성 시험: 점군 바이트/타임스탬프 보존, 단방향 전달, 검사 모드 무발행, 브리지 종료 후 외부 발행자 유지, 다중/미수신 거부, 같은 도메인 거부 통과.
- 현재 실제 source domain0에는 발행자0개. `lidar-check` 및 원커맨드 모두 입력 없음으로 정상 거부. 그 후 UI중지/UDP2368리스너없음 확인.
- **실제 localization을 켠 상태의 도메인 간 연결은 미검증**. 소스가 꺼져 있어 실제 점군 연결을 완료했다고 주장하지 않는다. 우리 프로그램은 중지 상태 유지.

수정: `run_ui.sh`, `run.sh`, 루트 `run_sensor_fusion.sh`, `cpp/src/lidar_domain_bridge.cpp`, `cpp/CMakeLists.txt`, `cpp/tests/lidar_bridge_smoke.py`.
빌드 등록되어 다음 `bash run.sh build-gpu`에 포함된다. 현재 `install_gpu/lib/a1_fusion_cpp/lidar_domain_bridge`에 검증 실행파일 배치 완료.

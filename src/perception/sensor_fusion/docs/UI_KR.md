# 2026-11-04 실험용 UI

C++ association_node가 실제 카메라/군집 위에 1600×860 화면을 생성한다.
카메라 영역은 원래 영상과 bin 투영을 유지한다. 차량 대응 후보는 붉은색,
카메라 bin/FOV는 노란색이다. LiDAR 기본 전방 표시 범위는 70m다.
범위 확대는 표시 축척 변경이며 70m 검출 성능 검증을 의미하지 않는다.
기존 군집 ROI는 전방 1~70m, 좌우 12m다.

실행:

    cd /home/ailab/git/sangbum/A1_Challenge-main/A1_SensorFusion_CPP
    bash run_ui.sh

종료는 터미널 Ctrl+C. 시작한 카메라/LiDAR/A₂/shadow/UI 프로세스만 종료한다.
로그와 최신 화면은 outputs/ui_날짜_시간에 저장한다. 차량 명령은 송신하지 않는다.
카메라/LiDAR 네트워크는 이미 구성된 상태를 사용하며 스크립트가 NIC를 변경하지 않는다.
기존 동일 토픽 발행자가 있으면 중복 기동을 거부한다.
기본 ROS_DOMAIN_ID=0, ROS_LOCALHOST_ONLY=1. 원시 GNSS IMU 드라이버도 같은 도메인/
통신 범위를 사용해야 한다. 이 실행기는 원시 GNSS IMU 드라이버를 만들거나 켜지 않는다.
설치 센서의 /gnss/imu 출력 확보는 별도 미완료 항목이다.

하단 네 카드:
- 속도: /lead_fusion/ego_motion의 IMU 적분 속도, m/s 및 km/h.
- 가속도: 동일 IMU 자세/바이어스 처리에서 중력을 제거한 가속도의 차체 종방향 성분, m/s².
  원시 IMU X값을 차량 가속도로 그대로 표시하지 않는다.
- 대상: 현재 붉은 후보의 LiDAR 원점 기준 중심 전방 X(m), apparent Vx(m/s).
  X는 범퍼 간격이 아니고 Vx는 자차 회전 미보상 값이다.
- P5: /lead_fusion/estimate의 최종 request_mps2(m/s²), raw P5와 gate.
  음수가 제동 요청이며 shadow 표시 전용이다. what-if는 본 계기에서 제외한다.

중요한 연결 한계:
현재 association_node의 붉은 ID와 기존 fusion_node의 P5 tracker는 별도다.
P5 카드에도 Separate P5 tracker / not linked to red ID를 표시한다.
이 UI 추가는 두 추적기를 같은 선행차의 완성된 제동 파이프라인으로 연결한 작업이 아니다.
IMU 수신/정지 초기화/장착 보정, 카메라–LiDAR 실측 보정과 시간 검증,
동일 대상의 보상된 상대속도 및 P5 연결 검증이 완료돼야 한다.
그 전까지 임의로 valid 또는 보정 플래그를 true로 바꾸지 않는다.

유효하지 않은 값과 입력 0.25초 초과는 --로 표시한다. 브레이크 valid_until도 확인한다.
LiDAR/association 입력 중단 0.3초 후에는 전체 화면에 STALE 배너를 표시하고
이전 대상과 P5 수치를 유효하게 유지하지 않는다.
IMU 적분에는 기존 정지 확인/보정 요구 및 10초 드리프트 제한을 그대로 적용한다.

검증:
- core_tests/association_tests 통과.
- 합성 IMU: 속도 0.995m/s, 중력 제거 종가속도 1m/s² 전달 및 stale 거부 통과.
- 합성 UI: 1600×860 출력, LiDAR 중단 STALE 배너, 단일/다중/시간 만료 판정 통과.
- 실센서 화면 및 run_ui.sh의 Ctrl+C 자식 프로세스 종료 확인.
- outputs/ui_live/latest.jpg는 실제 화면이다. IMU/P5 유효값은 없어 --로 표시됐다.

모델 출처:
기존 a2_mf20_v2s_n20.pt와 seg_pretrain_fx640_v3_backbone.pt를 재학습 없이
CUDA TorchScript로 변환한 cpp/models_gpu를 사용한다.
A₂ 820키, backbone 774키 누락 없음; 출처 SHA256은 models_gpu/export.json 참조.
카메라는 존재/bin 위치의 조건, LiDAR는 그 영역의 군집/metric 공급 역할이다.
정확히 한 군집만 두 조건을 만족할 때 후보가 되며, 한 센서의 단독 확정으로 대체하지 않는다.

## 통합 원커맨드 및 방향 최종 결정
작업 루트 /home/ailab/git/sangbum에서 ./run_sensor_fusion.sh 실행.
./run_sensor_fusion.sh status로 실행 상태/로그 경로 확인,
./run_sensor_fusion.sh stop 또는 실행 터미널 Ctrl+C로 종료.
동시 중복 실행은 파일 잠금으로 차단한다.

사용자가 오른쪽 45도 변경을 취소하고 기존 방향이 맞다고 재확인했다.
최종 lidar_roi_center_deg=0.0, 원래 -LiDAR Y 전방 기준이다.
ROI heading 조절 기능은 남지만 기본값은 0이다. 카메라 외부 보정값은 변경하지 않았다.
흰색 선택 반원/점군 선택에 동일 파라미터를 사용하며 회색 주변 점군을 함께 표시한다.

최신 변경: 원커맨드는 도메인107/localhost로 고정하고 MP4를 자동 녹화한다. 이전 도메인0 안내는 더 이상 적용하지 않는다. 상세: ISOLATION_RECORDING_KR.md.

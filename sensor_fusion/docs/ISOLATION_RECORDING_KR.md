# 실행 격리 / MP4 녹화 점검 (2026-11-03)

최종 원커맨드: 작업 루트에서 ./run_sensor_fusion.sh
상태: ./run_sensor_fusion.sh status
종료/MP4 마무리: ./run_sensor_fusion.sh stop 또는 실행 터미널 Ctrl+C.

## 점검 범위와 발견
/home/ailab/git의 접근 가능한 소스/설정/실행 스크립트와 현재 프로세스/UDP 포트,
실행 후 ROS 그래프를 점검했다. 압축 백업, bag 원시데이터, 빌드 결과와 vendored
라이브러리 전체의 의미 분석은 포함하지 않는다. 원본 팀원 파일은 수정하지 않았다.
근거는 cpp/evidence/isolation에 저장했다.

- A1_localization: /lidar_points, hesai_lidar, base_link를 사용.
  MOLA는 map/odom 계열 TF 소유 관계가 있고 GNSS bringup과 결합한다.
- A1-BSW: /control/target, /control/adapter/*, /control/status/*,
  /gsof_client/odom 사용. rehearsal domain 73.
- main_pc_work_20260918: 도메인 71,72,74, 시험 73/75/76, MORAI 82.
- Seunghwan 및 시뮬레이터/샘플 소스 검색에서는 해당 ROS API/도메인 문자열을
  찾지 못했다. 이는 비ROS 프로토콜이나 모든 실행 경우의 무충돌 인증은 아니다.
- 과거 sangbum 카메라 프로젝트는 같은 카메라 토픽/실행파일을 사용하며
  광역 pkill 명령이 있다. 그 폴더의 stop 명령은 우리 프로세스를 종료할 수 있다.
  ROS 도메인은 OS 수준의 kill이나 센서 독점권을 격리하는 기능이 아니다.

## 적용한 격리
run_ui.sh는 ROS_DOMAIN_ID=107, ROS_LOCALHOST_ONLY=1, rmw_fastrtps_cpp를 고정한다.
상속된 팀원 AMENT/Python/library 경로와 ROS_NAMESPACE/discovery server 설정을
제거하고 /opt/ros/humble 및 우리 설치본만 source한다.
전용 FastDDS 프로필의 UDP 인터페이스는 127.0.0.1로 제한하며 SHM을 함께 사용한다.
일반 run.sh 개발/시험 모드는 별도이며, 격리 보장은 이 원커맨드 경로에 적용된다.

토픽 이름은 기존 인터페이스를 유지하지만 다른 도메인과 discovery/메시지를 공유하지 않는다.
우리 노드는 /tf, /tf_static, /control/target 또는 차량 CAN 명령을 발행하지 않는다.
실행 전 도메인107에 다른 application node가 있으면 거부한다.
파일 잠금으로 중복 원커맨드를 막고, 다른 카메라 driver 프로세스 또는 UDP2368
사용이 감지되면 센서 중복 점유를 거부한다. 원격 호스트의 카메라 제어권까지
이 검사로 보장하지는 못하며 장치 드라이버 오류로 확인해야 한다.
팀원 노드와 동시에 같은 물리 카메라/동일 UDP 수신기를 독립 실행할 수 있다는 뜻은 아니다.

GNSS IMU를 도메인0에서 자동 브리지하지 않는다. 이후 원시 IMU 제공자는 107에
명시적으로 연결하거나 검토된 단방향 입력 브리지를 따로 설계해야 한다.
원시 IMU 부재와 동일 트랙 P5 연결 미완료는 이번 격리 작업으로 해결되지 않았다.

## 실행 검증
domain107.json: camera_driver_gv, hesai_ros_driver_node, a1_a2_camera,
a1_fusion, a1_association의 application node 5개 확인(daemon/audit 제외).
domain0.json: 우리 application publisher 및 수신 메시지 없음.
녹화 중 8초 측정 A2 21.04Hz, association 9.87Hz, shadow estimate 10.00Hz.
estimate의 발행 속도는 유효 P5 값 또는 제동 검증을 뜻하지 않는다.

## MP4
원커맨드는 outputs/ui_날짜_시간/recordings/ui_00001.mp4부터 자동 녹화한다.
UI 전체 1600x860, MPEG-4 Part2/mp4v 컨테이너 MP4, 10fps, 파일당 60초.
최신 화면 하나만 공유하는 별도 C++ 스레드로 기록하며 무제한 프레임 큐가 없다.
입력 중단 시 STALE 화면도 기록한다. frames.csv에 프레임별 실제시각/UI시각을 저장한다.
REC 표시가 나오며 저장 실패/여유 공간128MiB 미만이면 RECORDING ERROR가 뜬다.

완료된 파일은 바로 재생할 수 있다. 현재 쓰는 파일은 다음 구간으로 넘어가거나
정상 종료해야 MP4 메타데이터가 마무리된다. 강제 KILL/전원 차단 시 마지막 미완료
구간은 손상될 수 있다. 1분 분할은 이전 완료 구간을 보호하기 위한 것이다.
자동 삭제하지 않으므로 장시간 운용의 디스크 용량 관리는 별도로 필요하다.

실제 ui_20261103_234933/recordings/ui_00001.mp4:
ffprobe 1600x860,10fps,600frames,60.000s 확인, 전체 ffmpeg decode 오류 없음.
합성 종료 시험에서도 destructor 후 파일 재개방/6프레임 디코딩 통과.

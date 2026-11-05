# 현재 센서 설정 보존 — 2026-11-04

사용자 요청: 현재 LiDAR/GNSS 환경에 간섭하지 않고 우리 노드 실행. 이 폴더는 실행 직전 상태 스냅샷이다.

- localization PID143953, Hesai PID143960, GNSS PID143962. source domain0 / ROS_LOCALHOST_ONLY=0.
- LiDAR enp0s31f6,PC192.168.1.102/24,센서192.168.1.201,UDP2368. localization이 소유.
- GNSS enp4s0,PC192.168.53.20/24,센서192.168.53.100:5017. localization이 소유.
- enp5s0는 현재 `Wired connection 1`,192.168.53.20/24. 예전 카메라169.254.222.1 프로필이 활성 상태가 아님. 카메라169.254.222.118 경로가 USB테더링으로 향함. GNSS와 같은 로컬주소가 두 인터페이스에 있지만 사용자 요청에 따라 변경하지 않음.
- LiDAR/GNSS 네트워크·장치 설정·팀원 파일은 수정하지 않음. 외부 프로세스 재시작하지 않음.
- 현재 실행 선택: A1_CAMERA_ENABLED=0, A1_LIDAR_SOURCE_DOMAIN=0. 우리 association·shadow·UI/MP4와 단방향 bridge만 시작. 카메라/A2는 시작하지 않음.
- source bridge는 외부공유 발견 허용, destination context는 localhost로 제한. 원본 점군을 domain107 전용 입력으로만 전달.
- GNSS 토픽/TF/제어 메시지는 브리지하지 않음. 우리 UI의 IMU 속도나 P5가 유효해졌다는 뜻은 아님.

`external_config_hashes.json`은 외부 드라이버 설정 원본의 비교용 SHA256. `process_*.json`은 ROS 관련 환경만 저장(전체 환경/비밀정보 수집 안 함). `*_params_*.yaml`은 당시 launch 파라미터 사본. nmcli 프로필에 비밀번호를 포함하지 않음.

실행 명령:
```bash
A1_CAMERA_ENABLED=0 A1_LIDAR_SOURCE_DOMAIN=0 ./run_sensor_fusion.sh
```

우리 노드만 종료: `./run_sensor_fusion.sh stop`. 외부 localization/Hesai/GNSS는 종료 대상이 아니다.

## 실행 후 확인

- 우리 런처 PID148161, 출력 `outputs/ui_20261104_112823`, UI/MP4 실행 중.
- 8.04초 수신: 공용 점군80개, association79개. 마지막 처리50.33ms, 상태no_image. 카메라 미기동 상태와 일치.
- 기존 localization/Hesai/GNSS PID143953/143960/143962 유지. UDP2368은 기존 Hesai PID143960만 점유.
- 외부 LiDAR/GNSS 설정 SHA256과4개 네트워크 프로필 비교 모두 동일. `unchanged_check.json` 참조.
- source localhost0 → destination localhost1 교차 도메인 합성 회귀시험 통과. 양쪽 컨텍스트의 localhost 범위를 개별 지정하며, source를 통해 제어·TF·GNSS 토픽을 발행하지 않는다.
- 보존 자료: `live_check.json`, `our_run.env`, association/shadow 설정 사본, 빌드·시험 로그.

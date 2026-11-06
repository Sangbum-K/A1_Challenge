# 카메라 복구 / 기존 LiDAR·GNSS 보존 — 2026-11-04

사용자 요청으로 카메라 연결과 전체 인지 실행을 복구했다.

## 변경

현재 enp5s0의 Wired connection 1은 ipv4.method=shared이며 추가 보조주소가 유지되지 않았다. 보조주소·경로 추가를 시도했지만 동작하지 않아 해당 프로필 변경을 원복했고, 기존 공유 설정을 유지하는 MACVLAN 인터페이스를 추가했다.

- 신규 NetworkManager 프로필: `a1-camera-isolated`, UUID `90a5477c-eb38-484f-b7d7-f81eacc8e1e8`.
- 인터페이스 `a1cam0`, parent `enp5s0`, macvlan bridge mode.
- PC IP `169.254.222.1/16`, 카메라 전용 route `169.254.222.118/32`, metric40.
- 기본 인터넷 경로 없음, IPv6 disabled. 프로필은 영구 저장됨. 실제 재부팅 시험은 하지 않음.
- 카메라 ping2/2,0% loss. 카메라 경로는 a1cam0.

## 보존 확인

`Wired connection 1`, `a1-sensor-lidar`, `a1-sensor-gnss`, `Wired connection 4`의 connection/ipv4/ipv6 설정은 before/after 완전 동일.
LiDAR192.168.1.201은 enp0s31f6, GNSS192.168.53.100은 enp4s0, USB인터넷 기본 경로 유지.
외부 Hesai PID143960,GNSS PID143962 유지. UDP2368은 기존 Hesai 한 프로세스만 점유.
외부 localization·GNSS/LiDAR 드라이버를 재시작하거나 설정 파일을 수정하지 않았다.

## 실행

우리 LiDAR-only UI만 정상 종료하고 다음 명령으로 재시작했다.

```bash
A1_CAMERA_ENABLED=1 A1_LIDAR_SOURCE_DOMAIN=0 ./run_sensor_fusion.sh
```

출력: `outputs/ui_20261104_113216`, 런처 PID152604. 카메라/GPU A₂/공용 LiDAR 브리지/association/shadow/UI/MP4 실행.
카메라 설정: ExposureAuto Off, ExposureTime3000μs,GainAuto Off,Gain0dB. 드라이버 로그에서 두 Auto Off와 ExposureTime3000 읽기 확인. Gain0은 실행 인자 설정이며 해당 로그 요약에는 숫자 게인이 출력되지 않음.

`live_check.json`은 실제 ROS 이미지·A₂·LiDAR·association 수신 수치. 설정 사본과 경로 before/after를 이 폴더에 보존했다. 이전 LiDAR-only 실행/카메라 경로 없음 기록은 과거 상태다.

## 후속 확인 / 현재 상태

최종 런 `outputs/ui_20261104_113405`. Humble의 localhost 환경변수 재해석 때문에 bridge 대상 context 초기화 직전에 localhost=1을 명시하도록 보완.128,000점 메시지를 포함한 교차 도메인 회귀시험 통과.
11:33:13에 외부 localization 두 번째 실행(PPID153954, Hesai PID153960)이 추가됨. 기존 PID143960과 함께 UDP2368 바인딩 및 /lidar_points 발행자2개 확인. 우리 bridge는 다중 입력을 거부하고 대기 중. 외부 실행은 사용자가 다른 것은 건드리지 말라고 요청하여 종료하지 않았으며 확인 요청 중.
카메라/A₂ 수신 확인: live_check_final.json. 카메라 밝기 및 전체 융합 화면 검증은 LiDAR 중복 정리 이후 필요. 기존 네트워크4개 프로필은 최종 재확인에서도 변경 전과 동일.

## 최종 연결 복구 (후속 사용자 요청)

현재 런: outputs/ui_20261104_120315. 외부 localization PID175899/Hesai175910/GNSS175912를 유지하고 우리 런처만 재시작. 현재 외부 LiDAR 발행자는1개. 사용자 확인 없이 외부 프로세스 종료하지 않음.
동일 프로세스의 ROS localhost 범위 혼합을 피하기 위해 C++ bridge를 fork+exec한 source/sink 별도 프로세스로 분리했다. source는domain0/shared discovery 구독만, sink는domain107/localhost+isolated XML 발행만. 두 프로세스는 익명 Unix socketpair로 직렬화 원본점군 전달. 센서UDP포트와 별개이며 외부제어 발행 없음. 원본타임스탬프·필드·바이트 보존.
분리 프로세스/128,000점 교차도메인 테스트, 단방향, 부모종료후외부발행유지, 미수신/다중/loop거부 통과. 테스트도 source와observer를 독립 프로세스로 분리해 기존 두 context 시험의 한계를 제거했다.
10.0085초 실제 수신: camera210/A₂211/LiDAR58/association59. GPU history_ready=true, 인지처리 약71ms. 실제 통합 화면에서 카메라+BEV 확인. LiDAR 입력 간격이 불규칙해 간헐 STALE 존재. 안정10Hz/검출정확도 완료로 해석하지 않는다. 원인분석 없이 stale기준을 완화하지 않았다.
이번 턴에는 네트워크 설정을 추가 변경하지 않았다. 기존4개 프로필은 NetworkManager가 자동 갱신하는 connection.timestamp를 제외한 모든 필드가 이전과 동일. 카메라a1cam0:169.254.222.1→169.254.222.118,LiDARenp0s31f6:192.168.1.102→192.168.1.201,GNSSenp4s0:192.168.53.20→192.168.53.100 경로 유지. 수동 노출3000μs/게인0dB 유지.

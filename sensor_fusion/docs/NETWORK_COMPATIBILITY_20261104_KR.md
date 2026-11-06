> **후속 수정 완료:** [공용 LiDAR 연결](SHARED_LIDAR_KR.md). 우리 UI의 UDP드라이버 자동 기동 제거. 아래 충돌 설명은 수정 전 점검 기록이다.

# NetworkManager 변경의 localization / GNSS 영향 점검 — 2026-11-04

센서 드라이버와 localization을 다시 실행하지 않고 현재 OS 경로, 팀원 소스 설정, 짧은 소켓 수신을 대조했다. 이번 점검에서 네트워크 설정·팀원 파일은 변경하지 않았다. 점검 소켓은 모두 닫았다.

## 결론

현재 배선에서 GNSS TCP 연결과 LiDAR UDP 수신은 정상이다. 다만 NetworkManager 설정은 호스트 전체에 적용되므로 프로젝트에만 격리되는 변경은 아니다. GNSS 항법 데이터 수신 및 localization 정상 동작 전체를 검증한 것은 아니다.

| 대상 | 현재 설정 / 실제 확인 | 판단 |
|---|---|---|
| GNSS | enp4s0:192.168.53.20 →192.168.53.100 TCP5017 연결 성공 | a1_gnss_bringup/config/gsof_client.yaml의 목적지와 일치. 연결 후2초 수신 없음. 네트워크 연결 성공과 GSOF 출력은 별개 |
| LiDAR | enp0s31f6:192.168.1.102,192.168.1.201:10000→255.255.255.255:2368,1080바이트 수신 | 현재 브로드캐스트 전송이므로 예전 호스트.100에서 현재.102로 바뀐 것이 수신을 막지 않음. 향후 유니캐스트 전환 시 목적지 주소 확인 필요 |
| 인터넷 | USB테더링172.20.10.1 기본 경로 유지 | 센서 프로필은 기본 게이트웨이를 잡지 않음 |
| localization LiDAR 설정 | A1_localization/src/HesaiLidar_ROS_2.0/config/config.yaml:device_ip_address192.168.1.201,udp_port2368,host_ip_address빈 문자열 | 현재 호스트.102와 직접 충돌하는 고정 호스트 바인딩은 해당 설정에 없음 |

## 발견된 주의점

1. `/home/ailab/git/A1_localization/scripts/setup_lidar_host.sh`는 기본 호스트192.168.1.100/24,프로필a1-lidar를 설정하고 활성화한다. 인터페이스는 사용자가 인자로 지정한다. 현재 GNSS 포트인enp4s0에 실행하면 GNSS 주소를 바꿀 수 있고, LiDAR포트에 실행하면 현재.102를.100으로 바꾼다. 현재 배선에서는 이 초기 설정 스크립트를 다시 실행할 필요가 없다.
2. 센서 프로필 autoconnect priority100은 원래 프로필보다 우선한다. 과거 다른 용도의 케이블을 같은 포트에 연결할 때는 프로필 매핑을 다시 확인해야 한다. 원래 프로필은 보존되어 있다.
3. ROS_DOMAIN_ID가 달라도 같은 호스트의 UDP2368은 공유된다. 두 Hesai 드라이버를 동시에 실행하면 포트 점유 실패 또는 수신 분배 문제가 생길 수 있다. 운용 시 드라이버 하나의 점군을 필요한 노드에 전달하는 구조가 필요하다. 현재 프로젝트 드라이버는 종료되어 있고 점검 전2368리스너도 없었다.
4. 프로젝트 런처의 domain107/localhost 격리는 자식 프로세스 설정이다. NetworkManager 변경으로 GNSS ROS 토픽이나 localization 토픽 이름을 바꾼 것은 아니다. 다른 도메인의 localization에서 우리 점군을 자동으로 받을 수 있다는 뜻도 아니다.
5. GNSS TCP연결은 되지만2초 동안 항법 데이터가 없었다. 기존에도 보고된 문제이며 이번 점검만으로 원인을 출력 설정으로 확정하지 않는다. GSOF49/50 출력 및 실제ROS 수신은 미검증이다.

## 점검 범위와 한계

A1_localization 소스의 GNSS/LiDAR 설정·launch 및 네트워크 스크립트, main_pc_work_20260918 /Seunghwan의 관련IP/인터페이스/ROS환경 참조를 검색했다. 현재 실행되는 localization 런치 인자나 설치본 전체, GNSS출력설정, 지도정합·위치정확도는 실행검증하지 않았다. 외부 팀원PC가 이 호스트의 과거IP로 접속하는 구성도 이번 로컬 검사만으로 확정할 수 없다.

현재 포트 매핑: 카메라enp5s0,LiDARenp0s31f6,GNSSenp4s0. [네트워크 프로필 원본 기록](NETWORK_KR.md).

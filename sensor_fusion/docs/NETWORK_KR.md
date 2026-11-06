> **현재 카메라 경로 갱신:** [2026-11-04 카메라 복구 기록](network/camera_restore_20261104/README.md). enp5s0 공유 프로필 유지, MACVLAN a1cam0로 카메라 연결. 아래 초기 프로필 표는 이전 설정.

> [localization·GNSS 호환성 점검 (2026-11-04)](NETWORK_COMPATIBILITY_20261104_KR.md): 현재 TCP/UDP 연결 확인. 기존 setup_lidar_host.sh 재실행과 이중 드라이버 주의.

# 센서 전용 네트워크 — 2026-11-04 적용

NetworkManager에 영구 저장된 센서 전용 프로필을 생성하고 활성화했다.
기존 연결 프로필은 삭제하거나 수정하지 않았으며 변경 전 값은 `network/20261104_before/`에 저장했다.
적용 후 프로필/라우팅은 `network/20261104_after/`에 기록했다.

| 센서 | 프로필 | PC 포트 | PC 고정 IP | 센서 IP |
|---|---|---|---|---|
| 카메라 | a1-sensor-camera | enp5s0 | 169.254.222.1/16 | 169.254.222.118 |
| LiDAR | a1-sensor-lidar | enp0s31f6 | 192.168.1.102/24 | 192.168.1.201 |
| GNSS | a1-sensor-gnss | enp4s0 | 192.168.53.20/24 | 192.168.53.100 |

세 프로필은 autoconnect=yes, priority=100, IPv4 manual, never-default=yes,
DNS/게이트웨이 없음, IPv6 disabled다. 각 센서 IP에는 /32 직접 경로(metric40)를 저장했다.
따라서 인터넷 테더링/Wi-Fi를 바꾸어도 센서 설정을 매번 바꿀 필요가 없다.
센서 IP 또는 케이블 포트를 바꿀 때는 이 매핑을 함께 수정해야 한다.
우선순위가 높은 새 프로필이 자동 연결 대상으로 선택된다. 예전 프로필을 수동 선택하면 이전 설정이 적용되므로 피한다.

기존 enp4s0와 enp0s31f6의 192.168.1.0/24 중복은 새 활성 프로필에서 제거했다.
카메라 포트의 예전 laptop-share 기본 인터넷 경로도 새 프로필에는 없다.
인터넷은 기존 USB 테더링의 기본 경로를 유지했다. 다른 인터넷 프로필은 수정하지 않았다.

## 확인 결과

- 세 센서 모두 ping 2/2 응답, 손실0%.
- camera → enp5s0, LiDAR → enp0s31f6, GNSS → enp4s0 경로 확인.
- 센서 설정은 `nmcli connection add`로 저장한 연결 프로필이며 임시 `device modify`가 아니다.
- 재부팅을 실제 수행한 검증은 하지 않았다.
- 이번 작업은 네트워크 설정이다. 센서 드라이버/UI는 재시작하지 않았고 영상·점군·원시 IMU 데이터 수신을 새로 검증한 것은 아니다.

## 실행 / 점검

```bash
cd /home/ailab/git/sangbum
./run_sensor_fusion.sh
```

```bash
nmcli -f NAME,DEVICE connection show --active
ip route get 169.254.222.118
ip route get 192.168.1.201
ip route get 192.168.53.100
```

자동 연결이 선택되지 않았을 때 수동 활성화:

```bash
nmcli connection up a1-sensor-camera
nmcli connection up a1-sensor-lidar
nmcli connection up a1-sensor-gnss
```

## 이전 설정 복구 (필요할 때만)

아래는 원래의 네트워크 용도로 되돌리는 명령이다. 현재 센서 배선에는 이전 설정이 맞지 않는다.
새 프로필 자동 연결을 끄고 보존된 원래 UUID를 활성화한다. 인터넷 연결 프로필은 건드리지 않는다.

```bash
nmcli connection modify a1-sensor-camera connection.autoconnect no
nmcli connection modify a1-sensor-lidar connection.autoconnect no
nmcli connection modify a1-sensor-gnss connection.autoconnect no
nmcli connection up ba0a3e01-5064-44ef-8bba-253a3c8385ed
nmcli connection up 24ce62a5-0771-463f-a383-be33ff2a925a
nmcli connection up 57509148-9a34-40e1-aa70-69bd865e3885
```

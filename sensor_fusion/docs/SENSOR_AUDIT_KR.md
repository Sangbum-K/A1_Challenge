# 현재 배선 및 센서 수신 점검 — 2026-11-03

사용자가 센서를 연결한 뒤 실제 장치 검색과 인터페이스별 통신으로 확인했다.
NetworkManager 프로필 이름 `a1-lidar`, `camlink-onboard`, `laptop-share`는 과거 이름이며
현재 연결된 센서를 뜻하지 않는다.

| 장치 | 현재 NIC | 실제 주소 | 확인 결과 |
|---|---|---|---|
| Hikrobot MV-CS023-10GC, DA8931780 | enp5s0, 1Gb/s | 169.254.222.118 | GigE discovery·모델/serial 읽기·ping·ROS 영상 |
| Hesai LiDAR | enp0s31f6, 100Mb/s | 192.168.1.201 | 해당 NIC ping·드라이버 점군 수신 |
| Trimble 계열 GNSS/INS 장치 | enp4s0, 100Mb/s | 192.168.53.100 | ping·HTTP·TCP 5017 접속, 항법/IMU 데이터 미수신 |

과거 카메라 DA8931779/192.168.1.51과 현재 카메라는 serial이 다르다.
과거 카메라/LiDAR 포트 배치도 현재와 다르므로 저장된 장착 보정이 맞는다고 가정하지 않았다.

## 호스트 통신 경로

기존 주소를 유지하며 현재 장치를 확인하기 위해 아래 **임시** 변경을 적용했다.
`nmcli device modify`이므로 저장된 연결 프로필은 수정하지 않았다. 재부팅/재연결 시
사라질 수 있다. 장치 자체 IP나 GNSS 출력 설정은 변경하지 않았다.

```bash
nmcli device modify enp5s0 +ipv4.addresses 169.254.222.1/16
nmcli device modify enp4s0 +ipv4.addresses 192.168.53.20/24
nmcli device modify enp0s31f6 +ipv4.routes '192.168.1.201/32 0.0.0.0 40'
```

현재 각 주소의 `ip route get`이 위 NIC를 가리킴을 확인했다.
실제 연결 값은 `cpp/config/current_cabling.yaml`에 별도로 저장했다.

## 수신 실측

`outputs/live_sensor_audit2/sensors.json`의 약 25초 시험:

- 카메라 454개, 약 20.37Hz. 1280×720 BayerRGGB8, frame `hikrobot_camera_vis`.
  header 대비 수신 지연 중앙값 약 12.1ms. 이것은 카메라 수신률이며 A₂ 추론 FPS가 아니다.
- LiDAR 227개, 약 9.34Hz. scan당 128,000 slots, point_step 26, x/y/z/intensity/ring/timestamp.
  자체 header는 1,589,933,395초 부근이며 호스트와 약 203,779,843초 차이가 난다.
  따라서 이 자체시각을 동기화된 시각으로 쓸 수 없다.
- GNSS TCP 5017은 접속됐지만 직접 socket 수신 3초 및 드라이버 시험에서 GSOF 데이터를
  받지 못했다. 항법 속도 토픽과 내장 raw IMU 토픽도 확인되지 않았다.
  출력 설정 화면 GET은 HTTP 401이며 사용자가 지금 설정 확인은 어렵다고 했다.
  로그인 없이 설정 우회나 변경은 하지 않았다.

`cpp/config/lidar_host_time.yaml`은 복사본 원본을 보존하고 별도로 만든 호스트 수신시각
시험 설정(`use_timestamp_type: 1`)이다. 수신시각 사용이 노출/laser/IMU의 정밀 시각 동기화
완료를 뜻하지 않는다. 새 노드의 `clock_verified`는 계속 false다.

## IMU만 사용하라는 최신 요청

GNSS 항법 위치/속도는 사용하지 않는 `ego_source: imu`를 기본값으로 만들었다.
하지만 GNSS 수신기의 내장 IMU 원시 데이터도 실제 전송 경로가 필요하다.
현재 `/gnss/imu`는 코드가 요구하는 계약이며 실측 publisher가 아니다.
초기속도 0을 임의로 채우거나 LiDAR IMU로 바꿔 쓰지 않았다.

원시 IMU가 들어오면 C++에서 정지 초기화, 중력·바이어스 보정, 자세·속도 적분을 수행한다.
이는 제한된 시간의 추정이며 지속적인 절대속도 측정은 아니다. 속도 정답·외부 보정·센서 시간
검증이 없으므로 실제 제동값의 타당성 및 21 FPS 성능은 미검증이다.

## 검증 결과

새 폴더의 드라이버와 C++를 동시 실행한 최종 15초 시험은
`outputs/sensor_check/sensors.json`, 요약은 `cpp/evidence/live_sensor_summary.json`에 있다.
카메라 244개(19.18Hz), LiDAR 117개(8.11Hz), C++ A₂ 19개(1.55Hz),
P5 진단 메시지 119개였다. CPU 추론 지연 중앙값 약 670ms, history_ready=false다.
GPU가 아닌 현재 CPU LibTorch에서의 결과이며 21Hz 요구를 충족하지 못한다.
LiDAR 호스트시각 설정에서는 header 지연 중앙값 약 113ms로 바뀌었다.
IMU/INS 입력은 0개, P5는 `extrinsics_unverified`로 invalid이며 요청을 만들어내지 않았다.
시험이 시작한 드라이버와 C++ 프로세스는 모두 종료했다. 데이터가 없는 GNSS 드라이버는
정상 종료 대기 4초 후 종료되지 않아 해당 자식만 강제 종료했다.

- 기존 선배 원본 4개 해시 일치, Python 기존 시험 + C++ 차등 시험 38 passed.
- C++ P5 203사례 기준 비교 통과.
- 기존 5,401행 표 중 유효 847행 C++ 재계산 최대 오차 3.55e-15.
- C++ 카메라 전처리/추론과 Python 동일 영상 A₂ 최대 절대오차 0.
- 합성 ROS 융합: ego=10, lead=5, relative=-5m/s, request=-3m/s², 입력 소실 invalid.
- 합성 ROS IMU: 정지 보정 후 1m/s² 적분 결과 0.995m/s, 입력 지연 invalid.
- 새 폴더의 복사본 777파일, 66,938,864,883bytes SHA-256 일치(`COPY_MANIFEST.json`).

숫자 일치, 실제 센서 데이터 수신, 인지 정확도, 실차 제동 성능은 서로 다른 검증이다.

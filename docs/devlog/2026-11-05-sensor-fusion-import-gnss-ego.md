# sensor_fusion 가져오기와 GNSS ego 전환

- 시각: 2026-11-05 10:19, 11:02
- 커밋: `8192773` Add A1_Total sensor_fusion, `a353f97` Improve sensor fusion tracking and GNSS shadow input (작성자 Codex)
- 근거: 두 커밋의 diff, 커밋 안의 README와 PROJECT_CONTEXT. 작업 당시 대화 기록은 이 Mac에 없다.

## 요약

차량 PC의 통합 워크스페이스 A1_Total에 있던 센서 퓨전을 이 저장소로 가져왔다.
이어서 자차 입력을 GNSS odom으로 바꾸고, 앞차의 보이는 면이 바뀔 때 트랙이 끊기는 문제를 완화했다.
후면→측면 연결 로직은 다음 날 [추적 교체](2026-11-06-tracking-implementation.md)에서 다른 방식으로 대체됐다.

## 가져온 내용 (`8192773`)

파일 413개를 `src/perception/sensor_fusion/`에 추가했다. `cpp/`, `docs/`, `reference/`, `third_party/`, 실행 스크립트로 구성된다.
A1_Total 안의 위치를 그대로 옮긴 것이라 `run.sh` 등은 `<ws>/src/perception/sensor_fusion` 구조를 전제로 한다.
`config/ros.env`, `build.sh`, `tools/topic_audit.py`, `install/`은 A1_Total에만 있어서, 이 저장소만으로는 실행되지 않는다.

## 자차 입력 전환 (`a353f97`)

| 항목 | 이전 | 이후 |
|---|---|---|
| `ego_source` 기본값 | `imu` | `gnss_odom` |
| 입력 토픽 | `/gnss/imu` | `/a1_gnss/odom` + `/gsof_client/gsof/ins_solution_49` |
| GNSS 검증 플래그 | 없음 | `gnss_odom_verified: false` |

`fusion_node.cpp`에 `gnss_odom()`을 추가했고, `nav_msgs` 의존성이 생겼다. 다음 경우 ego를 거부한다.

| 조건 | 사유 문자열 |
|---|---|
| 플래그가 false | `gnss_odom_unverified` |
| 발행자가 하나가 아님 | `gnss_source_not_unique` |
| GSOF #49가 없거나 오래됨 | `gnss_status_missing_or_stale` |
| 정렬 상태 불량 (`imu_alignment != 4`, `gnss`가 1~6 밖) | `gnss_alignment_invalid` |
| 프레임이 `odom`/`base_link`가 아니거나 값이 유한하지 않음 | `gnss_odom_frame_or_fields_invalid` |
| 기울기 10° 초과 | `gnss_odom_tilt_unsupported` |

위치는 속도를 적분해서 만든다. `/localization/odom`의 map 보정이나 정지 유지는 ego로 쓰지 않고, 자동 fallback도 하지 않는다.

## 후면→측면 전환 연결

**문제:** 앞차가 돌면 LiDAR에 보이는 면이 바뀌어 크기와 중심이 크게 튀고, 트랙 게이트를 벗어나 ID가 끊긴다.

**수정:**

- 최신 카메라 영역과 LiDAR 군집이 일대일이고 이전 영역과 이어지면, 형상 변화만큼 측정 불확실성을 키운 뒤 연결을 판정한다 (`CameraSupport`, `shape_noise`, `assisted` 비용).
- 다른 살아 있는 트랙도 그 측정을 차지할 수 있으면 완화하지 않는다. 교차 차량 때문이다.
- 카메라 대응에는 필터된 박스 대신 측정된 원시 박스를 쓴다.
- 크기가 크게 변한 직후에는 ID가 유지돼도 3회 갱신 동안 `motion_reacquiring=true`로 두고, shadow P5를 `primary_motion_reacquiring`으로 무효화한다.

## 그 외 변경

| 항목 | 내용 |
|---|---|
| `sensor_height_m` | 1.8 → 1.2. `vehicle_extrinsics.yaml`의 LiDAR 높이 1.20 m 기준 |
| `base_from_lidar` | 평행이동을 `[1.16, 0, 1.20]`으로 입력 |
| 렌더 | 화면, 녹화, 미리보기, 구독자가 없으면 그리기를 건너뛴다 (`render_needed()`) |
| `run_ui.sh` | LiDAR 사전 점검의 `ROS_LOCALHOST_ONLY` 기본값을 `config/ros.env` 값으로 변경, 안내 문구 수정 |

## 시험과 한계

- `association_tests.cpp`에 후면→측면 전환 시험을 추가했고 `gnss_ego_smoke.py`를 새로 만들었다.
- README에 합성 시험을 통과했다고 기록돼 있다. 실차 코너의 추적률과 상대속도 정확도는 검증되지 않았다.
- A₂ 영역은 영속적인 카메라 ID가 아니며, 가림·교차 장면은 여전히 모호할 수 있다.
- `gnss_odom_verified`가 false이면 ego가 아예 거부돼 P5 값이 쌓이지 않았다. 이 문제는 [인식 점검](2026-11-06-perception-audit.md) 3번에서 드러났다.

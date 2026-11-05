# A1_Total 센서 퓨전

현재 실행 구조·토픽 계약·2 Hz 진단은 [통합 센서 파이프라인](../../../docs/SENSOR_PIPELINE_KR.md)을 기준으로 한다.

- 런타임: `cpp/` C++ ROS 패키지 `a1_fusion_cpp`. Python은 진단·시험·모델 변환용이다.
- 기본: 공용 domain0 `/lidar_points` 직접 구독. `config/ros.env`에서 공용 도메인 지정.
- LiDAR/GNSS 드라이버는 통합 workspace의 `src/external`, `src/localization`에서 한 번만 실행한다.
- `A1_FUSION_TRANSPORT=bridge`는 이전 domain107 격리 경로이며 LiDAR만 전달한다.
- 카메라 A₂ → association 동일 대상 추적 → shadow P5. 차량 CAN 명령을 보내지 않는다.
- 기본 자차 입력은 `/a1_gnss/odom` + GSOF #49 정렬 상태다(`ego_source: gnss_odom`).
  장착·시각·운동 검증은 미완료이며 `gnss_odom_verified`와 기존 검증 플래그는 false다.
  `/gnss/imu` 적분은 명시적으로 선택하는 실험 모드이며 자동 fallback하지 않는다.

```bash
cd /home/ailab/git/A1_Total
source env.sh
bash build.sh --packages-up-to a1_fusion_cpp
bash src/perception/sensor_fusion/run_sensor_fusion.sh lidar-check
bash src/perception/sensor_fusion/run_sensor_fusion.sh topic-audit --seconds 20
bash src/perception/sensor_fusion/run_sensor_fusion.sh
```

`stop`, `status`는 같은 런처에 인자로 지정한다. 센서 드라이버는 UI가 자동 기동하지 않는다.
기본 UI 실행은 카메라를 소유·기동한다. `A1_CAMERA_MODE=shared`는 같은 도메인의 기존 image_raw를 사용하고 카메라 드라이버를 시작하지 않는다.
`A1_CAMERA_MODE=off` 또는 기존 `A1_CAMERA_ENABLED=0`은 LiDAR-only 진단용이다.

`reference/`, `datasets/`, `docs/snapshots/`는 증거 자료다. 모델·녹화 원본을 보존한다.
이전 실행 상태가 누적된 문서는 [이전 README](docs/history/20261104_pre_integration_README.md),
[이전 인계 기록](docs/history/20261104_pre_integration_PROJECT_CONTEXT.md)에 원문 보존했다.
과거 문서의 상대 링크는 원래 sensor_fusion 루트를 기준으로 해석한다.

2026-11-04 영상 00001–00004 검토 후 형상 필터 수정:
실측 클러스터 점들의 XY 최소면적 사각형으로 최대 길이/폭(5.2/2.6 m)을 판정한다.
회전 시 축 정렬 폭이 커지는 문제를 줄이며, 투영 박스와 fragment 병합의 보수적 제한은 유지한다.
병합 후 전체 점으로 재계산하고 tracks.csv에 footprint_valid/length_m/width_m/yaw_rad를 남긴다.
이 yaw는 관측 표면의 축이며 차량 진행각이나 외부 보정값이 아니다.
기존 카메라 기반 대상 선택·얇은 후면 허용·0.3초 freshness·미검증 보정 상태를 유지한다.
합성 회전 차량/과대 물체 및 ROS 매칭 회귀 시험으로 확인한다. 해당 주행의 원시 점군이 없어
영상 속 실패가 이 변경으로 해결됐는지 재처리 검증은 불가능하며, 투영 오차는 별도 실측 보정이 필요하다.

2026-11-05 후면→측면 전환 연결 개선: 최신 카메라 영역과 현재 LiDAR 군집이 일대일로
대응하고 이전 영역과 연속될 때, 확인된 트랙의 형상 변화에 따른 측정 불확실성을
연결 판정 전에 적용한다. 원시 관측 박스를 카메라 대응에 사용하고, 크기 변화 직후에는
ID가 유지되어도 속도 재확인 동안 shadow P5 출력을 무효화한다. A₂ 영역은 영속적인
카메라 ID가 아니며, 가림·교차 장면은 여전히 모호할 수 있다. fragment 병합의 축 정렬
폭 제한과 관측 표면 중심 이동에 따른 속도 오차는 후속 개선 대상이다. 합성 시험을
통과했으며 실차 코너 주행의 추적률과 상대속도 정확도는 검증되지 않았다.

맵 기반 필터: localization이 선택한 동일 맵의 바닥·수직 평면 후보를 해시 검증 후
연결한다. 최근 ICP 품질과 라이다 시각의 TF가 없으면 기존 필터로 우회한다.
세 맵 비교, 토픽, 실행·재생성·검증 방법은 [맵 필터 안내](docs/map_filter/README.md)를 따른다.

### Live exposure control

The association window has an Exposure (us) field. Click it (or press E), type a value,
and press Enter or Apply. The +/-1ms buttons change exposure by 1000 us.
Actual shows camera readback; unavailable/stale means no recent confirmation.
The supported manual range is 15..1000000 us; long exposure can reduce camera FPS.
This control targets `/camera_driver_gv` (`camera_control_node` ROS parameter)
and requires manual exposure mode. UI-owned camera startup defaults to 35000 us,
overridable with `A1_CAMERA_EXPOSURE_US`. Shared cameras retain their current setting.
Values entered in the window apply to the current camera session.

# 맵 기반 바닥·벽 후보 필터 — 2026-11-04

`A1_Total`의 세 맵을 모두 읽고 원본 키프레임 점군에서 표면 캐시를 추출한다.
원본 `.mm`, `.simplemap`, georef는 수정하지 않는다. 새 파일은 각 맵 옆의
`map.surfaces.pcd`, `map.surfaces.json`이다.

| 맵 | 키프레임 | 원본 점 수 | 검토 결과 |
|---|---:|---:|---|
| `track_20261104/clean/map.mm` | 178 | 5,320,359 | 11/4 일부 구간. 원본보다 키프레임 수가 적지만 이름만으로 정확도가 검증된 맵은 아님 |
| `track_20261104/map.mm` | 321 | 10,283,772 | 11/4 일부 구간. 기존 top-down 이미지상 중앙 주행 구간 주변 |
| `track_merged/map.mm` | 2,739 | 27,347,584 | 10/8 저장된 전체 트랙 병합 맵. 전체 트랙 범위가 필요할 때 후보 |

세 맵의 지리 기준점과 ENU→map 변환이 다르다. 11/4 맵의 회전은 약 148°,
병합 맵은 ENU 축과 일치한다. 동일한 `map` 프레임 이름만 보고 다른 맵 점군을
섞으면 안 된다. 상세 원본 정보는 [map_inventory.txt](map_inventory.txt)에 있다.
현재 `maps/track`은 비어 있으므로 실제 실행에는 `map:=...`를 지정해야 한다.
기존 기본 맵이나 localization 동작을 임의로 다른 맵으로 바꾸지 않았다.

전체 점군에서 추출한 표면 수(검출 정확도가 아닌 캐시 구성 수치):

| 맵 | 바닥 후보점 | 벽 후보점 |
|---|---:|---:|
| 11/4 clean | 232,643 | 7,309 |
| 11/4 원본 | 137,624 | 7,754 |
| 병합 트랙 | 179,664 | 2,539 |

[세 맵 표면 비교 그림](map_surfaces_comparison.png): 회색은 바닥, 빨강은 벽 후보다.
낮은 가드레일이나 희박한 벽은 현재의 보수적인 평면/지지 범위 조건에서 빠질 수 있다.

## 실행 연결

기존 localization 런처에 `map_surface_filter` 인자(기본 true)를 추가했다.
localization 모드에서 선택된 **동일 `.mm`** 및 표면 PCD의 SHA-256이 manifest와
일치할 때만 `map_surface_server`를 시작한다. 캐시가 없거나 바뀌었으면 알림을
남기고 맵 필터만 우회한다. mapping 모드의 변화하는 맵에는 적용하지 않는다.
실행 중 MOLA 서비스로 맵을 교체했다면 localization 런처 전체를 같은 맵으로 재시작한다.

예: 전체 트랙 병합 맵을 localization에도 쓰기로 한 경우:

```bash
cd /home/ailab/git/A1_Total
source env.sh
ros2 launch a1_bringup localization.launch.py map:=$PWD/maps/track_merged/map
```

이는 센서 드라이버를 포함하는 실제 실행 명령이다. 이번 개발 검증에서는 실행하지 않았다.
이미 드라이버를 실행 중이면 기존 런처 옵션 `start_lidar_driver:=false start_gnss_driver:=false`를 사용한다.
퓨전 UI는 기존 실행 방법을 사용한다. 기본 direct 모드에서 localization과 같은 ROS domain이어야 한다.
bridge 모드는 LiDAR만 전달하므로 다른 도메인의 map/TF/quality를 자동으로 사용할 수 없다.

- `/localization/map_surfaces`: reliable/transient-local, map 프레임의 PointNormal 점군.
- `/localization/map_surfaces_epoch`: 10 Hz 서버 생존 및 해당 점군 세대 식별용 Header.
- `/lidar_odometry/pose_quality`: 기존 ICP 품질. 0.5 이상, 최근 0.3초 입력 필요.
- `/tf`, `/tf_static`: 해당 라이다 시각의 `map ← LiDAR header.frame_id` 변환 필요.
- `/lead_fusion/map_removed`: 구독자가 있을 때 제거된 점 발행, lidar_vehicle 프레임.
  intensity 1=바닥 후보, 2=벽 후보.

정확한 시각의 TF에 장착 위치·회전이 포함된다. 퓨전의 lidar_vehicle는
라이다 원점에 남아 있으므로 lidar_yaw의 역회전을 적용해 중복 회전이나
base_link 레버암 누락을 피한다. 보정값을 새로 추정하거나 덮어쓰지 않는다.

## 필터 동작과 한계

오프라인: 전체 키프레임 점을 0.2 m sparse voxel로 축소하고 반경 1.2 m의
최소 20점으로 국소 평면을 검출한다. 수평에 가까운 평면 중 2 m XY 셀의
최저점에서 0.3 m 이내인 표면은 바닥 후보, 충분한 수직/수평 지지를 갖는
수직 평면은 벽 후보로 저장한다. PCL의 큰 맵 voxel 인덱스 overflow를 피한다.
MOLA 표시용 최대점수 제한을 해제하므로 RViz용 축소 점군만 분석하지 않는다.

런타임: Patchwork++ → ROI/voxel → 맵 표면 비교 → 기존 잔여 바닥 제거 → clustering.
세 개의 가까운 맵 점 모두 0.3 m 안에 있고 같은 종류이며 각 평면까지 거리가
0.12 m 이내인 점만 제거한다. 무한 평면이나 맵 바깥으로 필터를 확장하지 않는다.
맵 없음, 서버 종료, 세대 불일치, 복수 발행자, 낮거나 오래된 품질, TF 누락 시
점들을 그대로 유지하고 기존 필터를 계속 사용한다.

이 맵에는 의미론적 벽 라벨이 없다. 따라서 **기하학적 수직 평면 후보**이며,
맵에 포함된 정차 차량·높은 시설물도 후보에 들어갈 수 있다. 벽에서 0.12 m 이내로
붙은 물체, 잘못된 localization, 오래된 시설물에는 실차 검증이 필요하다.
원시 주행 점군이 없어 11/4 영상 00001–00004의 실제 성능 개선은 재처리 검증하지 못했다.
스캔 내 ego deskew는 기존과 같이 미적용이다. 주행 중 스캔 내부 움직임이 커지면
맵과의 근접 매칭이 감소하거나 부정확해질 수 있어 동적 실차 검증이 필요하다.
2 Hz 입력 빈도 문제를 해결하는 변경도 아니다.

UI의 `MAP` 표시와 association.csv의 `map_filter_status`, `map_ground_removed`,
`map_wall_removed`를 확인한다. map_filter_enabled:=false로 퓨전 측 적용을 끌 수 있고,
localization의 map_surface_filter:=false로 맵 서버를 끌 수 있다.

## 재생성 및 검증

```bash
python3 src/perception/sensor_fusion/tools/prepare_map_surfaces.py \
  maps/track_merged/map.mm --binary install/lib/a1_fusion_cpp/map_surface_export
ctest --test-dir build/a1_fusion_cpp --output-on-failure
python3 src/localization/a1_localization/test/test_surface_cache.py
```

합성 시험: 바닥/벽 검출, 벽 앞 0.5 m 물체 보존, 맵 범위 밖 보존,
좌표 회전·이동, 품질 저하/중단, TF 누락, 맵 세대 불일치 및 서버 종료 우회,
맵과 캐시 해시 불일치 차단, association 파이프라인에서 벽 제거·앞 차량 유지.
기존 회전 차량·얇은 후면·추적 유지·모호 매칭·stale 시험도 함께 수행한다.

최종 확인: A1_Total에서 필요한 8개 패키지 빌드 성공, CTest 3/3 통과,
설치 바이너리의 ROS 맵 필터 및 association 회귀 시험 통과.
[최종 시험 로그](verification.txt)를 보존한다. 센서·차량 명령은 실행하지 않았다.

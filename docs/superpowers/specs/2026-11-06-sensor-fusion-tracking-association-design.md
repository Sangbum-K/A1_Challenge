# sensor_fusion 추적·카메라 대응 개선 설계

- 작성일: 2026-11-06
- 상태: 구현 중. 코드에서 가능한 범위 구현 완료, 현장 작업 대기 (16절)
- 대상 코드: `sensor_fusion/`
- 파라미터 값: 별도 표시가 없으면 첫 bag으로 조정할 시작값

## 1. 목표와 범위

### 목표

트랙 주행에서 앞차를 같은 ID로 오래 유지한다. 벽과 가드레일은 맵으로 거르고, 카메라는 차량 근거를 더한다.
원거리 차량은 움직임을 기록하고, 평가를 통과한 뒤에만 주 대상으로 쓴다.

### 사용자 결정 (2026-11-06)

| 항목 | 결정 |
|---|---|
| 추적 5개, 대응 3개 | 모두 구현 |
| Must fix 1~6 | 모두 진행 |
| 경로 기준 | 레이싱라인 |
| 정적 구조물 판단 | 맵 정보 사용 |
| F2 재설계 | 보류. 실차 제어기 모델 미확정 |
| 120 m 확장 | 차량으로 인식하고 움직임을 기록할 수 있을 때만 허용 |

### 범위 밖

- `core.cpp`의 `p5()` 식, gate, F1 결합 변경
- CAN 출력과 차량 명령
- 2D 검출기 추가
- 레이싱라인 속도 프로파일을 제어에 쓰는 일

### 유지할 원칙

- 차량 명령을 보내지 않는다.
- 측정하지 않은 보정값을 만들지 않는다. 검증 플래그는 실측 후에만 켠다.
- 50 m 이내에서는 카메라 근거 없는 물체가 주 대상이 되지 못한다 (T877 회귀 시험).
- 같은 A₂ 프레임이 근거를 두 번 올리지 못한다.
- 코드 시험, 센서 수신, 처리량, 실제 정확도를 구분해서 보고한다.

## 2. 구조

### 현재

```text
cloud → Patchwork++ → ROI/voxel → 맵 표면 제거 → 잔여 바닥 제거 → 군집 → 조각 병합
      → Tracker(차체 좌표, CV, 중심점)
      → 영역 임계값 → 1:1 IoU 매칭 → 주 대상 선택 → 렌더(같은 스레드)
```

### 변경 후

```text
/a1_gnss/odom → EgoHistory ───────────────────────────────────┐
                                                              ▼
cloud → Patchwork++ → ROI/voxel → 맵 표면 제거(시각 보정) → 잔여 바닥 제거
      → 군집 → Detection(world, 강/약 등급, map_static 비율)
      → Tracker: CTRV 예측 → 트랙 주도 수집 → 3단계 할당 → 갱신
      ──────────────────────────────┐
A₂ 프레임(21 Hz) → CameraEvidence ──┤
map TF + Raceline ──────────────────┤
                                    ▼
                  EvidenceSelector → 발행, CSV → RenderSnapshot
                                                     ▼
                                     렌더 스레드(별도 callback group)
```

## 3. 파일 변경

| 파일 | 변경 |
|---|---|
| `cpp/include/a1_fusion/core.hpp`, `cpp/src/core.cpp` | `EgoHistory`, `Ego::verified` 추가. `Engine::primary()` 시그니처 변경 |
| `cpp/include/a1_fusion/gnss_ego.hpp` (신규) | `/a1_gnss/odom` 해석을 두 노드가 공유 |
| `cpp/include/a1_fusion/tracker.hpp`, `cpp/src/tracker.cpp` (신규) | 트래커 교체 |
| `cpp/include/a1_fusion/camera_evidence.hpp`, `cpp/src/camera_evidence.cpp` (신규) | 근거 점수, 가림 판정, 대상 선택 |
| `cpp/include/a1_fusion/raceline.hpp`, `cpp/src/raceline.cpp` (신규) | CSV 로드, Frenet 투영 |
| `cpp/src/association.cpp`, `cpp/include/a1_fusion/association.hpp` | 군집·형상 함수만 남김. `Tracker`, `CameraTargetSelector`, `CameraSupport`, `unique_region_matches`, `select_primary`, `horizontal_iou` 삭제 |
| `cpp/src/map_surfaces.cpp`, `cpp/include/a1_fusion/map_surfaces.hpp` | 구조물 라벨, 커버리지, 거리별 허용오차 |
| `cpp/include/a1_fusion/map_structure.hpp`, `cpp/src/map_structure.cpp` (신규) | 키프레임별 점유 집계 로직. PCL 없이 시험 |
| `cpp/src/map_surface_export.cpp` | 위 로직 연결. 키프레임 접근 확인 전까지 보류 (16절) |
| `cpp/include/a1_fusion/map_filter_ros.hpp` | 접근자, 시각 보정, 상태 세분화 |
| `tools/prepare_map_surfaces.py` | 구조물 출력이 연결되면 `planar_surfaces_v2`로 변경 |
| `cpp/src/association_node.cpp` | 파이프라인 연결, 대기열, 전용 렌더 스레드 |
| `cpp/src/fusion_node.cpp` | M3, M5, 새 필드 사용 |
| `cpp/msg/*.msg` | 4절 |
| `cpp/config/calibration.yaml` (신규) | 카메라·LiDAR 기하 값을 한 곳으로 |
| `cpp/config/association.yaml`, `cpp/config/shadow.yaml` | 새 파라미터, 중복 제거 |
| `cpp/config/bag_qos_override.yaml` (복사) | `reference/A1_RealCamera_A2_21FPS/configs/`에서 복사 |
| `run_ui.sh`, `run.sh` | 노출, bag 녹화, `replay` 모드 |
| `cpp/tools/track_eval.py` (신규) | bag 재생 결과 평가 |
| `cpp/tests/tracker_tests.cpp`, `cpp/tests/camera_evidence_tests.cpp`, `cpp/tests/raceline_tests.cpp` (신규) | ROS 없이 실행 |
| `cpp/tests/association_tests.cpp`, `cpp/tests/map_surface_tests.cpp` | 시험 이동과 추가 |
| `cpp/tests/*_smoke.py` | 메시지·상태 문자열 변경 반영 |
| `cpp/CMakeLists.txt` | 신규 라이브러리와 시험 등록 |

`tracker`, `camera_evidence`, `raceline`, `map_structure`는 ROS와 PCL에 의존하지 않는다. `run.sh core`로 시험한다.

## 4. 메시지와 출력 정의

생산자와 소비자가 모두 이 패키지 안에 있으므로 호환 필드를 남기지 않는다.

### ClusterTrack.msg

```text
# 삭제: apparent_velocity_mps, horizontal_iou
float64[2] relative_velocity_mps   # 차체 축 기준 (앞차 속도 − 자차 속도)
float64 near_surface_x_m           # LiDAR 원점 기준, 가까운 후면 모서리의 차체 X. scan 시각
bool weak_measurement              # 이번 scan이 약한 측정으로 갱신됨
float64 camera_evidence            # 누적 근거 L
string camera_state                # visible | occluded | outside | silent
float64 map_static_ratio           # 맵 구조물과 겹치는 점 비율의 EMA. 맵 상태가 ok가 아니면 NaN
bool map_static                    # 9.2절 거부 조건
float64 far_score                  # 8.2절. 35 m 이내이거나 전제 실패면 NaN
bool far_vehicle                   # 8.2절 판정
float64 path_s_m                   # 자차 기준 레이싱라인 전방 거리. 없으면 NaN
float64 path_n_m                   # 레이싱라인 횡거리(절대값 아님, 부호 있음). 없으면 NaN
```

### Association.msg

```text
string ego_status        # compensated | unavailable
string map_status        # ok | no_structure | MapFilterRos::begin()의 나머지 상태 문자열
string raceline_status   # active | no_localization | unverified | misaligned | disabled
uint32 dropped_scans     # M6 누적값
```

`motion_compensated`는 `ego_status == compensated`일 때 true다.
`primary_status`는 `measured | camera_held | predicted`다.

`status`는 다음 중 하나다.

```text
no_image | no_a2 | a2_history_unready | no_target | camera_held
| candidate_extrinsics_unverified | candidate_clock_unverified | matched_geometry_only
```

### 필드 의미

| 필드 | 정의 |
|---|---|
| `center_m` | 후면 기준점 + `L̂/2`를 진행 방향으로 이동한 뒤 scan 시각 차체 좌표로 변환 |
| `size_m` | `[L̂, Ŵ, Ĥ]` 추정값 |
| `raw_center_m`, `raw_size_m` | 마지막 측정의 차체 AABB. 기존과 같음 |
| `range_m` | LiDAR 원점에서 후면 기준점까지 평면 거리 |
| `position_covariance_xy` | 차체 축으로 회전한 값 |
| `camera_matched` | 최근 0.3초 안의 프레임에서 `visible`이고 점수 ≥ 0.4 |
| `matched_bin_left/right` | 트랙의 투영 bin 구간. 없으면 −1 |
| `motion_reacquiring` | `!rear_visible` |
| `primary_camera_age_s` | 주 대상이 `camera_matched`였던 마지막 시점부터의 시간 |

`selection_reason` 값:

```text
extent_exceeded | insufficient_geometry | track_unconfirmed | track_expired_or_uncertain
| map_static | outside_target_range | outside_lane | camera_evidence_low
| far_not_vehicle | selected | standby
```

`ego_status=unavailable`이면 `relative_velocity_mps`는 차체 좌표 겉보기 속도다.

### BrakeEstimate.msg, EgoMotion.msg

```text
BrakeEstimate: float64 request_unverified_mps2
EgoMotion:     float64 speed_unverified_mps
```

## 5. Must fix

### M1. 노출

- `run_ui.sh:110`의 기본값 `35000`을 `3000`으로 변경한다.
- 2000/3000/5000 µs를 30초씩 찍어 band 포화율과 A₂ 재현율을 비교한다.
- 완료 조건: 주간 주행에서 band 포화율 5% 이하.

### M2. 카메라 보정

1. 현재 ROI(1280×720, offset 320/240)에서 체커보드로 fx, fy, cx, cy, 왜곡을 측정한다.
2. 10/20/30/50 m 표적으로 LiDAR-카메라 외부 파라미터를 맞춘다.
3. 값은 `calibration.yaml`에 둔다. `run.sh`가 `--params-file`을 추가로 넘긴다. 키는 `/**:`로 두 노드에 적용한다.
4. `project()`에 왜곡 계수 `k1, k2, p1, p2`를 추가한다.
5. 재투영 오차가 1 bin(32 px) 이하일 때만 `calibration_verified`를 켠다.

입력 축소 실험(전체 폭 1920 → 1280)은 fx≈1100이 확인될 때 별도로 한다. 이번 구현에 포함하지 않는다.

### M3. 미검증 값 기록

`a1::Ego`에 `bool verified{}`를 추가한다. `valid`는 "값이 쓸 수 있다", `verified`는 "장착·시각 검증 완료"를 뜻한다.

`gnss_odom()`은 값을 계산한 뒤 검증 여부를 기록한다.

```cpp
a1::Ego e = parse_gnss_odom(m, previous);   // 프레임·유한성·정렬 실패는 지금처럼 거부
e.verified = gnss_odom_verified_;
egos_.push(e);
publish_ego(e, e.verified ? "gnss_navigation" : "gnss_odom_unverified");
```

`publish_ego()`는 `speed_unverified_mps`를 항상 채운다. `speed_mps`와 `m.valid`는 `e.verified`일 때만 채운다.

`Engine::base()`는 `!e.verified`이면 `ins_invalid_or_unsynchronized`로 거부한다.
옛 `camera`, `fusion` 모드가 미검증 ego를 쓰지 못하게 하기 위해서다.

`Engine::primary()`는 `core.cpp:155-156`의 조기 반환을 없애고 끝에서 판단한다.

```cpp
finish(r, e);
r.request_unverified = r.request;
if (!r.valid) return r;
const char* reason = !config_.clock_valid ? "clock_unverified"
                   : !config_.calibration_valid ? "extrinsics_unverified"
                   : !e.verified ? "ego_unverified" : nullptr;
if (reason) { r.valid = false; r.request = nan; r.reason = reason; }
```

`fusion_node.cpp:213-214`의 조기 반환도 삭제한다. 메시지 검증 플래그는 `Engine`에 넘긴다.
`fusion_node.cpp:237`의 `primary_motion_verified` 처리는 그대로 두되 `request_unverified`는 유지한다.

`request_unverified_mps2`의 소비자는 UI와 CSV뿐이다. UI는 회색으로 `UNVERIFIED`를 붙인다.

### M4. bag 녹화

`A1_RECORD_BAG=1`이면 `run_ui.sh`가 별도 프로세스로 녹화한다.

```bash
launch bag ros2 bag record -o "$OUT/bag" --max-bag-size 4000000000 \
  --qos-profile-overrides-path "$ROOT/cpp/config/bag_qos_override.yaml" \
  /camera_driver_gv/vis/image_raw "$LIDAR_INPUT" /lead_fusion/a2 \
  /a1_gnss/odom /gsof_client/gsof/ins_solution_49 \
  /tf /tf_static /lidar_odometry/pose_quality
```

- 시작 전 빈 공간이 20분 분량(약 64 GB) 이상인지 검사하고, 부족하면 실행을 거부한다.
- `/lead_fusion/a2`를 넣어 재생 시 GPU 없이 같은 A₂ 입력을 쓴다. 재비교가 결정적이 된다.
- 맵 토픽은 녹화하지 않는다. 재생할 때 `map_surface_server`를 같은 PCD로 띄운다.

`run.sh replay <bag> [--rerun-a2]`:

1. `map_surface_server`를 시작한다.
2. association, fusion 노드를 `use_sim_time:=true`로 시작한다.
3. `ros2 bag play --clock --rate 1.0 <bag>`을 실행한다.

`MapFilterRos`의 품질 검사는 `steady_clock`을 쓰므로 재생 배속은 1.0으로 고정한다.

### M5. 카메라 미검출 시 대상 유지

9.1~9.4절의 근거 점수와 11절의 선택기로 해결한다. `fusion_node.cpp:217-220`의 조건 전체를 아래로 바꾼다.

```cpp
const bool status_ok = m.primary_status == "measured" || m.primary_status == "camera_held";
if (!status_ok || !tr.measured) { invalid("primary_camera_unconfirmed", t); return; }
```

`tr.camera_matched`, `m.association_ambiguous`, `primary_camera_age_s` 검사가 함께 빠진다.

간격 계산도 새 필드를 쓴다.

```cpp
target.gap = tr.near_surface_x_m + lidar_from_.translation().x() - cfg_.bumper_x;
target.relative_speed = tr.relative_velocity_mps[0];
```

`fusion_node.cpp:229`의 yaw rate 보정은 삭제한다.

### M6. LiDAR 대기열과 렌더 분리

먼저 기존 `association.csv`의 `processing_ms`로 지연을 확인한다.

```cpp
const auto ready = [&](const auto& e){ return age(e.first) >= .10; };
while (pending_.size() > 1 && ready(pending_[1])) { pending_.pop_front(); ++dropped_scans_; }
if (pending_.empty() || !ready(pending_.front())) return;
```

노드는 단일 스레드 executor로 돌아가므로 타이머로 옮겨서는 렌더가 인식을 계속 막는다.
`MultiThreadedExecutor`는 콜백 스레드가 고정되지 않아 OpenCV 창과 맞지 않는다. 그래서 **전용 렌더 스레드**를 쓴다.

- `process()`는 끝에서 `RenderSnapshot`을 만들어 mutex로 교체한다. 이미지와 점은 포인터나 사본으로 넘긴다.
- 렌더 스레드가 10 Hz로 최신 스냅샷만 그린다. HighGUI 호출은 전부 이 스레드에서 한다.
- `cv_bridge` 변환, BEV 그리기, `imshow`, `imwrite`, `recorder_->push`, `ExposureControl` 호출을 옮긴다.
- `ego_ui_`, `brake_ui_`는 같은 mutex로 보호한다.

UI의 BEV 점 샘플링(`i+=4`)을 없애고 더 밝게 그린다. 군집 품질과 무관한 표시 변경이다.

## 6. 맵 사용

### 6.1 현재 맵 필터의 한계

| 사실 | 영향 |
|---|---|
| 벽 라벨은 수직 범위 1.2 m 초과 조건이 필요 (`map_surfaces.cpp`) | 낮은 가드레일이 벽으로 저장되지 않음 |
| 병합 맵의 벽 후보는 2,539점뿐 | 시설물 대부분이 제외됨 |
| `match()`는 "맵 바깥"과 "표면 없음"을 구분하지 않음 | `active`여도 맵 밖이면 조용히 통과 |
| 허용오차가 거리와 무관 (0.3 m, 0.12 m) | 원거리는 localization 각도 오차로 불일치 |
| scan 내부 운동 보정 없음 | 주행 속도에서 매칭 감소 (README에 기재) |

### 6.2 구조물 라벨 추가

`map_surface_export`에서 평면 조건과 별개로 **정적 구조물** 점을 추출한다.

1. 키프레임별로 점을 0.2 m voxel에 넣고, voxel마다 관측한 키프레임 수를 센다.
2. 2 m 셀의 최저점보다 0.25~3.5 m 높은 voxel만 남긴다.
3. 서로 다른 키프레임 3개 이상에서 보인 voxel을 구조물로 저장한다 (`curvature=3`, 법선은 `(0,0,1)`).

3번이 지도 제작 중 지나간 차량을 걸러낸다. 움직이는 차는 같은 voxel에 3개 키프레임 동안 머물지 않는다.
**한계:** 지도 제작 때 주차돼 있던 차량은 구조물로 남는다. 그 위치에 다시 정차한 차는 거부된다.

키프레임별 접근은 `KeyframePointCloudMap`이나 `.simplemap`에서 얻는다. 어느 쪽을 쓸지는 15절에서 확인한다.

토픽과 PCD 형식은 유지한다. `prepare_map_surfaces.py`의 `algorithm`을 `planar_surfaces_v2`로 올리고 캐시를 다시 만든다.

`MapSurfaces`:

```cpp
void set(const Surfaces::ConstPtr&);                     // 라벨 3은 별도 KD-tree
int match(const Eigen::Vector3d& q, double range) const; // 표면 제거. 라벨 1, 2만
bool is_structure(const Eigen::Vector3d& q, double range) const;
bool covered(const Eigen::Vector3d& q) const;            // 5 m 안에 맵 점이 있는가
bool has_structure() const;
```

거리별 허용오차:

```cpp
const double radius = .30 + .003 * range;
const double plane  = .12 + .003 * range;
```

옛 PCD에는 라벨 3이 없으므로 `has_structure()==false`가 되고 `map_status=no_structure`로 표시한다.

### 6.3 시각 보정

`map_from_body_`는 scan 시각의 pose 하나다. 점이 측정된 시각의 차체 위치로 옮겨서 비교한다.

```cpp
Eigen::Vector3d q_map = map_from_body * body_at_t_from_body_at(point_time) * q_body;
```

`point_time`은 7.1절의 방위각 시각표에서 얻는다. ego가 없으면 지금처럼 보정 없이 비교한다.

### 6.4 맵 상태

`map_status`는 `MapFilterRos::begin()`의 기존 상태에 구조물 유무를 더해 만든다.
트랙별 커버리지는 측정 위치의 `covered()`로 판단한다.

| 항목 | `map_status=ok`, 위치 covered | 그 외 |
|---|---|---|
| 벽·바닥 제거 | 수행 | 수행 안 함 (기존 필터만) |
| `map_static` 거부 | 수행 | 판단 불가, 비율 NaN |
| 약한 측정의 새 트랙 생성 | 허용 | **금지** |
| `far_vehicle` | 판정 | **항상 false** |
| 주 대상 범위 | 설정값 | **50 m로 제한** |
| 50 m 이내 일반 경로 | 수행 | 수행, UI에 `MAP OFF` 경고 |

원거리 경로는 맵을 전제로 한다. 맵 없이 원거리 물체를 차량으로 판단하지 않는다.

실행 맵은 전체 트랙이 들어 있는 `track_merged`여야 한다. 11/4 맵은 일부 구간만 포함한다.

bridge 전송 모드는 맵과 TF를 받지 못하므로 항상 맵 없음 상태다. 기본값인 direct 모드를 쓴다.

## 7. 추적

### 7.1 T1. 자차 운동 보상

**목적:** 트랙 상태를 연속 odom 좌표계에 두어 실제 앞차 속도를 추정한다.

`fusion_node`의 `ego_at()`를 공용 클래스로 옮기고, 기존 `a1::Ego`, `to_world`, `to_body`를 그대로 쓴다.

```cpp
class EgoHistory {
 public:
  void push(const Ego& e);                 // 단조 증가만 허용, 2초 유지
  std::optional<Ego> at(double t) const;   // 앞뒤 표본 보간. 최신 이후 0.5초까지 등속 외삽
  void clear();
};
```

반환값의 `stamp`는 요청한 `t`다. `fusion_node`는 기존처럼 `pair_slop_s`를 넘는 외삽을 거부한다.

연속 위치는 지금처럼 odom 속도를 적분해서 만든다. map 보정 위치는 쓰지 않는다.
이 코드를 `gnss_ego.hpp`로 옮겨 두 노드가 함께 쓴다.

**시각표:** scan을 1° 방위각 칸으로 나눠 평균 시각을 구한다.

```cpp
struct ScanTiming {
  std::array<double,360> time;
  bool valid;                       // 시간 폭이 5 ms 이상일 때만 true
  double at(double azimuth) const;  // 무효면 scan 시각
};
```

호스트 수신 시각을 쓰는 드라이버 설정에서는 점 시각이 뭉칠 수 있다. 그때는 `valid=false`로 scan 시각을 쓴다.

**world 변환:** `elevated` 점 전체를 방위각 칸별 ego pose로 변환한다. T3가 이 점을 쓴다.
군집 시각은 군집 중심 방위각의 값이다. 점 단위 deskew 대신 방위각 칸 단위로 처리한다.

```cpp
struct Detection {
  std::vector<int> points;           // Scan 점 배열의 인덱스
  double time;
  Box body_box;
  Quality quality;                   // Reject | Weak | Strong
};
```

**ego 소실:** `at()`이 0.5초 외삽을 넘겨 실패하면 `ego_status=unavailable`이다. 전환은 `EgoLink`가 맡는다.

- 항등 pose를 쓰고 `Q`를 4배로 키운다. 지금과 같은 차체 좌표 추적이다.
- 좌표계가 바뀌므로 트래커를 초기화한다.
- `compensated`로 돌아갈 때는 ego가 1초 이상 연속이어야 한다. 깜빡임으로 초기화가 반복되지 않게 한다.

**상대속도:**

```cpp
const Vec v_lead(tr.v() * cos(tr.yaw()), tr.v() * sin(tr.yaw()), 0);
relative_velocity_mps = Rz(-ego.yaw) * (v_lead - ego.velocity);
```

### 7.2 T2. 후면 기준점

**목적:** 후면에서 측면이 보여도 추적점이 흔들리지 않게 한다.

```cpp
struct AnchorFit {
  Eigen::Vector2d anchor;     // 후면 중심
  double sigma_along, sigma_lateral;
  double width;
  bool rear_visible;
};
AnchorFit rear_anchor(const std::vector<Eigen::Vector2d>& xy, double heading,
                      const Eigen::Vector2d& sensor, double width_prior);
```

1. 진행 방향 축 `u`와 직교 축 `v`로 점을 변환한다. `a = p·u`, `b = p·v`.
2. `a_rear`는 `a`의 5% 분위수다. 점이 20개 미만이면 최솟값을 쓴다.
3. 후면 점은 `a < a_rear + 0.5`인 점이다. 폭은 후면 점의 `b` 범위다.
4. 폭이 1.2 m 이상이면 `b_c`는 양 끝의 중앙이다.
5. 폭이 좁으면 센서에 가까운 쪽 끝에서 `width_prior/2`만큼 옮긴다.
6. 센서 시선과 진행 방향의 각도가 60° 이하면 `rear_visible=true`다.

| 값 | 시작값 |
|---|---|
| `sigma_along`, 후면 보임 | 0.25 m |
| `sigma_along`, 후면 안 보임 | 3.0 m (필터가 종방향 측정을 사실상 무시) |
| `sigma_lateral`, 폭 충분 | `max(0.15, 0.003·range)` m |
| `sigma_lateral`, 폭 부족 | 0.6 m |
| `width_prior` | 트랙 EMA, 초기 1.9 m |

`heading` 우선순위:

1. 트랙 속도 3 m/s 이상이면 트랙의 ψ
2. 레이싱라인 active면 해당 위치의 접선
3. 자차 yaw

`near_surface_x_m`은 필터된 후면 양쪽 모서리 중 가까운 쪽의 차체 X다. 측정 유무와 관계없이 같은 정의를 쓴다.

이 변경으로 `shape_noise`, `assisted` 비용, 카메라 지원 분기(`association.cpp:215-257`)와 `CameraSupport`를 삭제한다.

### 7.3 T3. 트랙 주도 수집

**목적:** 차가 여러 군집으로 쪼개지거나 다른 물체와 붙어도 측정 하나로 만든다.

확정 트랙마다 측정 시각의 예측 위치에 진행 방향 박스를 만든다.

```cpp
struct OrientedRegion {
  Eigen::Vector2d origin;      // 예측 후면 중심
  double yaw, rear, front, half_width;
};
```

- 종방향: `[-0.8 - 2σ_a, L̂ + 0.8 + 2σ_a]`
- 횡방향: `±(Ŵ/2 + 0.6 + 2σ_l)`
- `L̂` 초기값 4.5 m, `Ŵ` 1.9 m, 값은 관측에서 EMA로 갱신한다.

처리 순서:

1. world 점 전체에서 영역 안의 점을 모은다. 트랙 수가 적으므로 전수 검사를 쓴다.
2. 점이 두 영역에 속하면 Mahalanobis 거리가 작은 트랙이 갖는다.
3. 트랙 축 기준 횡폭이 2.6 m, 종폭이 5.6 m를 넘거나 `vehicle_extent`를 넘으면 수집을 취소한다. 벽 조각은 최소면적 사각형에서 얇게 보이므로 트랙 축으로 다시 잰다.
4. 구조물 비율이 0.6 이상이면 수집을 취소한다. 코스팅 트랙이 가드레일을 끌어오는 일을 막는다.
5. 점이 3개 이상이면 그 트랙의 후보 측정으로 만든다.
6. 수집된 점을 일반 군집에서 빼고, 남은 점이 있으면 박스를 다시 계산한다.

수집 측정도 Mahalanobis 게이트를 통과해야 한다.

### 7.4 T4. 3단계 할당

**목적:** 약한 측정으로 기존 트랙을 잇고, 원거리 물체도 미확정 트랙으로 시작할 수 있게 한다.

```cpp
Quality classify(const Box& b, double range, double weak_min_range) {
  if (full_vehicle_geometry(b) || camera_surface_geometry(b)) return Quality::Strong;
  const auto d = b.hi - b.lo;
  const bool plausible = b.points >= 3 && d.head<2>().maxCoeff() >= .3 && vehicle_extent(b);
  return plausible && range >= weak_min_range ? Quality::Weak : Quality::Reject;
}
```

`classify()`는 `b.footprint` 계산 뒤에 호출한다.

군집 입력도 함께 바꾼다.

- `adaptive_clusters`의 `min_points`는 3으로 낮춘다.
- `association_node.cpp:201`의 필터를 거리별로 바꾼다.

```cpp
const bool far = range >= weak_min_range_m;
if (group.size() < (far ? 3 : cluster_min_points)) continue;
if (max_xy < .3 || max_xy > 10 || (!far && dim.z() < .15)) continue;
```

| 단계 | 트랙 | 측정 | 게이트 |
|---|---|---|---|
| 1 | 확정 | 수집 측정, Strong | χ² 9.21, 거리 3 m |
| 2 | 1단계에서 남은 확정 | Weak | χ² 5.99 |
| 3 | 미확정 | 남은 Strong, Weak | 7.5절 2점 초기화 |
| 생성 | 없음 | 남은 Strong | 없음 |
| 생성(약한 측정) | 없음 | 남은 Weak | 6.4절 맵 조건, 구조물 비율 0.6 미만 |

확정 트랙의 게이트 안에 있는 측정은 쓰이지 않았어도 새 트랙을 만들지 않는다. 같은 차에 ID가 둘 생기는 것을 막기 위해서다.

약한 측정으로 생긴 트랙은 최근 10 scan 중 4회 이상 측정돼야 확정된다.
미확정 약한 트랙은 `max_weak_tracks`(30)개까지만 유지하고, 넘으면 가장 오래 측정되지 않은 것부터 지운다.

비용:

```cpp
cost = .7 * mahal / gate + .3 * std::min(1., width_error / 1.5);
```

1위와 2위 비용 차가 0.03 미만이면 측정을 버리지 않는다. 낮은 쪽으로 갱신하되 `R`을 4배로 키운다.

### 7.5 T5. CTRV와 공분산 기반 수명

**목적:** 코너에서 앞차의 회전을 예측하고, 불확실성으로 소실을 판단한다.

상태: `x = [px, py, v, ψ, ω]` (world)

```text
|ω| ≥ 1e-3:
  px += v/ω ( sin(ψ+ωΔt) − sin ψ )
  py += v/ω ( cos ψ − cos(ψ+ωΔt) )
|ω| < 1e-3:
  px += v cos ψ Δt
  py += v sin ψ Δt
ψ += ωΔt
```

프로세스 잡음:

```text
G = [ Δt²/2·cosψ   0
      Δt²/2·sinψ   0
      Δt           0
      0            Δt²/2
      0            Δt   ]
Q = G diag(σa², σα²) Gᵀ      σa = 3 m/s², σα = 1 rad/s²
```

- 측정: `H = [I₂ 0]`, `R = Rot(h) diag(σ_along², σ_lateral²) Rot(h)ᵀ`
- 갱신: Joseph form
- 제약: `v ≥ 0`, `|ω| ≤ 1 rad/s`, ψ는 `[-π, π]`로 정규화

**2점 초기화:** world 좌표에서는 초기 속도가 틀리면 다음 scan에서 게이트를 벗어난다.
그래서 `hits==1`인 트랙은 칼만 게이트를 쓰지 않는다.

1. 생성 시 위치만 저장한다. `v=0`, ψ는 T2의 heading이다.
2. 다음 측정은 거리 `≤ 50 m/s × Δt + 1 m` 안에서 찾는다.
3. 짝을 찾으면 변위로 `v`를 정한다. 변위가 0.5 m 이상이면 ψ도 변위 방향으로 정한다.
4. `P = diag(.25, .25, 25, .5, .25)`로 시작한다.

예측은 트랙별 시각에서 측정 시각까지 수행한다.

```cpp
void Track::predict_to(double t);          // 상태 변경
TrackState Track::peek(double t) const;    // 카메라 투영용, 상태 불변. 음수 Δt 허용
```

수명:

| 항목 | 현재 | 변경 |
|---|---|---|
| 확정 | `hits >= 3` | 최근 8 scan 중 3회 이상 |
| 출력 가능 | 소실 0.3초 이내 | 소실 0.5초 이내, 위치 분산 ≤ 1 m² |
| 삭제 | 소실 0.8초 | 소실 1.5초 또는 위치 분산 합 > 4 m² |

`seen`은 `uint16_t` 비트마스크다.

## 8. 원거리 경로

### 8.1 추적과 대상의 분리

기록 범위와 대상 범위를 나눈다.

```yaml
processing_range_m: 120.0     # 추적과 로그
target_max_range_m: 50.0      # 주 대상. 8.3 통과 전까지 변경 금지
target_min_range_m: 3.0       # 끼어든 차를 잡기 위해 10에서 낮춤
weak_min_range_m: 40.0
```

전방 ROI(`association_node.cpp:164`)도 바꾼다. 지금은 차체 좌표 ±12 m, 높이 −3~4 m라서 코너와 오르막의 원거리 차를 잘라낸다.

```cpp
const double half_width = std::max(12., .58 * x);   // 전방 ±30°
const double z_margin = .1 * std::max(0., x);       // 경사 10% 여유
keep = std::abs(y) < half_width && z > -3 - z_margin && z < 4 + z_margin;
```

늘어나는 배경 점은 맵 필터가 지운다. 처리 시간은 M6의 `processing_ms`로 확인한다.

Patchwork++의 `max_range`도 120 m가 된다. `range_stage_counts`는 12칸으로 늘린다.

### 8.2 원거리 차량 판정

원거리 근거는 자주 빠지므로 탐지 근거를 AND로 묶지 않는다.
fx≈1100이면 차 폭은 65 m에서 1 bin, 100 m에서 0.65 bin이다.

**기록과 판정은 별개다.** 확정 트랙은 판정과 무관하게 위치, 속도, 점수를 `tracks.csv`에 남긴다.

**전제 조건** (탐지 근거가 아니라 시스템 상태와 거부 조건):

- 트랙 확정
- `map_status=ok`, 트랙 위치 covered
- `map_static=false`

전제를 만족하는 `range > 35 m` 트랙의 근거를 더한다. 없는 근거는 0점이다.

| 근거 | 조건 | 점수 |
|---|---|---|
| 지속성 | 최근 10 scan 중 측정 5회 / 8회 이상 | +1 / +1.5 |
| 움직임 | `v ≥ 3 m/s`, 속도 표준편차 ≤ 1.5 m/s, `ego_status=compensated` | +3 |
| 카메라 | `0.5 · clamp(L, 0, 3)` | 최대 +1.5 |
| 형상 | 관측 폭 ≥ 1.0 m | +0.5 |
| 레이싱라인 (active일 때만) | `|n| ≤ 8 m` / `|n| > 15 m` | +1 / −2 |

- `far_score ≥ 3.0`이면 `far_vehicle=true`, `< 1.5`가 되면 false로 돌아간다.
- 35~50 m 구간의 점수는 주 대상 유지에만 쓴다 (11절).

| 상황 | 계산 | 결과 |
|---|---|---|
| 달리는 앞차, 카메라·레이싱라인 없음 | 1 + 3 | 차량 |
| 달리는 앞차, 링 1개 | 1 + 3 | 차량 |
| 정지 차량, 트랙 위, 카메라 보임 | 1 + 1 + 1.5 | 차량 |
| 정지 차량, 트랙 위, 카메라 못 봄 | 1 + 1 | 기록만 |
| 가드레일, 벽, 맵에 있는 시설물 | `map_static` | 제외 |
| 맵 없음, localization 불량 | 전제 실패 | 원거리 경로 꺼짐 |
| 맵에 없는 새 구조물, 트랙 위, 카메라 오검출 | 1.5 + 1 + 1.5 | 차량으로 판단 |

마지막 경우는 트랙 위에 실제로 있는 물체이므로 허용한다.

### 8.3 대상 범위 확대 조건

`target_max_range_m`은 bag 평가 후에만 올린다.

1. 앞차가 있는 주행 bag을 재생해 `far_vehicle` 구간을 추출한다.
2. 영상으로 라벨을 확인한다.
3. 정밀도 95% 이상, 앞차가 보이는 구간의 유지율 70% 이상이면 범위를 80 m로 올린다.
4. 같은 조건을 다시 통과하면 120 m로 올린다.

통과하지 못하면 50 m를 유지하고 원인을 기록한다.

### 8.4 조건부 작업

`range_stage_counts`에서 Patchwork++가 원거리 점을 지면으로 지운 것이 확인될 때만 아래를 추가한다.

- 70 m 밖의 앙각 −0.5° 이상 점을 비지면으로 넘긴다.
- `map_status=ok`일 때만 적용한다. 경사로 노면은 맵의 바닥 표면이 지운다.

## 9. 카메라 대응

### 9.1 C1. 연속 점수

영역 임계값과 IoU 대신 트랙이 투영된 bin의 확률을 직접 쓴다.

```cpp
struct BinSpan { int left, right; bool valid; };
BinSpan bins_for(const Projection& p);          // 차지하는 bin
double span_score(const std::array<double,40>& p, BinSpan span,
                  const std::vector<bool>& owned_by_others);
```

- 소유 구간은 확장 없이 계산한다.
- 점수를 읽을 때만 좌우 1 bin 넓힌다. 보정 오차 여유다.
- 다른 트랙이 소유한 bin은 읽지 않는다.
- 점수는 남은 bin의 최대 확률이다.

입력은 프레임별 sigmoid 확률이다. 시간 평활은 C2가 맡는다.
`camera_regions`와 EMA 확률은 화면 표시와 CSV용으로만 남긴다.

투영 대상은 트랙의 후면이다. 예측 기준점에서 폭 `Ŵ`, 높이 `Ĥ`(초기 1.5 m)인 면을 만들고 기존 `project()`로 투영한다.

### 9.2 맵 거부

측정마다 `map_static_ratio`를 계산한다.

```cpp
ratio = count(is_structure(q_map, range)) / points;
tr.static_ema = (1 - .3) * tr.static_ema + .3 * ratio;
```

측정 3회 이상이고 `static_ema ≥ 0.6`이면 `map_static=true`다.
`map_static` 트랙은 거리와 무관하게 주 대상이 되지 못하고 bin도 소유하지 못한다.

벽에 붙어 달리는 차는 후면 점이 구조물에서 떨어져 있으므로 비율이 낮다.

### 9.3 C2. 근거 누적

처리 주기마다 지난번 이후의 A₂ 프레임을 시각순으로 처리한다.

```cpp
for (const auto& frame : a2_frames_after(last_a2_stamp_, t)) {
  if (!frame.history_ready) continue;
  const auto ego = ego_history.at(frame.stamp);
  if (!ego) continue;
  auto views = project_tracks(tracks, frame.stamp, *ego, calib);  // tr.peek(), 거리순
  assign_ownership(views);                                         // C3
  for (auto& v : views) evidence.observe(*v.track, frame.stamp, v.score, v.visibility);
  last_a2_stamp_ = frame.stamp;
}
```

`last_a2_stamp_`가 같은 프레임의 중복 처리를 막는다. 프레임이 `a2s_`의 보관 범위(40개)를 넘겨 빠지면 근거만 덜 쌓인다.

```cpp
void observe(Track& tr, double stamp, double s, Visibility vis) {
  const double dt = stamp - tr.evidence_stamp;   // 초기값은 트랙 생성 시각
  tr.evidence_stamp = stamp;
  tr.visibility = vis;
  if (vis != Visibility::Visible) return;        // 가림·화면 밖: 유지
  tr.L *= std::exp(-dt / cfg.tau);
  const double delta = std::clamp(logit(s) - logit(cfg.p0), cfg.neg, cfg.pos);
  tr.L = std::clamp(tr.L + cfg.alpha * delta, cfg.Lmin, cfg.Lmax);
}
```

| 파라미터 | 시작값 | 의미 |
|---|---|---|
| `tau` | 4.0 s | 자연 감소 |
| `p0` | 0.25 | 중립 확률 |
| `alpha` | 0.15 | 프레임당 반영 비율 |
| `pos` / `neg` | 1.5 / −0.2 | 미검출은 약하게만 감점 |
| `Lmin` / `Lmax` | −3 / 8 | 상하한 |
| `L_conf` | 3.0 | 주 대상 진입 |
| `L_drop` | 1.0 | 주 대상 해제 |

21 Hz 기준 계산값 (실측 아님):

| 상황 | 결과 |
|---|---|
| p=0.6 유지 | 약 0.7초에 진입 |
| p=0.1 유지, L=8에서 시작 | 약 4.4초에 해제 |
| 1초 검출, 1.5초 미검출 반복 | L이 약 4.5~7.7 사이 유지 |
| 트랙이 카메라에 안 보임 | L 유지 |

**음의 근거를 약하게 둔 이유:** 영상에서 차가 보일 때도 A₂ 영역은 약 41%만 표시됐다.
처음 값(τ=3, neg=−0.5)은 통합 시나리오 시험에서 다음 검출 전에 대상을 놓쳐 지금 값으로 바꿨다.

**A₂ 중단:** `camera_silent_max_s`(2.0초) 동안 새 프레임이 없으면 `camera_state=silent`이고 11절에 따라 처리한다.

### 9.4 C3. 가림 처리

**목적:** 한 bin에 트랙이 여럿이어도 후보를 버리지 않는다.

```text
1. 카메라 시각의 트랙을 거리순으로 정렬한다.
2. 소유 자격이 있는 트랙이 자기 bin 구간을 차지한다.
3. 뒤쪽 트랙은 남은 bin으로만 점수를 낸다.
4. 남은 bin이 없으면 Occluded가 되어 L을 유지한다.
```

소유 자격은 확정, `vehicle_extent` 통과, `map_static=false`, 영상 band와 수직 겹침이다.
가까운 배경 조각이 먼 앞차의 근거를 가로채지 못하게 하기 위해서다.

두 트랙의 거리 차가 3 m 미만이고 구간이 70% 이상 겹치면 앞뒤를 구분할 수 없다.
이때만 `association_ambiguous=true`로 표시하고 둘 다 L을 유지한다.

## 10. 레이싱라인

입력은 `s_m,x_m,y_m,yaw_rad,kappa_radpm,v_mps,ax_mps2` 형식의 CSV다.

```cpp
class Raceline {
 public:
  static Raceline load(const std::string& csv, bool closed_loop);
  struct Frenet { double s, n, yaw; };
  Frenet project(const Eigen::Vector2d& xy, int& hint) const;  // hint<0이면 전체 탐색
  double ahead(double s_from, double s_to) const;              // 폐곡선 처리
};
```

투영은 `scc_speed_replanner.cpp`의 `ProjectToFrenet`과 같은 방식이다.

```yaml
raceline_path: ""                     # 실행 머신의 CSV 절대 경로
map_from_raceline: [0.0, 0.0, 0.0]    # x, y, yaw. 실측 전 미검증
raceline_verified: false
```

- 위치는 `MapFilterRos`의 TF와 품질 검사를 재사용한다. `active()`, `map_from_body()` 접근자를 추가한다.
- `raceline_verified=false`면 `s`, `n`을 로그로만 남기고 판단에는 쓰지 않는다.
- 자차 `|n|`의 5초 중앙값이 5 m를 넘으면 `misaligned`로 표시하고 판단에서 뺀다.

### 같은 차로 판정

앞차 판단은 원본 replanner처럼 **자차가 달리는 경로** 기준이다.
자차는 레이싱라인에서 벗어날 수 있으므로 절대 `n`이 아니라 상대값을 쓴다.

```cpp
const double lateral = std::abs(tr.n - ego.n);
const bool in_lane = lateral <= (retained ? 2.5 : 2.0);   // 원본 lane_half_width = 2.0
```

트랙 안/밖 판단(8.2절)에는 절대 `n`을 쓴다.

레이싱라인을 못 쓰면 지금 동작인 좌우 12 m를 유지한다. 코너에서는 앞차의 차체 Y가 금방 2 m를 넘기 때문이다. 영상의 `T411`도 20 m에서 좌측 약 3.6 m였다. 이 상태에서는 옆 차로 오선택 위험이 그대로 남으며, `raceline_status`로 기록한다.

## 11. 대상 선택

```cpp
struct TargetObservation {
  int64_t id; double distance, range; bool usable, measured;
  double evidence; bool in_lane, vehicle_shape, map_static, far_vehicle;
  Visibility visibility; double score;
};
struct TargetSelection { int64_t primary{-1}, standby{-1}; std::string status; double camera_age; };

class EvidenceSelector {
 public:
  TargetSelection update(double stamp, double last_camera_stamp,
                         const std::vector<TargetObservation>& obs);
};
```

`distance`는 레이싱라인 active면 `path_s_m`, 아니면 `range_m`이다.

**진입 조건 `eligible(o)`:** `usable`, `measured`, `in_lane`, `!map_static`, `target_min ≤ range ≤ target_max`이고,

- `range ≤ 50 m`: `vehicle_shape`, `L ≥ L_conf`, 카메라 정상
- `range > 50 m`: `far_vehicle` (링 1개 차량은 `vehicle_shape`를 만족할 수 없으므로 요구하지 않는다)

**유지 조건 `retained(o)`:** `usable`, `in_lane`, `!map_static`이고 다음 중 하나:

- `L ≥ L_drop`
- `far_vehicle`이고 `range > 35 m`

규칙:

1. A₂가 `camera_silent_max_s` 넘게 멈추면, `far_vehicle`이고 `range > 35 m`인 대상만 유지하고 나머지는 해제한다.
2. 주 대상이 `retained`면 유지한다.
3. 아니면 `eligible` 중 가장 가까운 트랙을 고른다.
4. 대기 후보는 `eligible` 중 다음 트랙이다.
5. **끼어들기:** 주 대상보다 가까운 `eligible` 트랙이 `switch_hold_s`(0.3초) 이상 유지되면 교체한다.

5번이 없으면 끼어든 차를 무시하고 먼 차를 계속 보게 된다.
`L_conf`와 시간 조건이 있으므로 근거 없는 근거리 배경으로 튀지 않는다 (T877 유지).

| 상태 | 의미 |
|---|---|
| `measured` | LiDAR 측정 있음, 이번 주기 `camera_matched` |
| `camera_held` | LiDAR 측정 있음, 카메라 약함·가림·화면 밖 |
| `predicted` | LiDAR 미측정 |

## 12. 시험 계획

### 유지할 회귀 시험

| 시험 | 내용 |
|---|---|
| T877 | 카메라 근거 없는 가까운 배경은 주 대상이 되지 못함 |
| 반복 프레임 | 같은 A₂ 프레임은 근거를 올리지 못함 |
| ID 유지 | 2 scan 소실 뒤 같은 ID, 속도 유지 |
| 벽 거부 | 차량 범위를 넘는 측정은 트랙을 갱신하지 못함 |
| 시각 역행 | 시간이 되돌아가면 초기화 |
| 인접 차량 | 조각 병합이 두 대를 합치지 않음 |
| 맵 필터 | 벽 앞 0.5 m 물체 보존, 맵 범위 밖 보존, 품질 저하 우회 |

### 의도적으로 바뀌는 시험

| 기존 | 변경 |
|---|---|
| 카메라 소실 0.3초 뒤 대상 해제 | 보이는데 약하면 L이 줄어 해제, 안 보이면 유지 |
| 모호한 근접 측정은 갱신 안 함 | 낮은 비용 쪽으로 갱신, `R` 4배 |
| 카메라 보조 형상 연속성 | 후면 기준점으로 대체 |
| `selection_reason=='no_horizontal_overlap'` | `camera_evidence_low` |
| `reason=='camera_matched'` | `selected`, `standby` |
| 더 가까운 트랙으로 바꾸지 않음 | `switch_hold_s` 뒤 교체 |

### 신규 시험

| 항목 | 시험 |
|---|---|
| T1 | 원 궤적 자차(0.3 rad/s), 정지 물체: 월드 속도 ≈ 0, 같은 ID |
| T1 | 30 m 앞 같은 속도 차량, 3 scan 소실 뒤 같은 ID |
| T1 | ego 0.5초 끊김에는 초기화 없음, 그 이상이면 초기화 |
| T1 | 점 시각 폭이 없으면 `ScanTiming.valid=false` |
| T2 | 차가 30° 회전: 기준점 이동 0.3 m 이하 |
| T2 | 측면만 보임: `rear_visible=false`, 속도 튐 없음 |
| T3 | 조각 2개가 측정 1개가 됨 |
| T3 | 차와 붙은 일반 물체에서 차 부분만 수집 |
| T3 | 범위 초과, 구조물 비율 초과 수집은 취소 |
| T4 | 링 1개 약한 측정이 기존 트랙을 이음 |
| T4 | 약한 측정은 맵 정상일 때만 새 트랙을 만들고, 10 scan 중 4회면 확정 |
| T4 | 미확정 약한 트랙이 `max_weak_tracks`를 넘지 않음 |
| T5 | CTRV Jacobian과 수치 미분 일치, ω→0 극한 |
| T5 | 코너 주행 차량 오차가 CV보다 작음 |
| T5 | 정지 물체와 30 m/s 물체 모두 2점 초기화로 첫 2 scan에 연결 |
| C1 | 점수가 다른 트랙 소유 bin을 읽지 않음 |
| C2 | 보이지 않는 트랙의 L 유지, p=0.6 진입 시점 |
| C3 | 앞차가 뒷차를 가림: 뒷차 L 유지, ID 유지 |
| C3 | 가까운 배경 조각이 앞차 근거를 가로채지 못함 |
| 맵 | 저 가드레일(높이 0.8 m)이 라벨 3으로 저장, 벽 라벨로는 저장 안 됨 |
| 맵 | 키프레임 2개에만 보인 voxel은 저장 안 됨 |
| 맵 | `covered()`가 맵 밖에서 false |
| 맵 | 100 m 거리에서 허용오차 확대로 벽 점 일치 |
| 맵 | 가드레일 트랙은 `map_static`으로 주 대상 거부 |
| 맵 | 가드레일 옆 0.5 m를 달리는 차는 `map_static=false` |
| 맵 | 맵 없음: 약한 트랙 생성 금지, `far_vehicle=false`, 범위 50 m 제한 |
| 맵 | 옛 PCD: `map_status=no_structure` |
| 8.2 | 상황 표의 결과와 일치 |
| 8.2 | 근거 하나가 빠져도 달리는 차는 `far_vehicle=true` |
| 8.2 | 점수가 3.0과 1.5 사이를 오갈 때 판정이 뒤집히지 않음 |
| 8.2 | `far_vehicle=false`인 확정 트랙도 `tracks.csv`에 기록 |
| 8.1 | 100 m, 좌우 40 m의 코너 차량이 ROI 안에 남음 |
| 10 | 폐곡선 시작점을 넘는 `ahead()`, 자차가 라인에서 3 m 벗어나도 앞차 `in_lane` |
| 10 | 정렬 안 된 레이싱라인은 `misaligned` |
| 11 | 끼어든 차로 0.3초 뒤 교체, 1프레임 근접 물체로는 교체 안 됨 |
| 11 | 50 m 안으로 들어온 `far_vehicle` 대상이 35 m까지 유지 |
| 11 | `target_max_range_m=50`이면 50 m 밖 트랙은 주 대상이 되지 않음 |
| M3 | 플래그 false: `valid=false`, `request` NaN, `request_unverified` 유한값 |
| M3 | 미검증 ego로 옛 모드가 유효 결과를 내지 않음 |
| M6 | 150 ms 처리 지연에서 최신 scan만 처리, `dropped_scans` 증가 |
| M6 | 렌더가 500 ms 막혀도 인식 주기 유지 |

합성 시험의 원거리 점군은 실제 링 간격을 쓴다. `far_cloud_smoke.py`의 0.42 m 간격을 거리별 1° 간격으로 고친다. 99 m 시험은 링 1개가 된다.

### bag 평가

`cpp/tools/track_eval.py`는 `tracks.csv`, `association.csv`를 읽어 아래를 낸다.

- 주 대상 ID 전환 횟수, 대상 유지 구간 길이 분포
- 앞차 존재 라벨 대비 유지율과 오선택률
- `camera_state`, `map_status`별 시간 비율
- 거리대별 점 수와 측정률, 맵이 지운 점 수
- scan당 처리 시간 p50, p95, `dropped_scans`

라벨은 CSV(`시작초,끝초,앞차있음`)로 손으로 만든다.

## 13. 구현 순서

| 단계 | 내용 | 통과 조건 |
|---|---|---|
| 0 | M1, M4, M6 | bag 재생 동작, band 포화율 5% 이하 |
| 1 | M2 | 재투영 오차 1 bin 이하 |
| 2 | M3 | 미검증 값이 CSV에 기록 |
| 3 | 맵 라벨, 커버리지, 시각 보정 (6절) | 주행 bag에서 가드레일 트랙이 `map_static` |
| 4 | T1, T5 | 회전 시험 통과, ID 전환 감소 |
| 5 | T2, T3, T4 | 조각·가드레일 시험 통과 |
| 6 | C1~C3, M5, 레이싱라인, 선택기 | 매칭 유지율 증가, T877 유지 |
| 7 | 원거리 ROI, 기록 | 8.3 평가 시작 |

각 단계는 이전 단계의 bag을 재생해서 비교한다. 6단계는 1단계 통과가 전제다.

수치 목표 (첫 bag 후 재조정):

| 지표 | 영상 기준 현재 | 목표 |
|---|---|---|
| 차가 보일 때 매칭 유지율 | 약 7.5% | 70% 이상 |
| 처리 시간 p95 | 미측정 | 80 ms 이하 |
| 인식 주기 | 약 6.7 Hz (화면 기준) | 10 Hz |
| 주 대상 오선택률 | 미측정 | 라벨 구간의 2% 이하 |

## 14. 위험

| 위험 | 대응 |
|---|---|
| M2 없이 C1이 틀린 위치의 확률을 읽음 | 6단계는 1단계 통과 후 시작 |
| GNSS odom 품질 미검증 | `ego_status` 기록, bag에서 정지 물체 속도로 확인 |
| 움직임 점수가 ego 오차로 벽에 붙음 | 맵 거부가 먼저 적용, 속도 표준편차 조건 |
| 지도 제작 때 주차된 차가 구조물로 저장 | 키프레임 조건으로 못 거름. 그 위치 정차 차량 거부를 한계로 기록 |
| localization 저하 시 맵 판단 불가 | 원거리 경로 꺼짐, 50 m 일반 경로 유지 |
| 레이싱라인 미검증 동안 옆 차로 오선택 | 지금과 같은 위험. `raceline_status`로 기록 |
| CTRV가 정지·저속에서 불안정 | `v ≥ 0`, heading 사전값, ω 제한 |
| ROI 확대로 처리 시간 증가 | M6 측정, p95 초과 시 ROI 축소 |
| `MultiThreadedExecutor`로 공유 상태 경쟁 | 렌더 쪽 공유는 `RenderSnapshot` 하나로 제한 |
| A₂ 확률 분포가 시작값과 다름 | 파라미터 전부 설정값, 첫 bag으로 재조정 |
| 이 Mac에서 ROS·Eigen 시험 불가 | Nuvo에서 실행 |

## 15. 구현 전 확인 사항

| 항목 | 막히는 단계 |
|---|---|
| `yongin_raceline.csv`가 `track_merged` 맵과 같은 좌표인지 | 6 |
| `KeyframePointCloudMap`의 키프레임별 점 접근 가능 여부, 안 되면 `.simplemap` 사용 | 3 |
| localization 저장소의 `test_surface_cache.py`와 manifest 검사가 v2 캐시를 받는지 (이 저장소에 없음) | 3 |
| `/a1_gnss/odom` 발행 주기와 지연 | 4 |
| 점 `timestamp`가 방위각에 따라 변하는지 (`use_timestamp_type: 1`) | 3 |
| `/tf`, `/lidar_odometry/pose_quality`가 bag 재생에서 재현되는지 | 0 |
| XT32 `ring` 각도표와 리턴 모드 | 7 |
| 카메라 왜곡 크기 | 1 |

## 16. 구현 상태

### 코드 반영 완료

M1 기본값, M3, M4 스크립트, M5, M6, T1~T5, C1~C3, 원거리 입력과 점수, 선택기, 레이싱라인 모듈, 맵 허용오차·커버리지·시각 보정, 평가 도구.

### 현장 작업이 필요한 항목

| 항목 | 이유 |
|---|---|
| 구조물 라벨 내보내기 | 키프레임별 점 접근 API를 Nuvo에서 확인해야 한다. 그 전까지 `map_status=no_structure`이고 원거리 경로는 꺼져 있다 |
| M1 최종 노출값 | 현장 촬영 |
| M2 보정값 | 체커보드와 표적 측정. `docs/CALIBRATION_KR.md` 참고 |
| `bumper_x_m`, 검증 플래그 | 실측 |
| `raceline_path`, `map_from_raceline`, `raceline_verified` | 실행 머신의 파일과 좌표 정렬 확인 |
| 파라미터 조정, `target_max_range_m` 확대 | 주행 bag 평가 |

### 검증 수준

- ROS 없는 라이브러리 시험은 프로젝트가 쓰는 Eigen 기능만 구현한 임시 심(저장소 밖)으로 실행했다.
- ROS·PCL·OpenCV 코드는 스텁 헤더로 문법만 확인했다. 메시지 필드는 실제 `.msg`에서 생성했다.
- 실제 Eigen·ROS 빌드, 스모크 시험, 실차 성능은 미검증이다.

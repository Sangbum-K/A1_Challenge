# 추적·대응·대상 선택 교체 구현

- 시각: 2026-11-06 02:54 ~ 03:30
- 커밋: 없음. 작업 폴더에만 있다
- 근거: Claude Code 세션 기록, 커밋(`1a42852`)과 작업 폴더의 파일별 비교
- 기준 문서: [설계 문서](../superpowers/specs/2026-11-06-sensor-fusion-tracking-association-design.md)

## 요약

트래커와 카메라 대응을 새로 쓰고, 대상 선택을 근거 누적 방식으로 바꿨다. 녹화·재생·평가 도구도 추가했다.
**합성 시험만 통과했다.** 실제 Eigen·ROS 빌드, 스모크 시험, 실차는 하지 못했다.
가드레일 거부와 원거리 경로는 코드만 있고 아직 꺼져 있다.

## 동작 변화

| 항목 | 이전 | 이후 |
|---|---|---|
| 기본 노출 | 35,000 µs | 3,000 µs |
| 미검증 값 | 계산 전에 종료, 기록 없음 | 항상 계산해 기록. `valid=false`, `request=NaN` 유지 |
| 트랙 좌표 | 차체 좌표 | 연속 odom 좌표 (`ego` 소실 시 차체 좌표로 복귀) |
| 운동 모델 | CV | CTRV (`[x, y, v, ψ, ω]`) |
| 추적 기준점 | 박스 중심 | 후면 기준점 |
| 측정 할당 | Hungarian 1회 | 3단계 (강한 측정 → 약한 측정 → 미확정 트랙) |
| 트랙 확정 | `hits >= 3` | 최근 8 scan 중 3회 |
| 출력 유지 | 소실 0.3초 | 소실 0.5초 이내, 위치 분산 1 m² 이하 |
| 트랙 삭제 | 소실 0.8초 | 소실 1.5초 또는 위치 분산 합 4 m² 초과 |
| 카메라 대응 | 영역 임계값, IoU ≥ 0.15, 1:1 매칭 | 트랙 투영 bin의 확률을 직접 읽어 누적 |
| 같은 bin의 여러 트랙 | 후보를 버림 | 가까운 트랙이 소유, 뒤 트랙은 가림 상태로 근거 유지 |
| 카메라 미검출 | 0.3초 뒤 대상 해제 | 근거 L ≥ 1.0이면 유지 |
| 더 가까운 차 등장 | 교체 안 함 | 조건을 갖춰 0.3초 유지되면 교체 |
| 처리 거리 | 100 m | 120 m (기록용) |
| 주 대상 거리 | 10~50 m | 3~50 m. 상한은 bag 평가 후에만 확대 |
| LiDAR 대기열 | 오래된 프레임부터 처리 | 0.1초 지난 것 중 최신만 처리 |
| 렌더 | 인식과 같은 스레드 | 전용 스레드, 10 Hz, 최신 스냅샷 |
| 간격 계산 | 박스 중심 − 길이/2 | 후면 모서리의 차체 X + LiDAR 위치 − 범퍼 |
| 상대속도 | 겉보기 속도 − yaw rate 보정 | world 속도 차를 차체 축으로 변환 |

## 구현 항목

### Must fix

| 항목 | 내용 | 위치 |
|---|---|---|
| M1 | 노출 기본값 변경. `A1_CAMERA_EXPOSURE_US`로 변경 가능 | `run_ui.sh` |
| M3 | `Ego.verified` 추가. `valid`는 값을 쓸 수 있음, `verified`는 장착·시각 검증 완료를 뜻한다. `request_unverified`, `speed_unverified_mps` 기록. UI는 `UNVERIFIED` 표시 | `core.*`, `fusion_node.cpp`, 메시지 |
| M4 | `A1_RECORD_BAG=1` 녹화, 빈 공간 64 GB 미만이면 거부. `run.sh replay <bag> [--rerun-a2]`로 재처리 | `run_ui.sh`, `run.sh` |
| M5 | `primary_status`가 `measured` 또는 `camera_held`이면 유효. 카메라 프레임별 조건 삭제 | `fusion_node.cpp` |
| M6 | `dropped_scans` 집계, `RenderSnapshot`, 전용 렌더 스레드. BEV 점 샘플링 제거 | `association_node.cpp` |

M2(보정)는 측정이 필요해 코드 준비만 했다. `calibration.yaml`로 기하 값을 모으고, `project()`에 왜곡 계수 `k1, k2, p1, p2`를 추가했다. 절차는 [CALIBRATION_KR.md](../../sensor_fusion/docs/CALIBRATION_KR.md)에 적었다.

### 추적: `tracker.cpp` (386줄, 신규)

| 항목 | 내용 |
|---|---|
| T1 자차 보상 | `EgoHistory`, 방위각별 시각표 `ScanTiming`, 전환 관리 `EgoLink`. 점 시각 폭이 5 ms 미만이면 scan 시각을 쓴다 |
| T2 후면 기준점 | `rear_anchor()`. 후면이 안 보이면 종방향 측정 비중을 사실상 없앤다 |
| T3 트랙 주도 수집 | 예측 위치의 박스 안 점을 모아 측정 하나로 만든다. 크기 초과나 구조물 비율 0.6 이상이면 취소 |
| T4 3단계 할당 | 약한 측정으로 기존 트랙을 잇는다. 약한 측정의 새 트랙은 맵 정상일 때만, 10 scan 중 4회면 확정, 최대 30개 |
| T5 CTRV·수명 | Joseph form 갱신, 2점 초기화, 공분산 기반 수명 |

이 교체로 전날 추가한 `CameraSupport`, `shape_noise`, `assisted` 비용을 삭제했다. 후면 기준점이 같은 문제를 더 직접 다루기 때문이다. `horizontal_iou`, `unique_region_matches`, `select_primary`, `CameraTargetSelector`도 함께 삭제했다.

### 카메라 대응: `camera_evidence.cpp` (124줄, 신규)

| 항목 | 내용 |
|---|---|
| C1 연속 점수 | 트랙이 투영된 bin의 프레임별 확률을 읽는다. 다른 트랙 소유 bin은 읽지 않는다 |
| C2 근거 누적 | A₂ 프레임마다 트랙을 그 시각 위치로 옮겨 근거 L을 갱신한다. 같은 프레임은 두 번 반영하지 않는다 |
| C3 가림 처리 | 거리순으로 bin을 소유한다. 남은 bin이 없으면 Occluded로 L을 유지한다 |
| 대상 선택 | `EvidenceSelector`. 진입 L ≥ 3.0, 해제 L < 1.0. A₂가 2초 멈추면 원거리 대상 외에는 해제 |
| 원거리 판정 | 점수 합산 3.0 진입, 1.5 해제. 전제는 트랙 확정, 맵 정상, 비구조물 |

### 기타

- `raceline.cpp` (86줄, 신규): CSV 로드, Frenet 투영, 폐곡선 처리, 정렬 감시(`AlignmentMonitor`)
- `gnss_ego.hpp` (신규): `/a1_gnss/odom` 해석을 두 노드가 공유
- `map_surfaces.*`: 거리별 허용오차(`0.30 + 0.003·거리`), `covered()`, `is_structure()`
- `map_structure.*` (신규): 키프레임별 구조물 집계 로직. 아직 내보내기에 연결하지 않음
- `association_node.cpp` (+537/−248): 파이프라인 연결, ROI를 전방 ±30°와 경사 여유로 확대, 거리별 최소 점 수
- `track_eval.py` (115줄, 신규): 주 대상 전환, 유지율, 처리 시간 등을 bag 결과에서 계산
- 메시지: `ClusterTrack`, `Association`, `BrakeEstimate`, `EgoMotion` 확장. `apparent_velocity_mps`, `horizontal_iou` 삭제
- 문서: [CALIBRATION_KR.md](../../sensor_fusion/docs/CALIBRATION_KR.md) 신규. `ASSOCIATION_KR.md`, `TRACK_CONTINUITY_KR.md`, `F2_FUSION_DESIGN_KR.md`, `UI_KR.md`에 변경 안내 추가

## 주요 파라미터

모두 첫 bag으로 조정할 시작값이다. 전체 목록은 `cpp/config/association.yaml`에 있다.

| 파라미터 | 값 | 의미 |
|---|---|---|
| `processing_range_m` | 120 | 추적과 기록 범위 |
| `target_max_range_m` | 50 | 주 대상 상한. 8.3절 평가 전 변경 금지 |
| `target_min_range_m` | 3 | 주 대상 하한 |
| `weak_min_range_m` | 40 | 약한 측정을 받는 최소 거리 |
| `evidence_tau_s` | 4.0 | 근거 자연 감소 |
| `evidence_step_min` | −0.2 | 미검출 감점 |
| `evidence_confirm` / `evidence_release` | 3.0 / 1.0 | 주 대상 진입 / 해제 |
| `target_switch_hold_s` | 0.3 | 끼어들기 교체 대기 |
| `camera_silent_max_s` | 2.0 | A₂ 중단 판정 |

## 시험

테스트를 먼저 작성해 기능 부재로 실패하는 것을 확인하고 구현했다.

| 시험 | 결과 | 비고 |
|---|---|---|
| `core`, `association`, `tracker`, `camera_evidence`, `raceline` | 통과 | Eigen 대신 임시 심(저장소 밖)으로 컴파일 |
| `test_track_eval.py` | 통과 (2건) | 폴더 정리 후 재실행해도 통과 |
| 합성 통합 시나리오 | 130/130 스캔 유지, ID 전환 0회, 레일 선택 0회 | 코너, A₂ 약 40% 검출, LiDAR 소실, 정적 레일 |
| ROS 노드 4개와 시험 코드 | 문법 검사만 통과 | 스텁 헤더 사용, 메시지 필드는 실제 `.msg`에서 생성 |
| 쉘 스크립트 | 문법 통과 | |

시나리오는 직접 만든 합성 데이터이고 fx=1100을 가정하므로 실제 정확도와 무관하다. 시험의 합격 기준은 유지율 95% 이상, ID 전환 0회다.

유지하기로 한 회귀 시험 중 T877(카메라 근거 없는 근거리 배경은 주 대상 불가)도 포함돼 있다.

## 시험이 잡아낸 문제

| 문제 | 원인 | 수정 |
|---|---|---|
| 같은 차에 ID가 둘 생김 | 교체 과정에서 기존 보호 코드가 빠짐 | 확정 트랙 게이트 안의 측정은 새 트랙을 만들지 않게 복구 |
| 벽 조각이 크기 검사를 통과 | 최소면적 사각형에서는 벽 조각이 얇게 보임 | 트랙 축 기준 폭 검사 추가 |
| A₂ 40% 조건에서 대상을 놓침 | 첫 값 τ=3, neg=−0.5의 감소가 너무 큼 | τ=4, neg=−0.2로 변경. 카메라가 완전히 사라지면 약 4.4초 뒤 해제 |
| 끊긴 카메라가 2초간 매칭으로 보임 | `camera_matched` 판정이 시간을 보지 않음 | 최근 0.3초 프레임만 인정 |

## 설계와 달라진 점

- `MultiThreadedExecutor` 대신 전용 렌더 스레드를 쓴다. OpenCV 창은 스레드가 고정돼야 하기 때문이다.
- `near_surface_x_m`을 후면 양쪽 모서리 중 가까운 쪽의 차체 X로 단순화했다.
- `bag_qos_override.yaml`은 복사가 아니라 21FPS 원본에 재생 토픽 설정을 더한 것이다.
- 설계 12절의 M6 시험(150 ms 지연, 렌더 500 ms 차단)은 작성하지 않았다.

## 아직 켜지지 않는 기능

| 기능 | 이유 | 켜는 조건 |
|---|---|---|
| 가드레일 거부 (`map_static`) | 구조물 라벨 내보내기 미연결. 키프레임별 점 API를 이 Mac에서 확인할 수 없어, 추측으로 쓰면 Nuvo 빌드가 깨질 수 있다 | API 확인 후 연결, 맵 캐시 재생성 |
| 원거리 경로 | 위와 같음. 그때까지 `map_status=no_structure` | 위와 같음 |
| 레이싱라인 판단 | `raceline_path`가 비어 있고 `raceline_verified=false` | CSV 경로 입력, 좌표 정렬 확인 |
| 카메라 대응의 실제 효과 | fx가 틀리면 틀린 bin을 읽는다 | M2 보정 |
| `valid=true` P5 | 검증 플래그가 모두 false | 실측 후 |

## 남은 위험

- 지도 제작 때 주차돼 있던 차는 구조물로 남는다. 그 자리에 다시 정차한 차는 거부된다.
- 레이싱라인이 없으면 좌우 12 m 기준이라 옆 차로 오선택 위험이 그대로다.
- `bumper_x_m`이 0이라 간격이 실제보다 크게 계산될 수 있다.
- GNSS odom 품질이 나쁘면 벽이 움직이는 것처럼 보일 수 있다. bag에서 정지 물체 속도로 확인해야 한다.
- 이 코드는 실제 컴파일러와 라이브러리로 빌드된 적이 없다. Nuvo에서 빌드 오류가 날 수 있다.

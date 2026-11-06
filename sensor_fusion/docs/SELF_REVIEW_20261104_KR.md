> 후속 구현: [추적 연속성 및 같은 ID P5 연결](TRACK_CONTINUITY_KR.md). 아래 표는 변경 전 검토 기록이다. 주행라인 연동은 보류.

# 전체 작업 방향 및 코드 자체 점검 — 2026-11-04

## 목표와 이번 변경

**10~50m의 선행차량 후보 최대 2대**를 관측한다. 카메라는 존재/bin 방향 근거,
LiDAR는 거리·겉보기 상대속도·추적 ID를 담당한다. 실차 명령은 계속 제외한다.
거리 기준은 LiDAR 원점에서 필터된 군집 중심까지 수평거리 `sqrt(x²+y²)`다.
차량 범퍼 간격과 다르므로 향후 P5 입력으로 그대로 확정하면 안 된다.

이번 코드 변경:

- 단일 최고 bin에서 **분리된 연속 bin 영역 전체**를 추출하도록 변경. 각 영역은 이전 영역과 겹칠 때만 유지 임계값을 적용한다.
- 각 LiDAR 군집과 각 카메라 영역의 IoU 관계를 검사한다. 서로 유일한 일대일 대응만 camera_matched로 채택한다.
- 서로 다른 두 영역의 두 차량은 더 이상 단순히 차량 수가 2라는 이유로 모호 처리하지 않는다.
- 한 영역에 두 군집이 겹치거나 한 군집이 두 영역에 겹치면 모호 상태를 유지한다. 가려진 두 차량을 임의로 분리하지 않는다.
- 후보 거리 10~50m, 최대2대. 내부 장면 추적은 유지하고 후보 출력/기본 UI 박스만 제한한다. debug 모드는 진단용 전체 군집을 표시할 수 있다.
- 대응된 후보를 우선 표시하고 남은 슬롯에 LiDAR 단독 후보를 채운다. 같은 종류에서는 거리순, 동거리에서는 ID순이다. 이는 표시 선택 규칙이며 최종 제동 우선순위가 아니다.
- 빨강=카메라+LiDAR 대응 후보, 주황=LiDAR 단독 기하 후보. 주황색은 차량으로 확정한 결과가 아니다.
- Association에 candidate_track_ids, lidar_candidate_track_ids, association_ambiguous 추가.
  ClusterTrack에 lidar_candidate, camera_matched, matched_bin_left/right, range_m 추가.
- 기존 candidate_unique/candidate_track_id/verified_match는 단일 대상 의미를 유지한다. 두 후보는 배열을 읽어야 한다. 기존 소비자와 새 메시지 정의를 섞지 말고 재빌드한다.
- UI 대상 카드에 최대 두 ID·거리·겉보기 Vx 표시. IMU/P5 카드는 기존 유효성 검사를 유지한다.

10m 미만과 50m 초과는 이 실험의 후보에서 제외된다. 이는 그 영역에 장애물이 없다는 뜻이 아니다.
전체 전방 점군과 지면 제거 문맥은 유지한다. 50m 성능이 검증되었다는 의미도 아니다.

## 권장 센서 우선순위

| 상황 / 값 | 권장 역할 | 이번 코드 상태 |
|---|---|---|
| 거리, 상대속도, ID | LiDAR 측정·연속 추적 우선 | LiDAR 사용, 아직 자차 회전 보상 없음 |
| 차량 존재, 방위 | 카메라 A₂ 존재/bin으로 확인 | 독립 영역 일대일 대응 구현 |
| 카메라 순간 미검출 | LiDAR 후보는 표시하되 확정 상태와 구분 | 주황 LiDAR-only, 신규 camera_matched 금지 |
| LiDAR 미검출 | 카메라만으로 거리·속도 생성하지 않음 | 카메라 metric 채널 사용 안 함 |
| 겹침 / 시간 불일치 | 대응 불확실 표시, 확정값 생성 금지 | 모호 상태 / stale 거부 |
| 최종 제동 대상 | 주행 경로 안의 충돌 가능성으로 결정 | 미구현, 현재 거리순 표시와 구분 |

카메라를 매 프레임 절대 승인자로 두면 기존 관측처럼 존재 확률이 약한 순간 결과가 사라진다.
반대로 LiDAR만으로 차량을 확정하면 벽·가드레일·식생이 섞인다. 따라서 두 센서에 동일한
일괄 우선순위를 주기보다 위처럼 측정 종류와 확인 상태를 분리하는 것이 현재 프로젝트에 맞다.
기존에 카메라로 확인했던 트랙을 짧게 유지하는 별도 상태(예: 0.3초 grace)는 후속 항목이다.
지금은 미검출 상태를 빨간 확정 후보로 유지하지 않는다.

## 중요도 순 자체 점검 결과

| 우선 | 문제 / 근거 | 개선 및 완료 조건 |
|---|---|---|
| 1 | association과 fusion_node가 서로 다른 트래커를 사용함. 붉은 ID가 P5 대상이라는 보장이 없음 | 단일 TargetState 계약(ID, 시각, 거리 정의, 속도, 공분산, 상태)을 만들고 P5 입력을 통합. 현재 연결 완료로 주장 금지 |
| 1 | 카메라 내·외부 보정과 시간 offset 미측정 | 체커보드/공통 타깃 등으로 보정하고 10/20/30/40/50m 투영 오차·시간차 기록. verified 플래그는 계속 false |
| 1 | 원시 GNSS IMU 없음, 속도 적분의 초기값·바이어스·중력·장착 검증 미완료 | 실제 IMU 수신부터 검증. 가속도를 속도로 부르지 않고 장시간 적분 drift를 관리 |
| 2 | 고정 군집 간격 0.45m, 최소5점, 크기 최소 .6/.5/.5m. 먼 차의 부분 표면은 탈락 가능 | 거리별 점 간격 기반 적응형 연결/최소 점수, 부분 관측 품질 등급. 과병합률과 누락률 함께 평가 |
| 2 | 축 정렬 폭 2.6m 상한으로 회전 차량 탈락 가능; 관측된 초록차 폭 약2.83m | 회전 박스(OBB) 또는 주축 기반 크기 평가. 벽을 막는 조건을 함께 검증한 뒤 교체 |
| 2 | 40bin의 원거리 각도 분해능과 기존 A₂ 학습 특성 | 두 개 영역 추출이 두 차량 정확도 보장을 뜻하지 않음. 근접한 두 차량/가림/같은 bin 케이스는 모호 유지 |
| 2 | 센서 좌표에서 CV 속도 추정, 자차 yaw·scan motion 무보상 | 시간 동기된 ego motion으로 deskew와 상대속도 보상. 곡선 주행/요철 속도 스파이크 비교 |
| 2 | 거리순 최대2대는 주행 경로 기반 선택이 아님 | 경로/차선 corridor와 도달시간·접근률·ID 유지로 선정. 근거 없는 TTC 브레이크 연결 금지 |
| 3 | 경계 10/50m, IoU, 세 번째 후보 출현 시 표시 선택이 바뀔 수 있음 | 실측 후 경계 hysteresis/교체 지연 및 ID 유지 점수 도입. 이번 제한은 정확한 거리 경계 사용 |
| 3 | 현재 녹화는 UI MP4, 원시 동기 센서 전체 저장을 보장하지 않음 | 재현용 제한 용량 rosbag + 설정/모델 hash/시각 로그 설계. 녹화 보존 정책은 사용자와 결정 |
| 3 | 큰 과거 데이터가 reference/datasets에 중복 복사됨 | SHA256 비교 후 복사본만 삭제, 원본 연결. runtime 모델/LibTorch/드라이버/MP4 유지 |

## 폴더별 방향

- `cpp/src`, `cpp/include`: 실제 C++ 런타임. association은 관측/대응, core는 수치 규칙, fusion은 legacy shadow 경로다. 다음 큰 작업은 둘의 대상 계약 통합.
- `cpp/config`, `cpp/launch`, `run_ui.sh`: 실행의 기준 설정. 원커맨드 domain107/localhost 격리와 MP4 유지. 장치/도메인 점유 검사를 유지한다.
- `cpp/tests`, `cpp/evidence`: 경계·다중 대응·모호성·stale 회귀 시험 및 근거. 합성 시험과 실차 정확도를 구분한다.
- `drivers`, `third_party`: 센서 드라이버 및 Patchwork++ 원본. 팀원 워크스페이스를 수정하지 않는다.
- `cpp/models_gpu`, `vendor/libtorch`, `install_gpu`, `install`: 실행 의존성. CPU/driver fallback 경로도 있으므로 이름만 보고 삭제하지 않는다.
- `reference`, `datasets`: 원본 비교 및 재생 데이터. 중복 대용량 경로는 링크로 대체하므로 프로젝트 폴더만 다른 PC로 복사하면 해당 데이터는 따라가지 않는다.
- `outputs`: 실험 증거와 MP4. 이번 정리에서 실주행 기록은 삭제하지 않는다.
- `SENSOR_ENVIRONMENT_KR.md`: 실행 환경 기준 문서. PROJECT_CONTEXT의 과거 기록과 최신 변경을 구분한다.

## 참고 구현과 적용 범위

- [Patchwork++ 공식 저장소](https://github.com/url-kaist/patchwork-plusplus): 현재 지면 제거 구현. 차량 분류기는 아니다.
- [PCL Euclidean Cluster Extraction](https://pointclouds.org/documentation/tutorials/cluster_extraction.html): 현재 KD-tree 거리 기반 군집 방식의 참고. 임계 거리가 작으면 분할되고 크면 인접 물체가 합쳐지는 trade-off를 실측해야 한다.
- [SORT 공식 저장소](https://github.com/abewley/sort): Kalman/Hungarian 추적 개념 참고. 현재 코드는 LiDAR BEV에 맞춘 자체 C++ 구현이며 SORT를 그대로 실행하는 것이 아니다. 원본도 가림/재진입 한계를 명시한다.

이번 변경은 추론이며 위 프로젝트가 본 차량의 설정값을 보증하지 않는다.

## 이번 검증 결과 / 정리 완료

- `cpp/evidence/multi_target_core.txt`: core_tests, association_tests 2/2 통과. 두 bin 영역, 독립 hysteresis, 모호한 대응 거부, 10/50m 경계 포함.
- `cpp/evidence/multi_target_build.txt`: GPU 실행파일·새 ROS 메시지 전체 빌드 및 install_gpu 설치 성공.
- `cpp/evidence/multi_target_ros.txt`: domain109 합성 ROS 시험 통과. 단일/두 차량 대응, 같은 bin의 두 군집 거부, 세 차량 중 최대2대, 10~50m 후보 범위, CSV 두 ID 저장, NaN camera metric 무시, stale 거부, 1600×860 UI 확인.
- 변경된 진단 Python 파일 구문 확인 통과. 카메라 모델이나 가중치는 바꾸지 않았으며 새 코드의 실주행 검출률·50m 정확도·처리량은 이번에 측정하지 않았다.
- 원커맨드는 새 GPU 설치본 사용. CPU 개별 실행본은 새 메시지 소비 전에 `bash run.sh build`로 재빌드해야 한다.
- UI 현재 중지 상태. 센서를 자동 재시작하지 않았다. 다음 원커맨드 실행에 새 설정 적용.
- 작업폴더 약71GiB → 5.7GiB. 중복 데이터와 설치 ZIP 69,214,335,766 bytes(약69.2GB) 제거.
- 과거 출력 171파일, rosbag 2파일, LiDAR 시각화 17파일을 SHA256 대조 후 원본으로 상대 심볼릭 링크 연결. 기존 COPY_MANIFEST의 콘텐츠 경로는 계속 접근 가능. (2026-11-06 정정: `COPY_MANIFEST.json`은 삭제했다. git 이력 `86042bb`.)
- 실행 모델·LibTorch 압축 해제본·드라이버·코드·실주행 MP4 보존. 정리 내역은 `docs/cleanup/20261104.json`.

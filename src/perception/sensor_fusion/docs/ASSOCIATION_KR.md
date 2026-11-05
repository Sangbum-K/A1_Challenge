# 카메라–LiDAR 동일 대상 검증

현재 우선 목표는 제동 계산보다 먼저 영상의 A₂ 존재/bin과 LiDAR 군집이 같은 물체를
가리키는지 관찰하는 것이다. 실행 코드는 C++ `association_node`이며 제동/CAN을 발행하지 않는다.
기존 `fusion_node`의 제동 경로와 분리돼 있으며, 검증을 마친 뒤 검출 결과를 연결한다.

## 처리 경로와 참고 구현

1. **Patchwork++**: KAIST 공식 C++ 소스를 직접 컴파일해 지면/비지면 분리.
   [공식 저장소](https://github.com/url-kaist/patchwork-plusplus), [IROS 2022 논문](https://arxiv.org/abs/2207.11919).
   upstream 코드는 변경하지 않았고 BSD-2-Clause 라이선스와 고정 commit을 `third_party/SOURCES.md`에 기록했다.
   XYZ를 입력하므로 intensity를 요구하는 RNR은 껐다. 지면 높이 초기값 1.8m는 실측값이 아니다.
2. **Voxel + Euclidean clustering**: PCL 구현을 사용하고 Autoware의
   [클러스터링 구조](https://docs.autoware.org/main/tutorials/integrating-autoware/tuning-parameters-and-performance/tuning-parameters/perception-tuning/)를 참고했다.
   16cm voxel, KD-tree 거리 65cm, 최소 5점, 전방 1~70m/좌우 12m ROI가 초기값이다.
   희박한 잔여 지면이 물체들을 이어 붙이지 않도록, 인접한 1m 격자에 지면 지지가 3개 이상
   있을 때 지역 지면 높이 중앙값으로부터 15cm 미만인 점을 추가 제외한다. 이 후처리는 자체 adapter다.
   이 방식도 경사/연석/물체의 과분할·합쳐짐에 대한 실측 검증이 필요하다.
3. **추적**: [SORT 논문](https://arxiv.org/abs/1602.00763)의 CV Kalman + Hungarian 할당을
   LiDAR BEV 상태 x,y,vx,vy에 맞춰 자체 구현했다. Mahalanobis/거리/크기 gate,
   3회 관측 확인, 0.5초 소실 만료, 모호한 근접 후보 거부를 사용한다.
   원본 SORT 소스를 복사하거나 Autoware tracker 전체를 포팅한 것은 아니다.
4. **대응**: [Autoware ROI cluster fusion](https://autowarefoundation.github.io/autoware_universe/main/perception/autoware_image_projection_based_fusion/docs/roi-cluster-fusion/)의
   3D 군집→영상 투영 방식을 참고한다. 다만 A₂는 YOLO 2D 박스가 아닌 가로 bin이므로
   3D 상자 8개 꼭짓점 투영 → 가로 IoU + 모델 영상 띠(y=235..485) 겹침을 사용한다.
   같은 bin에 여러 확정 트랙이 들어오면 `ambiguous_clusters`; 유일한 경우만 후보 ID를 낸다.

카메라에서 읽는 채널은 ch0 존재 logit뿐이다. ch3 상대속도/ch6 거리 및 다른 metric은 사용하지 않는다.
기하학적 군집에 차량이라는 학습 분류가 생기는 것은 아니므로 화면의 상자는 **후보 물체**다.
현재 검증 모드는 전체 0~39 bin과 좌우 12m를 살핀다. 최종 주행의 선행차/주행경로 선택은 별도 단계다.

## 사용자 제공 장착값

- LiDAR -Y = 차량/카메라 전방, LiDAR +Z = 위쪽.
- LiDAR 원점의 차량 정렬 좌표: X=-LiDAR Y, Y=LiDAR X, Z=LiDAR Z.
- 카메라 위치: X=+0.80m, Z=-0.30m. 좌우 Y=0m는 아직 가정.
- 카메라는 정면/수평이라는 초기 가정. 미세 roll/pitch/yaw, 실제 fx/fy/cx/cy,
  렌즈 왜곡은 아직 측정되지 않았다. fx=fy=640은 보정 완료값이 아닌 초기 화면용 값이다.

따라서 `calibration_verified=false`, `clock_verified=false`를 유지한다. 후보가 한 개여도
`verified_match=false`다. 육안으로 한 프레임이 맞는 것만으로 보정 플래그를 켜면 안 된다.
`shadow.yaml`의 이전 제동 경로도 nominal 위치/방향만 반영하고 검증 플래그는 false다.

## 시각/운동 처리

LiDAR 점별 timestamp가 header와 일관되면 scan-end를 짝짓기 기준으로 사용한다.
카메라 추론 완료를 기다리도록 LiDAR 입력을 100ms 지연 큐에 보관하고, 가까운 A₂를 선택한 뒤
**그 A₂와 동일한 영상 header**를 찾아 그린다. 영상↔A₂ 차이는 5ms 미만,
LiDAR↔영상/A₂ 차이는 각각 80ms 이내여야 한다. 없는 프레임을 보간해서 대응 성공으로 처리하지 않는다.
이 짝짓기는 하드웨어 clock 동기화 또는 운동 deskew를 완료했다는 뜻이 아니다.

GNSS IMU가 없어도 정지 장면의 대상 대응을 점검할 수 있도록 이 노드는 항법 입력과 독립적이다.
추적 속도는 차체/센서 좌표에서 본 `apparent_velocity_mps`로 명명한다.
자차 회전 보상/scan deskew가 없으므로 이 값을 검증된 선행차 절대속도로 쓰지 않는다.

## 실행과 화면

```bash
cd A1_Challenge-main/A1_SensorFusion_CPP
bash run.sh build-gpu
ROS_DOMAIN_ID=93 ROS_LOCALHOST_ONLY=1 bash run.sh association-check --seconds 45
# 실제 모니터에 창도 띄우기:
ROS_DOMAIN_ID=93 ROS_LOCALHOST_ONLY=1 bash run.sh association-check --seconds 300 --show-window
```

드라이버·GPU A₂·검증 노드를 실행하고 시간이 끝나면 자신이 시작한 프로세스만 종료한다.
기존 같은 ROS 도메인에 발행 노드가 있으면 임의로 섞지 않고 중단한다.
검증 runner는 어두운 현재 장면을 위해 자동 노출 1~20ms/자동 gain 상한 12dB를 사용한다.
촬영 조건 변경은 모델 인지 정확도 검증과 별개다. 21Hz 벤치마크의 기존 수동 노출 설정은 유지했다.

`outputs/association_check/latest.jpg`: 좌측 영상에 투영된 트랙 상자/ID, 노란 카메라 bin 영역,
하단 존재 점수. 우측 BEV에 같은 색상/ID의 LiDAR 군집과 지면을 표시한다.
`association.csv`, `frames.json`, `summary.json`: 시각 차이, 트랙 좌표/크기/가로 IoU,
유일 후보/모호함/보정 미완료 등의 상태와 처리 시간을 기록한다.

ROS 출력:
- `/lead_fusion/association/image`: 합성 검증 화면
- `/lead_fusion/ground`, `/lead_fusion/nonground`, `/lead_fusion/clusters`: 단계별 점군
- `/lead_fusion/association`: 구조화된 트랙/후보/유효성 정보

장착/투영 값은 `cpp/config/association.yaml`. 실행 중 camera_position_m/camera_rpy_rad,
fx/fy/cx/cy는 ROS parameter로 조정해 화면을 확인할 수 있다. 이는 수동 초기 정렬 도구이며
자동 extrinsic calibration을 수행하는 기능은 아니다. 지면 초기 높이는 재시작 시 적용한다.

## 검증 범위와 발견된 한계

코어 시험: 전역 Hungarian 할당, 등속 운동의 ID/속도 수렴, 트랙 만료, 카메라 전후 투영.
합성 ROS 시험: 1cm 지면 잡음과 표면 샘플 물체를 사용해 Patchwork++부터 단일 후보 대응,
같은 bin의 두 물체 모호함 거부, 영상 만료 거부, A₂ metric 전체 NaN 무시를 확인한다.
완전히 무잡음인 인공 평면에서는 upstream TGR의 평탄도 분산이 0인 경우 일부 지면 패치를
비지면으로 남겨 군집이 합쳐지는 현상이 확인됐다. 이는 현 상태의 알려진 한계이며,
시험에 잡음을 넣었다고 이 수치적 경계 사례가 해결된 것으로 간주하지 않는다.
지역 지면 높이 후처리도 지면 지지가 없는 큰 영역까지 복구하는 알고리즘은 아니다.
실센서에서 지면/군집 정확도를 수치로 인증하려면 수작업 정답 또는 검증 데이터가 필요하다.

## 차량 크기와 진동 완화 (2026-11-03)

주황색 승용차 크기를 참고한 초기 설정이며, 차량 실측 치수 또는 분류 결과가 아니다.
실물 설정의 Euclidean tolerance를 0.65 m에서 0.45 m로 줄여 먼저 작은 군집을 만든다.
코드의 미설정 기본값은 기존 0.65 m이며 실물 실행에는 association.yaml을 사용한다. 작은 군집끼리 또는 작은 조각과 큰 군집을
XY 간격 0.45 m 미만, 높이 겹침 0.2 m 이상일 때 병합하되, 합친 관측 상자가
전방/좌측/위 축 기준 5.2 × 2.6 × 2.3 m 이내여야 한다.
두 군집 모두 XY 관측 면적 1.5 m² 이상이면 병합하지 않는다.
이는 전방을 향한 승용차용 보수적인 초기 규칙이다. 회전 차량/다른 차종과
근접 물체 구분은 추가 검증이 필요하며, 완전한 차량 크기로 상자를 강제로 늘리지 않는다.

대응 후보는 원시 관측 크기가 위 최대 범위 이내이고, 필터 크기가 최소
0.6 × 0.5 × 0.5 m여야 한다. 벽과 합쳐져 크게 잡히는 순간은 후보에서 제외한다.
큰 군집의 정확한 개별 차량 분할 및 가림 중 재식별까지 해결한 것은 아니다.

- 출력 center_m의 XY는 CV Kalman 위치, Z와 size_m은 시정수 0.25초 EMA.
- raw_center_m/raw_size_m은 해당 스캔의 실제 관측 범위다.
- 영상의 회색 얇은 상자는 원시 관측, 컬러 상자는 안정화한 추정이다.
- 카메라 존재 확률은 시정수 0.15초 EMA, 진입 0.50 / 유지 0.35,
  연속 영역 확장 0.25를 사용한다. 황색 영역은 차량 전체 bbox가 아닌 A₂ bin 영역이다.
- 동일 A₂ stamp를 반복 필터링하지 않는다. 시간 페어/이력 유효성 실패 시
  존재 상태를 초기화하며, 과거 값을 유효 검출로 유지하지 않는다.
- 필터는 지연을 추가한다. 원시 거리와 필터 거리의 제동 영향은 아직 검증하지 않았다.
  이 노드는 계속 대상 대응 검사용이며 F2/CAN에 연결되지 않는다.

C++ 시험에 가까운 조각 병합, 인접 차량 비병합, 큰 벽의 후보 제외,
이동하는 물체의 교대 ±0.2 m 잡음에서 위치/크기 제곱오차 감소를 추가했다.

# 차량 군집 / 진동 완화 — 2026-11-03
사용자: 주황색 승용차를 선행차 기준으로, 작은 군집과 큰 검출 진동 개선 요청.
association.yaml 초기 군집 0.45m + 0.45m 이내 조각 병합, 합친 최대
전방/좌측/위 크기 5.2/2.6/2.3m; 큰 두 군집은 병합하지 않음.
출력 XY는 Kalman state, Z/크기 EMA 0.25s; raw_center/raw_size 별도 저장.
카메라 존재 EMA 0.15s, 진입 .5/유지 .35/연속영역 .25. 유효 페어 상실 시 reset.
회색 원시 상자와 컬러 필터 상자 표시. 미관측 차량 부분을 인위적으로 채우지 않음.
최종 실물45초 outputs/association_stable_fine: 443 association, unique357 (모두 T3),
verified0, A2 20.9999Hz, LiDAR10.0019Hz. T3 연속369쌍 좌우위치 변화RMS
9.40cm→2.23cm, 폭 변화18.80cm→4.71cm. 이는 정확도 아닌 필터 전후 변동량.
초기 입력/이력 준비 외에도 후보 없음35프레임 존재. 완전한 검출 유지 해결은 아님.
장착 미세각/내부보정/시간검증 및 주행 중 가림/인접 차량 시험 미완료.
F2/CAN 연결 없음. 기존 fusion_node와 새 association_node를 혼동하지 말 것.
실물 시험에서 시작한 드라이버/추론/GUI는 종료함.


검증: C++ 코어 2/2 통과, 최종 0.45m 설정 합성 ROS의 단일/다중 후보 및
stale 거부 통과. 합성 시험도 FASTRTPS_DEFAULT_PROFILES_FILE에
cpp/config/fastdds_large_images.xml을 지정해야 한다. 기본 DDS로 실행한 시험은
영상 페어 불가로 실패했고, 실물과 같은 대용량 DDS 설정 재시험은 통과했다.
실물 감사 결과의 children은 camera/lidar/a2/association 모두 정상 종료(0)다.

## 전방 180° / 카메라 방향 확인

사용자는 이전 화면의 약 10.2m CANDIDATE 상자만 실제 차량을 올바르게 검출한 것으로
확인했다. 이를 해당 장면의 관찰로 기록하며, 거리 10.2m나 특정 track ID로 검출을
고정하지 않는다. 그 밖의 벽/구조물 군집은 차량 검출 정답이 아니다.
show_debug_clusters=false 기본값은 카메라/BEV에 대응 후보 상자만 표시한다.
true에서는 비후보를 회색 CLUSTER (not vehicle)로 표시한다.
내부 군집과 모호한 후보 판정은 숨기거나 삭제하지 않는다.

현재 driver transform_flag=false이며 원시 /lidar_points는 hesai_lidar의 360° 자료다.
body_from_lidar(yaw=+pi/2)는 X_forward=-Y_lidar, Y_left=X_lidar, Z_up=Z_lidar.
새 /lead_fusion/front_points는 동일 변환 후 X_forward>=0인 전방 반원만 발행한다.
frame_id는 lidar_vehicle다. 기존 ground/nonground 토픽은 지면 알고리즘 점검용 360°이며
변경하지 않았다. 군집용 ROI X>1, X<70, abs(Y)<12는 전방 반원 전체와 별개다.

BEV의 흰 화살표는 LiDAR 전방, 흰 좌우 경계와 15m 호는 전방 180°를 뜻한다.
노란 선은 투영 코드와 동일한 회전/카메라 위치로 계산한 카메라 광축과 영상 좌우 경계다.
fx=640, width=1280인 현재 임시 모델의 수평 화각은 약 90°로, LiDAR 반원보다 좁다.
이 그림이 코드 좌표의 일관성을 보여줄 뿐 실제 카메라 보정 완료를 뜻하지 않는다.
camera_rpy_rad와 intrinsics는 실측 후 갱신해야 한다. 검증 플래그는 계속 false다.
상자의 x=10.2m는 LiDAR 원점 기준 전방 중심 좌표이며 방사 거리나 범퍼 간격이 아니다.

시험에 원시 LiDAR 전/후/좌/우의 광학축 변환과 후방 점의 front_points 제외를 추가했다.

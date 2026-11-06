# 카메라 대응 대상 거리 추적 변경 — 2026-11-04

사용자 분석 범위: `outputs/ui_20261104_120315`의 12:04:00~12:10:00(KST)만.
핵심 장면은 `ui_00004.mp4` 재생 22초 전후: 카메라 앞차 영역은 있으나
LiDAR 거리 주 대상은 오른쪽 다른 위치에 표시되거나 소실됨. 이 관측으로 보정각이나
실제 앞차 거리를 추정해서 설정하지 않았다. 기존 영상/CSV는 원본 그대로 보존한다.

구현:
- `CameraTargetSelector`: 현재 카메라·LiDAR 일대일 대응이 있고 확인 횟수 3회인
  트랙만 신규 주 대상/대기 대상으로 선택. 같은 카메라 타임스탬프 반복은 추가 확인이 아님.
- 같은 ID를 유지하고, 더 가까운 LiDAR 단독 군집으로 교체하지 않는다. 카메라 지원이
  소실되면 마지막 대응 시각 기준 최대 0.3초만 유지. `held_camera`, `held_ambiguous`,
  `predicted`는 관측 확정과 구분한다. 반복 LiDAR 입력으로 이 시간을 연장하지 않는다.
- 카메라 대응 후보는 차량 전체 깊이를 요구하지 않는다. 유한한 관측 크기 상한은 유지하고,
  점 5개 이상·수평 관측 폭/깊이 중 큰 값 0.6m 이상·높이 0.3m 이상인 표면을 허용한다.
  기존 주 대상은 부분 관측 조건(수평 0.8m/높이 0.15m)을 적용할 수 있으나 새 대상에 적용하지 않는다.
  이는 학습된 차량 분류기가 아니며 AABB 상한/군집 합침/보정 문제를 모두 해결한 것은 아니다.
- `fusion_node`: 현재 측정되고 카메라 대응이 확인된 주 대상만 P5 입력으로 사용.
  유지/예측, 모호성, 카메라 미대응은 invalid. 기존 extrinsics/clock/ego/motion 검증도 필요하다.
  `BrakeEstimate.source=lidar_primary` 이름과 P5 수식은 유지하지만 주 대상 선택 의미는 변경됨.
- UI: 거리 카드의 대상은 카메라 지원 주 대상/대기 대상만. 빨강=현재 대응,
  청록=유지/예측, 노랑=카메라. 대응 없는 거리로 빈칸을 채우지 않는다.
- `association.csv`: camera_stamp, camera_regions, camera_probabilities, primary_camera_age_s 추가.
- `association.tracks.csv`: 모든 트랙의 ID/관측 여부/확인 횟수/관측 나이/크기/중심/
  거리/속도/공분산/투영 상자/최대 겹침/카메라 영역/선택 여부/탈락 이유 기록.
  range_stage_counts는 기존처럼 거리대별 전처리 점 수를 보존한다.
- ROS 메시지 `Association`, `ClusterTrack`에 진단 필드 추가. 연결된 메시지 사용자는
  같은 install_gpu 환경을 다시 source해야 한다. 런처의 기본 경로는 install_gpu다.

검증: GPU 빌드/설치 완료, CTest 2개 및 합성 ROS association/P5/far-cloud 시험 통과.
재현: `bash run.sh target-test` (센서 드라이버 없이 localhost domain 93/94/95).
상세 근거: `cpp/evidence/camera_target_20261104/validation.json`.

설정은 cpp/config/association.yaml의 camera_match_confirmations=3,
camera_target_hold_s=0.3. 기존 미검증 intrinsics/extrinsics와 시각 보정값은 유지했다.
실센서를 자동 재시작하지 않는다. 다음 수집에는 신규 진단 CSV가 자동 생성된다.
지정 구간의 원시 점군·A₂ 전체 출력이 보존돼 있지 않아 기존 MP4/집계 CSV만으로
새 알고리즘의 실측 거리 정확도를 재검증할 수 없다. 구현 검증과 실차 정확도를 구분한다.

> **최신 상태 (2026-11-04):** 사용자 요청으로 카메라·GPU A₂·LiDAR bridge·fusion·association UI/녹화·domain107 ROS 데몬 종료 확인. 작업 공간 빌드 로그 747개(23,381,355바이트) 삭제. 아래 실행 중 표시는 과거 기록이다.

> **출력 정리 (2026-11-04):** 오늘 주행·녹화·진단 기록 보존. 이전 날짜 출력 26개 폴더 및 재생성 가능한 중간 파일/캐시 등 총 249개 파일(226,696,200바이트) 삭제. 과거 문서의 11월 3일 outputs 링크는 삭제된 기록일 수 있음. 원본 데이터 링크·모델·실행환경 유지. [삭제 목록](docs/cleanup/20261104_outputs.json).

> **최신 실행: 카메라+LiDAR 연결 복구 (2026-11-04).** `outputs/ui_20261104_120315`에서 UI/MP4 실행 중.
> 카메라21Hz·A₂21Hz, 이번10초 실측 공용 LiDAR 약5.8Hz/인지5.9Hz. 수신 간격 때문에 간헐 STALE 발생; 안정10Hz 검증 아님.
> 외부 localization/Hesai/GNSS 종료·재시작 없음. IP 경로 유지. bridge를 독립 source/sink 프로세스+로컬 socketpair로 분리해 domain0→107 전달 복구.
> [설정·검증 기록](docs/network/camera_restore_20261104/README.md). 아래 중복 입력 차단·종료 대기 기록은 과거 상태.

> **현재 연결 제한:** 외부 localization 두 실행의 `/lidar_points` 발행자2개를 감지해 우리 LiDAR 입력 전달 차단 중. 카메라/A₂는 수신 중. 외부 실행 종료는 사용자 확인 대기.

> **최신 실행 (2026-11-04): 카메라 복구 완료.** 카메라·GPU A₂·공용 LiDAR·UI·MP4 실행 중 (`outputs/ui_20261104_113405`).
> 기존 LiDAR/GNSS/공유/인터넷 프로필 보존. 카메라 전용 MACVLAN `a1cam0`만 추가. 자동 노출·게인 OFF, 노출3,000μs·게인0dB 설정.
> [네트워크 변경·보존 확인](docs/network/camera_restore_20261104/README.md). 아래 카메라 꺼짐 기록은 이전 상태.

> **카메라 수동 설정 (2026-11-04):** 사용자 요청으로 UI 실행기의 자동 노출·자동 게인 OFF. 노출3,000μs, 게인0dB. 다음 카메라 실행부터 적용하며 현재 카메라 미기동 상태는 유지.

> **현재 실행 (2026-11-04):** localization 공용 LiDAR → domain107 인지·UI·MP4 실행 중.
> 기존 LiDAR/GNSS 네트워크·설정·프로세스 유지. 카메라 포트가 바뀌어 **A1_CAMERA_ENABLED=0**으로 실행(카메라/A₂ 없음).
> [현재 설정·보존 확인·실행 명령](docs/network/current_before_shared_start/README.md). 출력: `outputs/ui_20261104_112823`.

> **최신 공용 LiDAR 연결:** [설계·명령어·검증](docs/SHARED_LIDAR_KR.md).
> 우리 원커맨드의 Hesai 드라이버 자동 기동 제거. 기본 domain0 `/lidar_points` → C++ 단방향 bridge → domain107 전용 입력.
> `./run_sensor_fusion.sh lidar-check`로 수신 검사. 입력 없으면 센서 자동 기동 없이 종료. 현재 공용 발행자 없음, UI 중지 유지.

> **최신 상태: 2026-11-04 종료 완료.** 사용자 요청으로 프로젝트 센서 드라이버·인지·UI·녹화 및 도메인107 조회 데몬 종료.
> 재생성 가능한 컴파일 객체·Python 캐시 2,297개(49,051,542바이트) 정리. 원시 데이터·MP4·모델·실행파일 보존.
> [정리 내역](docs/cleanup/20261104_shutdown.json). 아래 실행 중 표시는 이전 시점 기록이다.

> **최신 실차 기록 (2026-11-04):** [관측·수정·검증·미완료 항목 정리](docs/session_20261104/README.md).
> 거리 적응형 XY 군집 / 부분 관측 유지 / 처리·BEV100m / P5후보10~50m / 카메라진입0.4 적용.
> **옆모습 AABB 폭3.18m가 최대2.6m에 걸리는 문제 확인. OBB 수정은 아직 미완료.**
> 현재 `outputs/ui_20261104_101515`에서 실행·MP4 녹화 중. 아래 중지 상태 등은 이전 기록.

> **네트워크 최신 설정:** [센서 전용 영구 프로필](docs/NETWORK_KR.md). 2026-11-04 적용, 세 센서 ping 성공. 이후 인터넷 변경 시 센서 IP 재설정 불필요.

# 최신: LiDAR 연속 추적 / 동일 ID P5 — 2026-11-04

[추적 연속성 변경·검증](docs/TRACK_CONTINUITY_KR.md). 가까운 주 대상 1대만 P5, 먼 차량은 대기.
초기 coast0.3초/memory0.8초, 관측·예측 구분, 재연결 시 ID·속도 유지, 모호 측정의 중복 ID 억제.
기본 shadow mode=association으로 UI와 P5 대상 ID 연결. raw P5 수식 유지, 이 모드의 camera gate는1.
실차 IMU·보정·운동 검증은 미완료로 유효 제동값 미검증. CAN 송신 없음. 주행라인 연동은 사용자 요청으로 보류.
아래 ‘별도 P5 트래커’ 기록은 이전 구현이다. 현재 UI는 중지 상태.

# 정리·검증 완료 — 2026-11-04
중복 데이터 SHA256 확인 후 원본 링크로 전환, LibTorch 설치ZIP 제거. 약71GiB→5.7GiB,69.2GB제거.
참조 경로 원본 의존성은 docs/cleanup/20261104.json 확인. 원본·실주행MP4·런타임 유지.
GPU 빌드/core2시험/합성ROS2대·3대cap·동일bin모호·stale·CSV시험 통과. 실주행 미검증.
UI는 중지 상태이며 자동 재시작하지 않음. 아래 과거 ‘실행 중’ 기록과 구분.

# 최신 변경 — 2026-11-04

[작업 방향·코드 자체 점검](docs/SELF_REVIEW_20261104_KR.md): 후보 최대2대, 수평 중심거리 10~50m.
카메라 독립 bin 영역과 LiDAR 일대일 대응, 빨강=대응 후보/주황=LiDAR 단독.
P5 동일 ID 연결·실측 보정·원거리 군집 개선은 미완료. 아래는 이전 기록을 포함한다.

[현재 센서 설정·실행 환경 (2026-11-03)](SENSOR_ENVIRONMENT_KR.md) — 원커맨드, 네트워크, GPU, 보정 상태와 미완료 항목.

# 주행 관측 결과 — 2026-11-03
첫30초 association277개 중 no_camera_presence264, 후보10, 모호3. 두번째30초 주황차약6.5m 후보301/301, A2 20.997Hz history_ready630/630. 고정 m 단위 AABB 폭2.6m 제한은 비스듬한 차량을 탈락시킬 수 있음(초록차 투영영역 트랙폭약2.83m). 군집 자체와 최종후보필터 구분 필요. 단일peak bin만 선택하므로 다중차량 대응 미구현. 주행중 설정/프로세스변경 없이 관측만 수행, MP4계속녹화. outputs/driving_diagnosis 및 _2 참조.

# 격리 및 연속 MP4 — 2026-11-03
원커맨드 도메인107 고정, localhost+UDP127.0.0.1, 상속된 팀원 ROS/library 경로 제거.
camera드라이버/UDP2368 점유 검사, 도메인107 application node 존재 시 기동거부.
팀원 원본 수정 없음. 감사 근거 cpp/evidence/isolation, 문서 docs/ISOLATION_RECORDING_KR.md.
UI 별도 C++ 스레드 1600x860 10fps MP4,60초 분할,recordings/frames.csv,
정상종료 finalize. 첫실물60초600프레임 전체디코딩 통과. 최신구간은 쓰는중 재생불가.
실행중 outputs/ui_20261103_234933에서 계속녹화. 임의로 중지하지 말 것.
사용자가 현재 주행 중 초록/주황 두 차량 검출이 약하다고 하여 읽기전용 관측 시작.
현재 주행 중 설정변경/재시작 금지 방향으로 관측하고 원인 분석 중.
rawIMU 부재/같은 트랙 P5 미연결은 계속 미완료.

# 최신 결정: 원래 방향 복원 / 원커맨드 — 2026-11-03
사용자가 우측45도 요청을 취소: 원래 방향이 맞다고 확인.
최종 association.yaml lidar_roi_center_deg=0.0. -LiDAR Y 전방, 카메라 변환 유지.
루트 run_sensor_fusion.sh 신규: 실행 / status / stop. run_ui.sh 파일잠금/PID관리,
디스플레이 자동선택, 자식 프로세스 정리. 실제 start/stop/restart 확인.
최종 요청에 따라 UI를 다시 실행해 열어둠. 다음 작업 전 status로 실행 여부 확인.
앞선 우측45도 시험은 중간 실험이며 최종 설정이 아님.
IMU수신과 동일트랙P5 연결은 여전히 미완료, CAN송신없음.

# 내일 실험용 UI / 기존 A2 체크포인트 확인 — 2026-11-03
C++ UI 1600x860, 전방70m BEV 눈금, 후보 red, 하단 속도/종가속도/대상/P5 카드.
run_ui.sh 새 런처: 실제 camera/lidar/A2/fusion/association 기동, 로그와 최신 JPG,
Ctrl+C로 시작한 프로세스 그룹 종료. ROS_DOMAIN_ID 기본0, 원시IMU도 같은도메인 필요.
Ego에 gravity-removed acceleration 추가; fusion_node가 /lead_fusion/ego_motion 발행.
IMU 정지/보정/드리프트 제한 유지. UI는 valid와 freshness를 검사, 결측/만료 --.
LiDAR 중단0.3초 STALE 배너. P5 what-if 제외, valid_until 검사, CAN출력없음.
중요: 붉은 association ID와 기존 P5 tracker는 아직 별개. UI에 명시했으며
동일선행차 metric→P5 통합 완료로 주장하지 말 것. GNSS rawIMU 여전히 미수신.
실물 preview outputs/ui_live/latest.jpg, 새 런처 기동/종료 확인.
합성UI/IMU와 코어 시험 통과 cpp/evidence/ui_*; 사용법 docs/UI_KR.md.
사용자 질문 확인: 기존 a2_mf20_v2s_n20.pt + seg_pretrain_fx640_v3_backbone.pt
그대로 CUDA trace; A2 820키/backbone774키 누락0. 재학습 없음.
현재 UI 카메라 존재/bin과 LiDAR 확정군집/투영 겹침의 교집합이 단일일 때만 후보.
카메라가 위치 기준, LiDAR가 metric; 한쪽 단독으로 정상 확정하지 않음.

# 전방 방향과 표시 정리 — 2026-11-03
사용자는 이전 10.2m CANDIDATE만 정상 차량이라고 확인. 다른 구조물 군집은 정답 아님.
거리/ID 하드코딩 없이 기본 후보 상자만 표시(show_debug_clusters=false).
debug true일 때 비후보는 회색 CLUSTER(not vehicle), 내부 후보/모호성 판단은 유지.
드라이버 추가 transform 없음, 코드 +pi/2로 forward=-LiDAR Y, left=LiDAR X.
기존 원시/ground/nonground는 360도. 새 /lead_fusion/front_points는
lidar_vehicle에서 X>=0인 180도 반원. BEV 흰색 LiDAR전방/반원,
노란색 실제 투영 설정으로 계산한 카메라축/FOV. 현재 intrinsics 가정상 약90도.
실제 yaw/내부보정이 맞다는 인증 아님. calibration/clock verified=false 유지.
코어 축/투영 시험 및 합성 ROS 후방점 제외, 단일/다중/만료 시험 통과.
outputs/association_axes에 30초 실물 확인 기록. 영상상 차량 후보가 좌측이고
BEV도 좌측임을 확인. 미세각/시간 보정 완료로 해석하지 말 것.

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

# 최신 진행 상태 — 2026-11-03 후속 작업

사용자 우선순위: 카메라와 LiDAR가 동일 대상을 대응하는지 시각적으로 검증부터 구현.
지면제거/클러스터링/추적은 대표 GitHub/논문을 참고할 것. 그 다음 LiDAR metric으로 F2 계산.
차량 명령 송신은 여전히 제외한다. LiDAR -Y가 차량/카메라 전방이라는 사용자 확인을
shadow.yaml yaw +π/2에 반영(+Z up 가정, 위치/미세각 미측정, verified=false 유지).

GPU 작업 완료: vendor/libtorch 공식 cxx11 ABI 2.4.1/cu118, CUDA toolkit ../../cuda-11.8.
cpp/models_gpu CUDA trace, TF32 off, profiling executor off, FP32.
run.sh build-gpu/camera-driver-gpu/camera-gpu/benchmark-gpu/sensor-check-gpu 추가.
기본 DDS에서는 프레임 누락: raw Aravis 21Hz 누락0 확인 후 fastdds_large_images.xml
64MiB SHM/4MiB 메시지/UDP buffers16MiB로 해결. 드라이버와 추론 양쪽에 적용 필요.
최종 실카메라 60초 1260/1260,20.997Hz,history_ready100%,p50 46.2ms,p95 55.0ms.
C++ vs Python GPU 같은 영상 출력 최대오차0, 원본vsCUDA trace max1.43e-6.
core/기존P5 시험38passed. 로그 cpp/evidence/gpu_*. GPU 시험 프로세스 모두 종료.
이후 새 동일대상검증 파이프라인 구현 중이며 아직 실측 보정/IMU 입력은 미완료.

아래는 이전 CPU 단계 기록이다(최신 GPU 결과와 혼동하지 말 것).

# 작업 인계 — 2026-11-03

최신 변경: 사용자가 GNSS 출력 설정을 지금 할 수 없어 내장 IMU만 쓰기를 요청했다.
`ego_source=imu`를 기본으로 하고 C++ 초기 정지/중력/바이어스 보정 및 제한된 시간의
strapdown 속도 적분을 추가했다. raw IMU 토픽은 아직 없고 장치 TCP에 바이트가 없다.
이것은 구현 완료와 별개의 실센서 입력 차단 요인이다. 기존 INS 모드는 선택적으로 남긴다.

사용자 목표: 카메라 A₂ + GNSS 내부 INS 속도로 선행차 상대 위치/속도/거리와 P5를 계산하는 모드,
카메라 bin만 + LiDAR 거리/속도를 쓰는 모드를 C++로 구현. 이후 사용자가 새 독립 폴더와 기존 자료
전체 복사를 요청했다. 원래 P5 폴더에서 새로 만든 cpp만 여기로 이동했고 기존 원본들은 수정하지 않았다.

검증 완료: C++ 코어 시험, Python 기준 203사례 수치 비교와 기존 시험 합계 38 passed,
선배 파일 4개 해시 일치, 과거 P5 표 5,401행 중 유효 847행 최대 오차 3.55e-15,
ROS 합성 융합에서 상대속도 -5/앞차속도 5/요청 -3 및 입력 만료,
C++ 영상→A₂와 Python 동일 입력 최대 오차 0.

현재 호스트 시스템 Python은 torch 2.14.0+cpu, torchvision 0.29.0+cpu.
변환 시험은 /tmp 전용 timm 1.0.30으로 전체 체크포인트 key 로딩 후 실행했다.
기존 권장 배포 환경 torch 2.4.1/cu118/timm 1.0.20과 다르므로 현재 생성된 TorchScript를
구버전 런타임에 바로 호환된다고 가정하지 않는다. 배포 라이브러리와 같은 환경에서 재변환한다.
RTX A5000 24GB / driver 470.256.02는 host nvidia-smi로 확인. GPU 처리량은 이번에 미검증.

실센서 주소/케이블은 과거와 달랐다. 최신 확인은 docs/SENSOR_AUDIT_KR.md에 추가한다.
보정 플래그는 기본 false로 유지하며 실차 수신, 모델 출력 정확도, 처리량, 실제 제동 검증은 구분한다.

최종 독립 폴더 실물 시험: 카메라 19.18Hz, LiDAR 8.11Hz, CPU C++ A₂ 1.55Hz,
history_ready=false. 원시 IMU 0개, P5 invalid(extrinsics_unverified). 현재 상태에서 실시간
유효 제동값을 낼 수 있다고 보고하면 안 된다. 로컬 빌드한 센서 드라이버까지 사용한 시험이며
시험 후 시작한 프로세스는 종료했다. 결과는 cpp/evidence/live_sensor_summary.json.
현재 배선용 호스트 보조 IP와 LiDAR /32 경로는 임시 적용 상태로 남아 있고 저장 프로필은 유지했다.

# 2026-11-09 카메라 intrinsic · 카메라-LiDAR extrinsic 보정

실내 주차장에서 Hikrobot 카메라(ROI 1280x720, offset 320/240)와 Hesai PandarXT-32를 보정한 기록이다. bag과 누적 점군(`.ply`)은 크기 때문에 넣지 않았다. 원본은 차량 PC `~/calib_data/20261109/`에 있다.

## 결과

| 항목 | 값 |
| --- | --- |
| intrinsic fx, fy, cx, cy | 1038.771, 1046.083, 654.937, 384.473 |
| distortion (plumb_bob k1 k2 p1 p2 k3) | −0.158338, 0.069947, −0.001570, 0.003519, 0 |
| extrinsic `camera_in_lidar_body` xyz [m] (앞, 왼쪽, 위) | **1.021, −0.035, −0.369** (이전 명목값 0.8, 0, −0.3) |
| extrinsic `camera_in_lidar_body` rpy [°] | **−0.62, 0.63, −2.09** (이전 명목값 0, 0, 0) |
| `T_lidar_camera` (x y z qx qy qz qw, 도구 출력) | −0.03524, −1.02045, −0.36884, 0.00912, 0.71096, −0.70298, −0.01660 |

`camera_in_lidar_body`는 LiDAR 원점에서 차량 축(앞/왼쪽/위)으로 잰 카메라 몸체(x 앞, y 왼쪽, z 위)의 위치·자세다. LiDAR(`hesai_lidar`)는 차량 기준 yaw +90°로 달려 있다.

비교 사진: `extrinsic/images/moving_scene02/compare_nominal_top_calibrated_bottom.png`
(위 = 명목값, 아래 = 보정값, 점 색 = 거리). 명목값은 주황 차 윤곽이 약 40 px 오른쪽으로 밀려 있었고, 보정값은 차와 벽 모서리가 맞는다.

## 확인하지 않은 것

- 원거리(40 m 이상) 정합. 이 장면의 물체는 20 m 이내다. 이 카메라에서 1 px ≈ 0.055°이고, 각도 0.2° 오차면 40 m에서 약 14 cm 어긋난다.
- 아래 "퍼짐"은 같은 데이터에서 초기값만 바꾼 반복성이다. 실제 정확도가 아니다.
- 그래서 `geometry_verified`는 false로 두었다.

## 과정

### 1. intrinsic

`camera_calibration cameracalibrator`, 체커보드 8x6 내부 코너(9x7 칸), 칸 0.035 m, 영상 75장. 결과 `intrinsic/ost.yaml`. 예시 영상 `intrinsic/sample_left-0030.png`.

### 2. extrinsic 도구

[koide3/direct_visual_lidar_calibration](https://github.com/koide3/direct_visual_lidar_calibration) (commit `02a0dc0`). 카메라 영상과 LiDAR intensity의 NID(정규화 상호정보)를 최소화한다. 보드가 필요 없다.

의존성: GTSAM 4.2a9, Ceres 2.1.0 (apt의 2.0에는 `ceres/manifold.h`가 없어 빌드가 실패한다), Iridescence `2eaabb0`. 모두 소스 빌드 후 `/usr/local`에 설치했다. 도구 workspace의 `deps_src/`에는 `COLCON_IGNORE`를 둬야 colcon이 의존성을 다시 빌드하지 않는다.

### 3. 카메라 노출

실외 설정(`exposure_us:=3000`, gain 0)으로는 실내 영상 평균 밝기가 0.4/255였다. 아래 설정으로 다시 띄웠다(평균 83/255).

```bash
ros2 launch camera_bringup camera_bringup.launch.py exposure_auto:=Continuous exposure_max_us:=40000 gain_auto:=Continuous gain_max_db:=20
```

### 4. 정지 데이터 시도 (실패)

차를 세워 두고 15 s 녹화 → `preprocess -a` → 명목값을 초기값으로 `calibrate`. 첫 녹화에는 사람이 지나가서 평균 영상에 잔상이 남아 다시 찍었다.

초기값을 흔들면 결과가 흩어졌다. 32채널을 정지 상태로 찍으면 줄 사이가 비어 NID에 뚜렷한 최소점이 생기지 않는다.

| 초기값 변화 (xyz [m], rpy [°]) | 결과 xyz | 결과 rpy |
| --- | --- | --- |
| 0, 0 | 0.954, −0.021, −0.368 | −0.74, 0.45, −2.25 |
| +0.1/+0.05/+0.05, 0 | 0.987, −0.052, −0.365 | −0.55, 0.62, −1.82 |
| −0.1/−0.05/−0.05, 0 | 0.669, −0.043, −0.338 | −0.27, 0.07, −0.94 |
| 0, +2/+2/+2 | 0.829, 0.017, −0.221 | 0.69, 0.66, 1.93 |
| 0, −2/−2/−2 | 0.803, −0.006, −0.294 | −2.37, −1.39, −1.76 |
| +0.1/0/+0.05, 0/−2/+2 | 0.913, −0.012, −0.255 | 0.05, −1.96, 1.64 |

사진: `extrinsic/images/static_attempt/`, 로그: `extrinsic/logs/static_init_tests/`.

### 5. 차를 밀며 녹화 (채택)

`scripts/rec.sh scene02`: 녹화 시작 후 3 s 정지 → 손으로 차를 천천히 밀고 보닛을 눌러 흔들기 → 30 s 뒤 자동 종료. 영상 변화로 0~4 s 정지, 6~15 s 이동을 확인했다(`motion_check_0_10_20_31s.png`). scene01은 1.9 s에서 끊겨 쓰지 않았다.

```bash
ros2 run direct_visual_lidar_calibration preprocess sel02 pre_dyn02 -a -d     # -d: 움직인 scan을 정합해 누적 (점 53만 → 798만)
# calib.json 의 results.init_T_lidar_camera 에 초기값을 넣고
ros2 run direct_visual_lidar_calibration calibrate --data_path pre_dyn02 --background --auto_quit
```

초기값 6개로 다시 돌린 결과:

| # | 초기값 변화 (xyz [m], rpy [°]) | 결과 xyz | 결과 rpy | 마지막 NID |
| --- | --- | --- | --- | --- |
| 0 | 0, 0 | 0.915, −0.060, −0.119 | 1.38, 0.33, −1.50 | 0.922 |
| 1 | +0.1/+0.05/+0.05, 0 | 1.020, −0.036, −0.370 | −0.62, 0.63, −2.08 | **0.899** |
| 2 | −0.1/−0.05/−0.05, 0 | 1.020, −0.035, −0.369 | −0.62, 0.64, −2.09 | **0.899** |
| 3 | 0, +2/+2/+2 | 1.021, −0.035, −0.369 | −0.62, 0.64, −2.09 | **0.899** |
| 4 | 0, −2/−2/−2 | 0.799, −0.009, −0.266 | −2.17, −1.75, −2.03 | 0.925 |
| 5 | +0.1/0/+0.05, 0/−2/+2 | (중단) | | |

1~3번이 같은 값으로 모였고(퍼짐 1 mm, 0.01°) NID가 가장 낮다. 0, 4번은 NID가 더 높은 국소해다. 5번은 결론을 바꾸지 않아 중단했다. 채택값은 1~3번 평균이다. 로그: `extrinsic/logs/moving_init_tests/`.

## 파일

| 경로 | 내용 |
| --- | --- |
| `scripts/rec.sh` | 녹화 스크립트 (3 s 정지 → 30 s 밀기 안내 → 자동 종료) |
| `scripts/calib_util.py` | a1.yaml ↔ `T_lidar_camera` 변환, 투영 비교 그림 (확인용으로 쓴 그대로) |
| `intrinsic/` | `ost.yaml`, `ost.txt`, 예시 영상 |
| `extrinsic/result/calib.json` | 채택 결과 (도구 출력) |
| `extrinsic/images/` | 카메라 영상, LiDAR intensity, 명목값·보정값 투영 비교 |
| `extrinsic/logs/` | preprocess·calibrate 로그, 초기값 시험별 결과 |
| `repo_changes/` | A1-Challenge-2026에 반영한 설정 파일 사본 (`a1.yaml`, `camera_info_1280x720.yaml`, 퓨전 `calibration.yaml`) |

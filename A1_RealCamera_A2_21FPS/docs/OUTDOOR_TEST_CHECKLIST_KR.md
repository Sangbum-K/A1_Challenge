# 실외 인식 테스트 체크리스트 (카트)

카트에 본체와 모니터를 올려 실외로 나가서, 실제 차량이 접근/이탈할 때 a₂가 반응하는지를
화면으로 확인하는 시험이다. 실행은 전부 `./run.sh` 하나로 한다.

```bash
cd /home/ailab/git/sangbum/A1_Challenge-main/A1_RealCamera_A2_21FPS
./run.sh help
```

카트는 정지에 가까우므로 `ego_motion=[0,0]`(기본값)이 맞다. `--ego-vx`는 건드리지 않는다.

## 1. 실내에서 (출발 전)

```bash
./run.sh check
```

`FAIL`이 있으면 출발하지 않는다. 카메라 랜선은 `camlink`(enp5s0) **하나만** 연결한다.
`camlink10g`(enp6s0)도 같은 192.168.1.100/24를 쓰기 때문에 둘 다 꽂으면 충돌한다.

## 2. 카트 장착

- [ ] 렌즈 초점을 **원거리**에 맞춘다. 실내 모니터(약 1 m)에 맞춰둔 초점은 수십 m 앞차에서 흐리다.
- [ ] 카메라를 카트에 단단히 고정하고, GigE 커넥터 락킹까지 확인한다.
- [ ] 모니터가 햇빛에 보이는 각도인지 확인한다. 안 보이면 그늘을 만들거나, 나중에
      `outputs/live_*/a2_overlay.mp4`로 확인한다(기본으로 저장된다).

## 3. 조준 + 노출 맞추기

```bash
./run.sh aim
```

모델 로드 없이 몇 초 만에 뜬다. 초록 박스가 **모델이 실제로 보는 250×1280 영역**,
노란 선이 중앙 관심 bin 16~25다.

두 가지를 맞춘다.

**노출** — 실내용 15000 µs는 주간 실외에서 하얗게 날아간다. `[`와 `]`로 실시간 조절하면서
화면 상단 판정을 본다.

- `OVEREXPOSED` (clipped > 5%) → `[`로 노출을 낮춘다
- `UNDEREXPOSED` (crushed > 20%) → `]`로 올린다
- `SOFT` → 렌즈 초점을 다시 맞춘다

**조준** — 차가 다니는 노면이 초록 박스 안에 들어와야 한다. 카메라 각도로 맞추는 게 1순위고,
안 되면 `./run.sh aim --offset-y 300`처럼 센서 crop 위치를 바꾼다.

`q`로 끝내면 다음에 쓸 `--exposure` 값과 `--roi-y0/--roi-y1`을 출력해준다.

> `[`/`]` 노출 조절은 camera_aravis2가 `ExposureTime`을 ROS 파라미터로 노출할 때만 된다.
> 안 되면 화면에 `exposure not tunable`이라고 뜬다. 그때는 값을 바꿔가며 재실행한다:
> `./run.sh aim --exposure 1500`

## 4. 인식 확인 run

```bash
./run.sh live --exposure 2000        # 3단계에서 찾은 값
```

- 실시간 화면 + `a2_overlay.mp4` + 전체 `(40,7)` JSON 로그
- 화면에서도 `[`/`]`로 노출을 계속 조절할 수 있다
- `q` 또는 `Esc`를 누르면 `summary.json`을 쓰고 종료

화면 해석은 기존과 같다.

- ROI의 빨간 세로 strip: `sigmoid(a₂[:,0])` object probability
- `d=...m`: `a₂[:,6] × 40`, `vL/vT=...m/s`: 종/횡 상대속도 `a₂[:,3:5] × 20`
- 상단 `PEAK ... RAW a2`는 선택된 bin의 원본 7채널이며, `DECODED`는 이를 단위 변환한 값
- `frames.jsonl`의 `a2_bins.raw_40x7`에는 매 프레임 전체 40×7 원본값이 저장된다
- 빨간 왼쪽 화살표는 접근, 파란 오른쪽 화살표는 이탈
- 노란 테두리 bin 16~25는 고정 중앙 관심영역이며 **차선 인식 결과가 아니다**

나중에 다시 분석하고 싶으면 원본까지 녹화한다(약 1.1 GB/분).

```bash
./run.sh live --exposure 2000 --bag
./run.sh replay outputs/live_20260918_150000/bag
```

## 5. 처리량 숫자가 필요하면 따로 잰다

화면·영상·JSON 로그를 켠 run은 렌더링 비용이 출력 FPS에 섞인다. 기준값은 전부 끈 run에서 잰다.

```bash
./run.sh bench --secs 60 --exposure 2000
```

## 6. 이번 시험으로 주장할 수 있는 것과 없는 것

주장할 수 있다.

- 실외 실차 입력에서 camera→a₂가 21 FPS를 유지하는가
- 앞차 접근/이탈에 object probability와 `vL_rel` 방향이 반응하는가

주장할 수 없다.

- **거리(m)·상대속도(m/s)의 수치 정확도.** 1280×720은 1920×1200 센서의 중앙 crop이고
  수평 FOV가 학습 카메라와 같다는 calibration 근거가 없다. `summary.json`의
  `geometry_valid_for_accuracy`는 계속 `false`다.

## 7. 문제가 생기면

```bash
./run.sh stop        # driver/runner가 남아 있을 때
./run.sh features    # GenICam feature 이름 확인 (카메라가 idle일 때만)
tail -f outputs/live_*/camera_driver.log
```

## 8. 20 m 거리와 상대속도 현장 시험 (2026-09-27)

카메라를 **고정**하고 상대 차량만 움직이는 시험이면 `--ego-vx 0 --ego-wz 0`을 유지한다.
카메라 자체가 주행 차량과 함께 움직이면 현재 `--ego-vx`는 run 전체에 고정된 값이므로
가감속 구간의 종/횡 속도 정확도 검증에는 쓰지 않는다. 실시간 ego 속도 입력이 먼저 필요하다.

1. `./run.sh check` 후 generic 커널에서 `./run.sh bench --secs 60`으로 21 FPS를 확인한다.
   `live` 화면에서 `unique past 20/20`, skip 누적, FPS, 포화/암부를 함께 본다.
2. 렌즈 초점·노출을 맞추고 장착 **높이, pitch, yaw, 렌즈, 센서 ROI**를 적는다. 현재
   `configs/camera_info_1280x720_uncalibrated.yaml`의 K/왜곡값은 실측 보정값이 아니다.
   높이만 바꿔서 출력 거리 20 m를 만들면 다른 거리에서 맞는다는 보장이 없다.
3. 거리 GT는 차량의 어느 기준점(범퍼/차량 중심 등)까지 잰 것인지 기록한다. 각 높이에서
   정지 차량을 10/20/30 m 같은 여러 지점에 두고 **동일 노출·ROI·threshold 0.5**로
   각각 최소 10초 기록한다. 각 위치에서 1초 temporal history가 찬 뒤 값을 평가한다.
   예: `./run.sh live --exposure 2000 --obj-thr 0.5 --bag --out outputs/h120_d20_static`
4. 정지 시험에서는 카메라·차량 모두 정지했을 때 종/횡 상대속도가 0 m/s 부근인지,
   거리 중앙값·시간 변동폭·선택 bin 전환 횟수를 함께 본다. 원본과 화면의 group 평균을
   구분한다. 정지 차량에서 속도가 계속 나오면 threshold만 올려 숨기지 않는다.
5. 같은 장착 상태에서 차량을 천천히 접근·이탈시키고, 좌·우 이동은 별도 구간으로 나눈다.
   출발/정지 시각과 실제 이동 거리·시간을 기록해 상대속도 GT를 만든다.
   접근의 종방향 상대속도는 음수, 이탈은 양수로 기대한다. 횡방향 부호는 먼저
   실제 좌·우 움직임과 모델 좌표계가 어떻게 대응되는지 확인한다.
6. 0.5는 시작 기준이다. 동일한 원본 bag을 `./run.sh replay <bag> --obj-thr 0.3`/`0.5`/`0.7`
   로 비교하고 오검출, 누락, bin 전환을 함께 본다. threshold는 object 표시/lead 선택만
   바꾸며 A₂ 거리·속도 회귀값을 고치지 않는다.

정지 차의 속도 흔들림을 줄일 때는 먼저 노출·초점·카메라 진동·ROI/FOV·21-frame history와
선택 bin 전환을 확인한다. 그 뒤 같은 차량의 bin을 시간에 따라 추적하고 **표시용** 속도에
짧은 중앙값/EMA와 진입·해제 hysteresis를 적용할 수 있다. 반드시 원본 A₂도 남겨
모델 오차와 표시 안정화 효과를 따로 평가한다.

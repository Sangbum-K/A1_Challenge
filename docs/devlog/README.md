# 개발 기록

A1_Challenge의 작업 경위를 날짜별로 모아 둔다. 무엇을 바꿨는지는 git에도 남지만,
왜 그렇게 정했고 어디까지 확인했는지는 여기에 남긴다.

시각은 모두 한국 시간(KST)이다.

## 현재 상태 (2026-11-06 오후 기준)

| 항목 | 상태 |
|---|---|
| 커밋 | 로컬 `main`이 `origin/main`보다 3커밋 앞서고, push하지 않았다. 추적·대응 교체와 폴더 정리는 아직 커밋하지 않았다 |
| 코드 시험 | 합성 시험만 통과했다. 실제 Eigen·ROS·PCL·OpenCV 빌드와 스모크 시험은 실행하지 못했다 |
| 실차 | 추적·대응 교체 후 한 번도 돌리지 않았다 |
| 꺼져 있는 기능 | 가드레일 거부와 원거리 경로(구조물 라벨 미연결), 레이싱라인 판단(CSV 미설정) |
| 제동 출력 | shadow 전용이다. 검증 플래그는 모두 false이고 차량 명령을 보내지 않는다 |

## 다음 할 일

순서대로 진행한다. 1번이 이번 변경 전체의 첫 실제 검증이다.

1. 변경을 커밋해 Nuvo로 옮기고 빌드와 시험을 돌린다. 차량 PC의 위치는 지금처럼 `A1_Total/src/perception/sensor_fusion`이다.

   ```bash
   cd /home/ailab/git/A1_Total && source env.sh
   bash build.sh --packages-up-to a1_fusion_cpp
   bash src/perception/sensor_fusion/run.sh verify
   bash src/perception/sensor_fusion/run.sh ros-test
   bash src/perception/sensor_fusion/run.sh target-test
   ```

2. 현장에서 노출값을 고르고 카메라·LiDAR를 보정한다. 절차는 [CALIBRATION_KR.md](../../sensor_fusion/docs/CALIBRATION_KR.md)에 있다.
3. `A1_RECORD_BAG=1`로 주행 bag을 확보한다.
4. 맵의 키프레임별 점 접근 방법을 확인해 구조물 라벨을 연결한다.
5. `track_merged` 맵과 레이싱라인의 좌표 정렬을 확인한다.

## 타임라인

| 시각 | 작업 | 기록 |
|---|---|---|
| 11/05 10:19 | A1_Total의 sensor_fusion을 저장소에 추가 | [가져오기와 GNSS ego](2026-11-05-sensor-fusion-import-gnss-ego.md) |
| 11/05 11:02 | 자차 입력을 GNSS odom으로 전환, 후면→측면 연결 개선 | [가져오기와 GNSS ego](2026-11-05-sensor-fusion-import-gnss-ego.md) |
| 11/05 23:58 ~ 11/06 00:18 | 원격 pull, 충돌 해결, 병합 | [원격 병합](2026-11-06-merge-origin-main.md) |
| 11/06 00:18 ~ 00:33 | 폴더 파악, 주행 영상 분석, 전체 점검 | [인식 점검](2026-11-06-perception-audit.md) |
| 11/06 00:34 ~ 02:54 | 개선안과 설계 문서 작성, 재검토 2회 | [설계 과정](2026-11-06-tracking-design-review.md) |
| 11/06 02:54 ~ 03:30 | 추적·대응·대상 선택 교체 구현 | [구현](2026-11-06-tracking-implementation.md) |
| 11/06 13:48 ~ 13:57 | 저장소 폴더 구조 정리 | [폴더 정리](2026-11-06-repo-restructure.md) |
| 11/06 오후 | git 추적 범위 정리, `reference/` 추적 해제 | [reference 추적 해제](2026-11-06-reference-removal.md) |

## 관련 문서

- [추적·대응 설계 문서](../superpowers/specs/2026-11-06-sensor-fusion-tracking-association-design.md): 현재 동작의 기준이다. 16절에 구현 상태가 있다.
- [sensor_fusion README](../../sensor_fusion/README.md): 실행 방법
- [sensor_fusion 인계 문서](../../sensor_fusion/PROJECT_CONTEXT.md): 누적 결정과 인계 사항

## 기록 작성 규칙

- 파일명은 `YYYY-MM-DD-주제.md`다. 작업 하나에 파일 하나를 쓰고, 위 타임라인에 한 줄을 추가한다.
- 맨 위에 시각, 관련 커밋, 근거 자료를 적는다.
- 사용자 결정과 그 이유를 적는다. 코드만 봐서는 알 수 없는 내용이기 때문이다.
- 코드 시험, 합성 시나리오, 센서 수신, 실차 결과를 구분한다. 실행하지 못한 항목은 못 했다고 적는다.
- 지난 기록이 틀린 것으로 밝혀지면 본문을 지우지 않고 정정을 덧붙인다.
- 상태는 위 "현재 상태"와 "다음 할 일"에서 최신으로 유지한다.

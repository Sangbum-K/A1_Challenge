# Lead Vehicle LiDAR Fusion

2026-10-12 설계 기준. 실시간 처리는 C++17/ROS 2, 학습·재생·평가는 Python을 사용한다.

**현재는 설계 및 검증용 기반 프로젝트다. 센서퓨전 완성품이나 차량 제어 프로그램이 아니다.**

- [전체 설계](docs/DESIGN_KR.md): 10 Hz 처리, 좌표·시간, 로컬/GNSS 분리, 제어 인터페이스.
- [기존 리포 활용 방안](docs/BACKENDS_KR.md): CenterPoint + Autoware tracker 우선, A₂ 보조, BEVFusion 비교.
- [복사된 P5 결과](evidence/lidar_p5_20261008_0300_1200/p5_table.html).
- `reference/`: 기존 A₂ 모델·가중치, P5 Python 구현·테스트, 로컬 설정, 분석 원본을 실제 복사했다. 원본 보존용이다.
- `core/`: 기본 파라미터의 C++ P5 수치 코어 및 명령행 검증기. ROS 노드·센서 입력·CAN 출력은 없다.
- `COPY_MANIFEST.json`: 복사 출처와 SHA256. 출처 경로는 이력이며 실행 의존성이 아니다.

검증: 이 폴더에서 `bash verify.sh`. CMake, C++17 컴파일러, Python3, numpy, pytest가 필요하다. 다른 프로젝트의 venv나 소스를 참조하지 않는다.

외부 Autoware/NVIDIA 소스와 모델은 **아직 설치·복사하지 않았다**. 백엔드 도입 단계에서 호환 버전과 가중치를 선정해 이 프로젝트 안에 고정한다. 원시 rosbag은 용량 때문에 복사하지 않았고 분석 결과만 포함했다.

`reference/analysis_snapshot`, `reference/localization_snapshot`, `reference/senior_code`의 원본 스크립트에는 과거 절대 경로·드라이버 설정이 남아 있다. 보관 자료이며 새 실행 경로에서 import/launch하지 않는다. 원본 재생 도구의 이식과 실시간 노드 구현은 후속 단계다.

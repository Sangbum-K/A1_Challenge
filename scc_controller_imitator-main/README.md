# scc_controller_imitator

`controller_imitator` 의 **이미지 확장판**. 기존 (trajectory + ego state) 입력에 **전방 이미지
2장 (현재 + Δyaw 회전 보정된 이전 프레임)** 을 추가하여 선행차 인지 등의 SCC 기능을
신경망이 얻도록 한다. **출력은 controller_imitator 와 완전 동일** (steer + accel_brake).

## 1. 입력·출력 차이

| 구분 | controller_imitator | **scc_controller_imitator** |
|------|---------------------|------------------------------|
| 입력 | trajectory + ego state | trajectory + ego state + **image_curr + image_prev_rot** |
| target | steer / accel_brake / **steer_accel_brake** | 동일 (기본 `steer_accel_brake`) |
| 출력 텐서 | `(B, T, D)` | `(B, T, D)` (동일) |
| horizon T | 40 | **30** (jiwon_e2e wp_rel_gt 길이) |

채널 순서·전후처리 계약은 `controller_imitator/PROJECT_STRUCTURE_BRIEF_KR.md` 와 일치.

## 2. 환경 구축

### 옵션 A. 기존 `ml_con` env 재사용 (권장)
```bash
conda activate ml_con
pip install -r requirements.txt
python scripts/sanity_check_env.py
```

### 옵션 B. 격리된 `scc_ml_con` env 생성
```bash
conda env create -f environment.yml
conda activate scc_ml_con
python scripts/sanity_check_env.py
```

### 또는 helper script
```bash
bash scripts/setup_env.sh   # SCC_ENV=ml_con (기본) 사용, env 활성 후 pip 설치 + sanity check
```

## 3. 디렉터리 구조

```
scc_controller_imitator/
├── config/                       # ini 설정
│   └── scc_config.ini
├── scripts/
│   ├── sanity_check_env.py       # 모델 forward 더미 통과 확인
│   └── setup_env.sh              # env 활성 + pip install
├── src/
│   ├── datamodule/
│   │   └── scc_dataset.py        # image + ego + trajectory dataset 스텁
│   └── model/
│       ├── models.py             # get_model(args)
│       ├── image_models/
│       │   └── image_scc_models.py   # ImageEgoSCCPredictor / ImageTrajEgoSCCPredictor
│       └── utils/common.py
├── outputs_multi/                # 학습 산출물 저장 경로
├── training_status/              # CSV 기반 실험 관리 연결 지점
├── data/                         # 전처리 캐시 (gitignore)
├── requirements.txt
├── environment.yml
└── .gitignore
```

## 4. 모델 (계획)

- `TwoImgTrajEgoController` — 이미지 2장 + trajectory + ego → **steer + accel_brake** 시퀀스
  - `LateralStripEncoder` — 단일 이미지 → (40, D) lateral 토큰. **백본 configurable**
    (`resnet18` / `efficientnet_b0` (기본) / `efficientnet_b1` / `tf_efficientnetv2_s` /
    `tf_efficientnetv2_b0/b1` / `mobilenetv3_large_100`. ImageNet pretrained, timm 기반.
    마지막 spatial map → AdaptiveAvgPool2d((1, 40)) → 1×1 Conv1d → (40, D)).
    **siamese 공유 가중치**.
  - `ImageFusionBlock` — (tokens_curr, tokens_prev) → image_memory (40, D). **3 단계 + 2 옵션 head**:
    (0) **per-frame self-attn (shared weights, 기본 1 layer)** — ≤10 m 근거리 객체의
    3~6 토큰 span 을 응집. (1) **cross-attn Q=curr', KV=prev'** — 객체 단위 curr↔prev
    매칭 (= 암묵적 tracking). (2) (옵션) post self-attn over 40 — 추가 정제.
    옵션 supervision head (모두 train-only, baseline 비활성):
    **(a) SegHead** — Step 0 출력 + spatial_feat → `(K=5, 8, 40)` 픽셀 분류 (jiwon_e2e SAM2 GT 재활용),
    **(b) AuxHead** — image_memory 분기, `[obj, v_lat, v_long, proximity]` 회귀.
    출력 토큰 i 는 lateral 위치 i 에 있는 객체의 상태를 담음.
  - Cross-attention (trajectory ↔ image_memory): Q=traj tokens (30, D), K/V=image_memory (40, D)
  - ego_mlp(5 → D) broadcast add 로 조건화
  - Output head: controller_imitator 와 동일 채널 규약 (`[control...] + (옵션) [ego x,y,heading,speed]`)

자세한 다이어그램은 `architecture.md`, 전체 설계는 `DESIGN.md` 참고.

**Note**: 현재 폴더에 있는 `image_scc_models.py` 의 `ImageEgoSCCPredictor` / `ImageTrajEgoSCCPredictor`
는 초기 SCC-전용 스텁이라 구현 단계에서 위 모델로 교체된다.

## 5. controller_imitator 와의 통합 포인트

- 학습 스크립트 / 자동화 (`auto_trainer.py`) 는 추후 `controller_imitator` 의 것을 fork 하여
  `scc_controller_imitator` 의 dataset / get_model 만 갈아끼우는 식으로 연결한다.
- CSV (`ML_Experiment_Table.csv`) 컬럼에 `model_name = ImageEgoSCCPredictor` 등을 추가해
  기존 실험 관리 흐름에 합류 가능.

## 6. 다음 작업 후보

- [ ] bag → (image, ctrl_io) 동기화 전처리 스크립트 (`ctrl_io_bag_preprocess_*.py` 패턴 차용)
- [ ] train / inference 스크립트 (`train_multi_output_*.py`, `inference_multi_output.py` fork)
- [ ] `ML_Experiment_Table.csv` 호환 컬럼 정의
- [ ] 평가 메트릭 (속도 오차, 가속 jerk 등 SCC 특화 지표)

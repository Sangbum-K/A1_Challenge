# SLURM 학습 wrapper — scc_controller_imitator

AILAB 클러스터(`partition-4090`, apptainer/.sif)에서 SCC controller_imitator 학습을
돌리기 위한 sbatch wrapper 모음. 학습 스크립트(`scripts/train_*.py`) 자체는 무수정이고,
경로(SCC_DATA_DIR)만 NFS 로 치환한다.

> 상태 표기: [confirmed]=AILAB-12 에서 확인 / [hypothesis]=클러스터에서 검증 필요 / [assumption]=가정

---

## 1. 제출 위치 (중요)

- AILAB-12 (`10.0.0.20`) 에는 **slurm client 없음** → `sbatch` 제출 불가. [confirmed]
- 제출/모니터는 반드시 **cluster login 노드**에서. (AILAB-12 는 데이터/스크립트 산출용)
- NFS 는 양쪽이 공유: AILAB-12 의 `/home/ailab-12/AILabSSD` == 클러스터 노드의
  `/home/ailab/AILabSSD` (동일 서버 `166.104.167.165:/home/ailab/AILabSSD`). [confirmed]
  - wrapper 안의 모든 절대경로는 클러스터 시점인 `/home/ailab/AILabSSD/...` 기준으로 작성.

---

## 2. 파일 목록

> **깨서 처음 보면 `SETUP_RUNBOOK.md` 부터.** SSH 키 등록 → workspace 동기화 → ckpts 복사 →
> env 검증 → 제출까지 단계별 절차 + 경로 매핑표 + 미해결 항목이 거기 정리돼 있다.

| 파일 | 용도 | GPU | 비고 |
|---|---|---|---|
| `SETUP_RUNBOOK.md` | **클러스터 구축 절차서** (가장 먼저) | — | SSH/sync/ckpts/env/제출 |
| `slurm_train_a2_multiframe.sh` | a_2 multi-frame head 학습 | gpu:1 | `launch_a2_multiframe.sh` 의 NFS 판. init=a2_v7b |
| `slurm_sweep_a2.sh` | a_2 hyperparam sweep (array) | gpu:1 | (lr×alpha×nf) 8조합. 본선 대량 |
| `slurm_train_prof635_side.sh` | prof635 frozen main + f_2 side(pedal) | gpu:1 | DATASETS 5종 default, main=main_prof654 |
| `slurm_train_iy_side.sh` | f_2 iy side cross-attn (+ci-aug) | gpu:1 | `launch_v25h_B.sh` 의 NFS 판. init=iy_v25c |
| `slurm_build_caches.sh` | (선택) i_2 v7b cache 재생성 | gpu:1 | 통상 학습엔 불필요(cache 이미 복사) |
| `slurm_gen_seg_masks_driverb.sh` | DriverB SAM seg mask | gpu:1 | SAM deps·meta json 선행(주석) |
| `slurm_train_ailab.sh` | (legacy) mmdetection3d viz | gpu:4 | 본 작업과 무관, 양식 참고용 |

제출 예시 (cluster login 노드):
```bash
sbatch slurm_train_a2_multiframe.sh a2_2f_yaw 2 on        # NAME NF YAW
sbatch --array=0-7%2 slurm_sweep_a2.sh                    # a_2 sweep 8조합, 동시 2
sbatch slurm_train_prof635_side.sh  prof635_distbrake distbrake brake_only  # NAME SUFFIX DELTA
sbatch slurm_train_iy_side.sh v25h_B2_cluster 4           # NAME SIDE_LAYERS
```

---

## 3. 경로 매핑 (cluster 시점)

| 항목 | 경로 | 상태 |
|---|---|---|
| `.sif` 이미지 | `/home/ailab/AILabSSD/01_SlurmWorkspace/a1_e2e_fixed_v2.sif` | [confirmed] 존재(05-25 17:55) |
| repo (WORKSPACE) | `/home/ailab/AILabSSD/03_Shared_Repository/jiwon/scc_controller_imitator` | [hypothesis] **아직 없음 — 생성 필요** |
| 데이터 (SCC_DATA_DIR) | `/home/ailab/AILabSSD/02_Data/57_A1_E2E/01_processed/scc_controller_imitator/data` | [confirmed] 복사 완료 |
| slurm 로그 | `/home/ailab/AILabSSD/99_Management/slurm-logs/` | [confirmed] 존재 |
| conda env | 컨테이너 내 `a1_learning` | [hypothesis] deps 검증 필요(§5) |

`SCC_DATA_DIR` 하위 구조 (모두 복사됨):
`cache/`, `prebake/`, `aux_targets_*/`, `a2_targets/`, `ckpts/`(repo data/ckpts 사본),
`control/`(repo data/control 사본).

> 학습 스크립트의 ckpt 기본 경로는 repo 상대(`data/ckpts/...`)지만, wrapper 는
> 안전하게 `${SCC_DATA_DIR}/ckpts/...` 절대경로로 명시 지정한다 → repo data/ckpts 의
> git 동기화 여부와 무관하게 동작. (data/ckpts 중 14/47 만 git-tracked) [confirmed]

---

## 4. repo NFS 사본 동기화 (제출 전 필수)

`WORKSPACE`(NFS repo) 가 **아직 없으므로** 최초 1회 생성, 이후 매 학습 전 동기화한다.
둘 중 택1:

```bash
# (A) git clone (권장) — origin = github.com/ailab-hanyang/scc_controller_imitator
cd /home/ailab/AILabSSD/03_Shared_Repository/jiwon
git clone https://github.com/ailab-hanyang/scc_controller_imitator.git
# 이후 업데이트: cd scc_controller_imitator && git pull

# (B) AILAB-12 에서 rsync (커밋 안 된 로컬 변경까지 반영하려면)
rsync -a --exclude='.git' --exclude='outputs*' --exclude='data/local' \
  /home/ailab-12/git/scc_controller_imitator/ \
  /home/ailab/AILabSSD/03_Shared_Repository/jiwon/scc_controller_imitator/
```

주의:
- 현재 작업 트리에 **uncommitted 변경/untracked launcher** 존재(train_iy.py, closed_loop_node.py,
  iy_model.py 등 M, launch_v25*.sh 등 ??). git clone 만 하면 이들이 빠진다. [confirmed]
  → 커밋되지 않은 학습 코드를 쓰려면 (B) rsync 또는 commit&push 후 clone.
- 학습 출력은 `WORKSPACE/outputs_multi/<NAME>/` 에 쓰인다(gitignored). NFS 쓰기 권한 확인.

---

## 5. 컨테이너 env 패리티 — 클러스터에서 검증 필요 [hypothesis]

`a1_e2e_fixed_v2.sif` 의 `a1_learning` env 에 우리 deps 가 있는지 **미확인**.
AILAB-12 의 `ml_con` env 기준값(reference): `torch==2.4.0`, `torchvision==0.19.0`,
`timm==1.0.26`, `numpy==2.4.4`. [confirmed: ml_con]

학습 의존(`requirements.txt`): `torch>=2.4`, `timm>=1.0`(seg backbone), `numpy`,
`Pillow`, `opencv-python`, `pandas`, `scipy`, `matplotlib`, `tqdm`, `wandb`.
- a_2 / prof635 학습 코어 import = `torch`, `numpy` 만. timm 은 seg backbone 경로에서 사용.

검증 절차 (cluster login 노드, 1회):
```bash
apptainer exec --nv /home/ailab/AILabSSD/01_SlurmWorkspace/a1_e2e_fixed_v2.sif \
  bash -lc 'source /opt/conda/etc/profile.d/conda.sh; conda activate a1_learning; \
    python -c "import torch,timm,numpy,cv2,PIL; \
      print(torch.__version__, torch.cuda.is_available(), timm.__version__, numpy.__version__)"'
```
- torch>=2.4 + `cuda.is_available()=True` + timm import OK 면 통과.
- 누락 시: (a) `.sif` 안에서 `pip install -r requirements.txt` 가능한지, 아니면
  (b) `.sif` 재빌드. **확정 전 학습 제출 금지.**
- numpy 2.x 관련: 학습은 무관하나, cv_bridge 등 ROS 경로는 numpy2 segfault 이력 있음(학습엔 미사용).

---

## 6. 미해결 / 가정 요약

| # | 항목 | 상태 | 액션 |
|---|---|---|---|
| 1 | `.sif` deps(torch/timm/cuda) 보유 | [hypothesis] | §5 검증, 없으면 재빌드 |
| 2 | NFS repo(WORKSPACE) 사본 | [hypothesis] 없음 | §4 로 생성+동기화 |
| 3 | uncommitted 학습 코드 반영 | [confirmed] 미반영 위험 | commit&push 또는 rsync |
| 4 | partition/gres 가용성 | [assumption] partition-4090 | login 노드 `sinfo` 확인 |
| 5 | wandb 자격증명 | [assumption] OFF 기본 | 필요시 컨테이너 WANDB_API_KEY |
| 6 | data/control 학습 직접 의존 | [confirmed] 없음 | 명세 따라 복사만(closed-loop 용) |
| 7 | 데이터 경로 SSD vs Dataset NFS | [confirmed] SSD(7× 빠름) 채택 | legacy 스크립트는 Dataset 사용했음 |

---

## 7. 데이터 복사 메모

- 복사 대상(학습 필수 subset): `cache`(27G), `prebake`(11G), `aux_targets_*`(1.5G),
  `a2_targets`, repo `ckpts`(1.3G), repo `control`. 총 ~41G.
- 제외: `scc_synth_*`, `scc_traj_bank_*`, `scc_logged_bank_*`, `seg_masks_*`,
  대형 manifest json (재합성/전처리용, SLURM 학습엔 불필요).
- 원천: `/home/ailab-12/AILabDatasetLocal/.../data` (로컬 NVMe, 클러스터 비가시).
- `i2_mem_cache_*_v7b_prev20.npz` 는 05-26 02:37~02:39 재생성 최신본 반영됨. [confirmed]

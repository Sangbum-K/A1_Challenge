# SLURM 클러스터 구축 RUNBOOK — scc_controller_imitator

본선 대량 실험(a_2 sweep, f_2/Side SCC, cache build, seg mask gen)을 AILAB 3080x4 클러스터에서
돌리기 위한 ready-to-run 절차서. **깨서 위→아래 순서대로 따라하면 제출까지 도달**하도록 작성.

> 상태 표기: [confirmed]=AILAB-12 에서 확인 / [hypothesis]=클러스터에서 검증 필요 / [assumption]=가정
>
> AILAB-12 (`10.0.0.20`) 에는 slurm client 도, login 노드 passwordless SSH 도 없음. [confirmed]
> → 이 문서/스크립트는 작성만 완료. **실제 제출은 사용자가 깨서 클러스터에서.**

---

## 0. 깨서 할 첫 3가지 (TL;DR)

| 순서 | 액션 | 절 |
|---|---|---|
| 1 | SSH 키 등록 (passwordless 로 login 노드 접속) | §1 |
| 2 | NFS workspace 생성 + repo 동기화 (rsync 권장 — uncommitted 코드 포함) | §3 |
| 3 | ckpts·control NFS 복사 보완 (현 rsync 가 빠뜨림) + .sif env 검증 | §4, §5 |

위 3개 끝나면 §6 제출 예시로 바로 sbatch.

---

## 1. SSH 키 등록 (가장 먼저 — 현재 passwordless 안 됨) [confirmed: 거부됨]

login 노드: `ssh -p 2208 ailab@166.104.167.165` (Host 3080x4). 현재 publickey/password 모두 거부.

```bash
# (1) AILAB-12 에 키 없으면 생성
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -C "ailab-12-scc" -f ~/.ssh/id_ed25519 -N ""

# (2) 키 등록 — 비밀번호를 1회 입력해야 함 (이 단계는 비번 필요)
ssh-copy-id -p 2208 -i ~/.ssh/id_ed25519.pub ailab@166.104.167.165

# (3) 검증 — 비번 없이 접속되면 성공
ssh -p 2208 ailab@166.104.167.165 'hostname; sinfo -s'
```

ssh-copy-id 가 막히면(비번 입력도 거부) 클러스터 관리자에게 `~/.ssh/id_ed25519.pub` 내용 수동 등록 요청
(login 노드 `~/.ssh/authorized_keys` 에 append). 이게 전체 작업의 **선행 blocker**.

편의 설정(선택) — `~/.ssh/config`:
```
Host ailab-login
    HostName 166.104.167.165
    Port 2208
    User ailab
    IdentityFile ~/.ssh/id_ed25519
```
→ 이후 `ssh ailab-login`, `rsync -e 'ssh' ... ailab-login:...` 로 단축.

> NFS 가 양쪽 공유라 rsync over ssh 대신 **로컬 cp/rsync(마운트 경유)** 만으로 데이터 이동 가능
> (§4 참조). SSH 는 sbatch 제출/모니터(§6) 용도가 핵심.

---

## 2. 경로 매핑 표 (AILAB-12 ↔ 클러스터) [confirmed via mount]

NFS 서버 `166.104.167.165` 두 export 가 양쪽에 동일 데이터로 마운트:

| 항목 | AILAB-12 (로컬 마운트) | 클러스터 (`/home/ailab/...`) | NFS export | 비고 |
|---|---|---|---|---|
| AILabSSD | `/home/ailab-12/AILabSSD` | `/home/ailab/AILabSSD` | `:/home/ailab/AILabSSD` (nfs4.2, 1MB) | 빠름(7×). 학습 데이터·.sif·로그 |
| AILabDataset | `/home/ailab-12/AILabDataset` | `/home/ailab/AILabDataset` | `:/volume3/AILabDataset` (nfs4.1) | 대용량 raw·기존 jiwon repo·ci-aug |
| 로컬 NVMe (원천) | `/home/ailab-12/AILabDatasetLocal/...` | — (**클러스터 비가시**) | — | rsync 원천. 직접 bind 불가 |

핵심 디렉토리 (클러스터 시점):

| 용도 | 클러스터 경로 | 상태 |
|---|---|---|
| `.sif` 이미지 | `/home/ailab/AILabSSD/01_SlurmWorkspace/a1_e2e_fixed_v2.sif` | [confirmed] 9.96GB, 05-25 17:55 |
| repo (WORKSPACE) | `/home/ailab/AILabSSD/03_Shared_Repository/jiwon/scc_controller_imitator` | [hypothesis] **부재 — §3 에서 생성** |
| 데이터 (SCC_DATA_DIR) | `/home/ailab/AILabSSD/02_Data/57_A1_E2E/01_processed/scc_controller_imitator/data` | [진행중] rsync 중(§4) |
| slurm 로그 | `/home/ailab/AILabSSD/99_Management/slurm-logs/` | [confirmed] 존재(쓰기가능) |
| ci-aug 데이터 | `/home/ailab/AILabDataset/02_Custom_Dataset/35_e2e_morai/controller_imitator/` | [confirmed] NFS 존재 |
| jiwon_e2e (seg gen용) | `/home/ailab/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing` | [confirmed] NFS 존재 |
| DriverB raw (seg gen용) | `/home/ailab/AILabDataset/02_Custom_Dataset/57_A1_E2E/00_raw` | [hypothesis] 경로 검증 필요 |

> 경로 주의 (중요): `AILabSSD` 에는 `03_Shared_Repository` 가 **없고** `04_Shared_Repository` 만 있음. [confirmed]
> 기존 jiwon repo 사본은 `AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator`(05-16 구버전) 에 존재.
> 본 runbook + 모든 wrapper 는 **WORKSPACE 를 `AILabSSD/03_Shared_Repository/jiwon/...` 로 신규 생성**하는 방향으로
> 통일한다(데이터·로그가 모두 빠른 AILabSSD 에 있으므로). §3 에서 디렉토리를 만든다.
> jiwon_e2e / ci-aug 등 raw·외부 자산은 기존 위치(AILabDataset)를 그대로 참조.

---

## 3. NFS workspace 생성 + repo 동기화 (제출 전 필수) [hypothesis: 부재]

`AILabSSD/03_Shared_Repository/jiwon/scc_controller_imitator` 는 **아직 없음**. 최초 1회 생성 후 매 학습 전 동기화.

### 권장: AILAB-12 → NFS rsync (uncommitted 코드·untracked launcher 포함)

현재 작업 트리에 commit 안 된 변경(train_iy.py, closed_loop_node.py, iy_model.py 등 M /
launch_v25*.sh 등 untracked)이 있어 **git clone 만으로는 누락**. [confirmed via git status]

```bash
# 클러스터 login 노드 또는 AILAB-12 (NFS 마운트 동일) 어디서든 가능.
# AILAB-12 에서 실행하는 예 (로컬 마운트 경로 사용):
DEST=/home/ailab-12/AILabSSD/03_Shared_Repository/jiwon/scc_controller_imitator
mkdir -p "$DEST"
rsync -a --info=progress2 \
  --exclude='.git' \
  --exclude='outputs_multi' --exclude='outputs*' \
  --exclude='data/local' --exclude='data/scc_synth_*' \
  --exclude='data/seg_masks_*' --exclude='data/*.npz' \
  --exclude='__pycache__' --exclude='*.pyc' \
  /home/ailab-12/git/scc_controller_imitator/ \
  "$DEST/"
```
- repo 의 `data/ckpts`·`data/control` 은 위 rsync 에 포함(코드가 repo 상대 경로로도 fallback).
  단, wrapper 는 안전하게 `${SCC_DATA_DIR}/ckpts`(절대) 를 명시 — §4 의 ckpts 복사가 본선.
- `outputs_multi/` 는 학습 출력이 쓰일 위치(gitignored). rsync 제외. 클러스터에서 자동 생성됨.

### 대안: git clone (commit&push 한 뒤에만)

```bash
cd /home/ailab/AILabSSD/03_Shared_Repository/jiwon
git clone https://github.com/ailab-hanyang/scc_controller_imitator.git
# 업데이트: cd scc_controller_imitator && git pull
```
→ **uncommitted 학습 코드가 빠지므로**, 이 방식은 AILAB-12 에서 commit & push 를 끝낸 경우에만.

---

## 4. ckpts·control NFS 복사 보완 (현 rsync 가 빠뜨림) [confirmed gap]

현재 AILAB-12 에서 돌고 있는 rsync 는:
```
rsync -a --include=cache/*** --include=prebake/*** --include=a2_targets/*** \
  --include=aux_targets_*/*** --exclude=* \
  /home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data/  →  <SCC_DATA_DIR>/
```
→ `cache/`, `prebake/`, `a2_targets/`, `aux_targets_*/` 만 복사. **`ckpts/`·`control/` 제외됨.** [confirmed]
또한 원천 로컬 data 디렉토리에는 `ckpts`/`control` 폴더 자체가 없음 → **원천은 repo 의 `data/ckpts`·`data/control`**. [confirmed]

보완 명령 (rsync 완료 후, 또는 병렬로):
```bash
SCC=/home/ailab-12/AILabSSD/02_Data/57_A1_E2E/01_processed/scc_controller_imitator/data
REPO=/home/ailab-12/git/scc_controller_imitator

# ckpts (1.3G) — wrapper 들이 ${SCC_DATA_DIR}/ckpts/{a2_v7b_...,main_prof654.pth} 를 절대경로로 참조
rsync -a --info=progress2 "$REPO/data/ckpts/"   "$SCC/ckpts/"
# control (0.4M) — closed-loop·raceline·pedal_map 용 (학습 직접 의존 X, 명세 따라 복사)
rsync -a --info=progress2 "$REPO/data/control/" "$SCC/control/"
```

복사 진척 확인:
```bash
du -sh $SCC/cache $SCC/prebake $SCC/aux_targets_* $SCC/ckpts $SCC/control 2>/dev/null
# 기대: cache ~27G, prebake ~11G, ckpts ~1.3G, control ~0.4M (합계 ~41G)
ls $SCC/ckpts/a2_v7b_d128_alpha075_best_loss.pt $SCC/ckpts/main_prof654.pth   # 핵심 init ckpt 2종 존재 확인
```

> 진행중 rsync 가 cache(27G)/prebake(11G) 를 다 옮길 때까지 대기. 깼을 때 `du -sh`(위) 로 완료 판정.
> cache/prebake 가 비어있으면 학습 job 이 입력 npz 누락으로 죽음 → 반드시 완료 후 제출.

---

## 5. 컨테이너 env 패리티 검증 (제출 전 1회 필수) [hypothesis]

`a1_e2e_fixed_v2.sif` 의 `a1_learning` conda env 에 학습 deps 가 있는지 **미확인**.
참조 기준(AILAB-12 `ml_con`): `torch==2.4.0`, `torchvision==0.19.0`, `timm==1.0.26`, `numpy==2.4.4`. [confirmed: ml_con]
학습 코어 import = `torch`, `numpy`. seg backbone 경로에서 `timm`. (a_2/prof635 학습엔 cv2/PIL 도 import.)

```bash
# login 노드에서 1회
SIF=/home/ailab/AILabSSD/01_SlurmWorkspace/a1_e2e_fixed_v2.sif
apptainer exec --nv "$SIF" bash -lc '
  source /opt/conda/etc/profile.d/conda.sh; conda activate a1_learning;
  python -c "import torch,timm,numpy,cv2,PIL,pandas,scipy; \
    print(torch.__version__, torch.cuda.is_available(), timm.__version__, numpy.__version__)"'
```
- `torch>=2.4` + `cuda.is_available()=True` + 전 import OK → 통과, §6 제출 진행.
- 일부 누락 시: 우선 `.sif` 내 `pip install -r requirements.txt` 시도(쓰기 불가하면 `--writable-tmpfs`
  는 휘발 → 영구 반영 안 됨). 영구 필요시 §5.1 재빌드.
- **검증 통과 전 학습 job 제출 금지.** (실패시 GPU 시간만 소모)

> seg mask gen(`slurm_gen_seg_masks_driverb.sh`)은 추가로 `groundingdino`+`sam2`+jiwon_e2e import 필요.
> 이건 학습 deps 와 별개로 .sif 에 거의 없을 가능성 높음 — 해당 wrapper 안에 preflight 가 있어 즉시
> fail 하게 돼 있음. seg gen 은 본선 학습과 분리해서 별도 검증.

### 5.1 .sif 재빌드 절차 요약 (deps 누락 시) — 환경 빌드 PDF 파이프라인

PDF 원문은 별도 보관. 요약 흐름:
1. Docker base: `nvidia/cuda:<12.1>-cudnn-devel-ubuntu<20.04>` 에서 시작.
2. miniconda 설치 → `conda create -n a1_learning python=3.11` (또는 기존 env 재현).
3. deps 설치: `pip install -r requirements.txt` (+ torch/torchvision cu121 빌드). 학습 코드로 import/smoke 검증.
4. `docker commit <container> ailab/ailab-docker-hub:<TAG>` → `docker save -o <TAG>.tar ailab/ailab-docker-hub:<TAG>`.
5. `singularity build <TAG>.sif docker-archive://<TAG>.tar` → 산출물을 `~/AILabSSD/01_SlurmWorkspace/` 로 이동.
6. wrapper 의 `SIF_IMAGE` 를 새 TAG 로 교체.

> [assumption] 위 단계 명령은 PDF 의 표준 절차 요약. 정확한 base 태그/버전은 PDF 확인 후 채울 것.

---

## 6. 제출 + 모니터 (검증 끝나면)

전제: §1(SSH), §3(WORKSPACE), §4(ckpts/control + cache/prebake 완료), §5(env OK) 통과.

```bash
ssh -p 2208 ailab@166.104.167.165              # 또는 ssh ailab-login
cd /home/ailab/AILabSSD/03_Shared_Repository/jiwon/scc_controller_imitator/slurm_script
sinfo -s                                        # partition/노드 가용성 (partition-4090 등)

# --- a_2 multi-frame head (NAME NF YAW) ---
sbatch slurm_train_a2_multiframe.sh a2_2f_yaw   2 on
sbatch slurm_train_a2_multiframe.sh a2_2f_noyaw 2 off
sbatch slurm_train_a2_multiframe.sh a2_3f_yaw   3 on

# --- a_2 sweep (lr/epoch/focal 등 대량) ---
sbatch slurm_sweep_a2.sh                         # 배열 job — §7

# --- f_2 / Side SCC (NAME PEDAL_SUFFIX DELTA) ---
sbatch slurm_train_prof635_side.sh prof635_distbrake distbrake brake_only
sbatch slurm_train_prof635_side.sh prof635_ttc       ttc       brake_only

# --- f_2 (iy / side cross-attn, ci-aug 포함) ---
sbatch slurm_train_iy_side.sh v25h_B2_cluster        # §7

# --- (선택) i_2 v7b cache 재생성 — 통상 불필요 ---
sbatch slurm_build_caches.sh
```

모니터:
```bash
squeue -u ailab                 # 내 job 상태
squeue --me                     # 동일
scontrol show job <JOBID>       # 상세
tail -f /home/ailab/AILabSSD/99_Management/slurm-logs/<job-name>_<jobid>.out
scancel <JOBID>                 # 취소
```
대시보드(내부망): `http://166.104.167.165:3080/#`

산출물: `<WORKSPACE>/outputs_multi/<NAME>/best_loss.pt` (또는 `best.pt`).
필요시 AILAB-12 마운트(`/home/ailab-12/AILabSSD/03_Shared_Repository/...`) 로 회수.

---

## 7. sbatch 스크립트 인벤토리

| 파일 | 용도 | GPU | time | partition | 비고 |
|---|---|---|---|---|---|
| `slurm_train_a2_multiframe.sh` | a_2 multi-frame head | gpu:1 | 24h | partition-4090 | init=a2_v7b. NAME NF YAW |
| `slurm_sweep_a2.sh` | a_2 hyperparam sweep | gpu:1 | 24h | partition-4090 | **신규** — array job (lr×epoch×focal) |
| `slurm_train_prof635_side.sh` | prof635 frozen main + f_2(pedal) | gpu:1 | 12h | partition-4090 | DATASETS 5종. NAME SUFFIX DELTA |
| `slurm_train_iy_side.sh` | f_2 iy side cross-attn (+ci-aug) | gpu:1 | 24h | partition-4090 | **신규** — launch_v25h_B 의 NFS 판 |
| `slurm_build_caches.sh` | (선택) i_2 v7b cache 재생성 | gpu:1 | 12h | partition-4090 | 통상 불필요 |
| `slurm_gen_seg_masks_driverb.sh` | DriverB SAM seg mask | gpu:1 | 48h | partition-4090 | SAM deps·meta json 선행(절 주석) |
| `slurm_train_ailab.sh` | (legacy) mmdet3d viz | gpu:4 | — | partition-4090 | 본 작업 무관, 양식 참고용 |

> 모든 신규/보완 wrapper: `SCC_DATA_DIR` = `${AILabSSD}/.../data` (NFS, 빠름), `WORKSPACE` =
> `AILabSSD/03_Shared_Repository/jiwon/scc_controller_imitator`, ckpt init 은 `${SCC_DATA_DIR}/ckpts/...` 절대경로.
> partition 명(`partition-4090`)·gres 는 [assumption] — `sinfo -s` 로 실제 명칭 확인 후 필요시 일괄 치환.

---

## 8. 미해결 / 가정 (제출 전 반드시 확인)

| # | 항목 | 상태 | 액션 |
|---|---|---|---|
| 1 | login 노드 passwordless SSH | [confirmed] 안 됨 | §1 ssh-copy-id / 관리자 수동 등록 |
| 2 | `.sif` 학습 deps(torch/timm/cv2/numpy/cuda) | [hypothesis] 미검증 | §5 검증, 없으면 §5.1 재빌드 |
| 3 | NFS WORKSPACE 사본 | [hypothesis] 부재 | §3 rsync 생성 (uncommitted 포함) |
| 4 | cache/prebake rsync 완료 | [진행중] | §4 `du -sh` 로 완료 판정 후 제출 |
| 5 | ckpts/control NFS 복사 | [confirmed] 누락 | §4 보완 rsync |
| 6 | partition/gres 실제 명칭 | [assumption] partition-4090 | `sinfo -s` 확인 |
| 7 | wandb 자격증명 | [assumption] OFF 기본 | 필요시 컨테이너 `WANDB_API_KEY` env |
| 8 | seg gen .sif SAM deps + meta json | [hypothesis] 거의 없음 | wrapper 내 preflight 가 fail 시킴. 별도 처리 |
| 9 | `AILabSSD/03_Shared_Repository` 신규 생성 vs `AILabDataset/03` 기존 사용 | [confirmed] AILabSSD 채택 | §2 주의 / §3 mkdir |
| 10 | partition 별 GPU(3090/4090/A6000) 메모리 vs batch | [assumption] | OOM 시 batch-size↓ 또는 4090/A6000 지정 |

---

## 9. 데이터 복사 메모

- 복사 대상(학습 필수 subset): `cache`(~27G), `prebake`(~11G), `aux_targets_*`(~1.5G),
  `a2_targets`, repo `ckpts`(~1.3G), repo `control`(~0.4M). 총 ~41G.
- 제외(재합성/전처리용, SLURM 학습 불필요): `scc_synth_*`, `scc_traj_bank_*`, `scc_logged_bank_*`,
  `seg_masks_*`, 대형 manifest json.
- 원천: `/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data`(로컬 NVMe, 클러스터 비가시) +
  repo `data/ckpts`·`data/control`(ckpts/control 원천).
- f_2(iy side) 가 쓰는 `i2_mem_cache_*_{v6,v7b}_prev20.npz`, `pedal_label_*_{ttc,distbrake}.npz`,
  `inlane_gt_*.npz`, `f1_prebake_*_{fixed,trajfix,brakeaug}_prev20.npz` 모두 source cache/prebake 에 존재 →
  rsync 패턴(`cache/***`,`prebake/***`)에 포함됨. [confirmed]
- ci-aug(`controller_imitator_data_multi_valid_v0237`)는 별도 경로 → AILabDataset NFS 에 이미 존재,
  복사 불필요. wrapper 가 `/home/ailab/AILabDataset/02_Custom_Dataset/35_e2e_morai/...` 로 직접 bind. [confirmed]

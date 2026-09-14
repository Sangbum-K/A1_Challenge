# 학습 (어느 모듈을 어떻게)

학습 순서: **seg pretrain → a_2 → f_2**. f_1(#654)은 frozen(재학습 안 함). seg/a_2 는 글로벌·저빈도(재학습 시 하류 cascade), f_2 만 신규 데이터로 반복.

현재 best ckpt 레지스트리: `data/ckpts/CURRENT_BEST.json` (script 들이 참조; 신규 학습 후 수동/post-hook 갱신).

---

## 1. seg pretrain — `scripts/train_seg_pretrain.py`

- 모델 `SegPretrainModel` (EfficientNetV2-S + U-Net 디코더, 4-class).
- GT = SAM seg masks (`generate_image_seg_masks.py` 산출, `seg_masks_v2/{frame}.npz`). 데이터셋 `seg_pretrain_dataset.py`.
- loss = cross-entropy (ignore=-100). default ep20, bs8, lr3e-4.
- 출력: `best_backbone.pt`(i_1 encoder 호환, 핵심) + `best_full.pt`(seg head 포함, debug overlay 용).
- 현행: `seg_pretrain_fx640_v3` (mIoU 0.921). **재학습 시 모든 i1/i2 캐시 + a_2 + f_2 cascade.**

---

## 2. a_2 (AuxHead) — `train_a2_head.py` (full) / `train_a2_v5_from_cache.py` (cache-only)

- **full** (`train_a2_head.py`): raw 이미지 pair 에서 `A2Model` end-to-end. loss = obj BCE + masked regression(scale v=20, dist=40). 인자 `--pretrained`(seg backbone 주입), `--d-out 128`, `--freeze-enc-fusion`, `--close-lead-meta --close-weight 3.0`(근거리 lead 가중), `--w-obj/--w-reg`. ep25 bs12 lr2e-4. 데이터 `a2_pair_dataset.py` (aux_targets `(40,7)` GT).
- **cache-only** (`train_a2_v5_from_cache.py`): i_1 캐시 + aux_targets 만(NFS 이미지 불필요). encoder.proj+fusion freeze, **fc_proj+head(~33K)만 학습**. LDS inverse-density reweight, `predict_var` 지원. `--i1-cache --aux-dir --resume-from --d-out 128`.
- 현행 ckpt: `a2_v7b_d128_alpha075_best_loss.pt` (focal α=0.75, reg loss fix 후). **알려진 한계: vL_rel corr 0.40** (상대속도 지각 약함).

aux_targets GT 생성: `prepare_a2_aux_targets.py` (raw object → per-column (40,7)).

---

## 3. f_2 (side, prof635) — `scripts/train_prof635_side.py` (현행 핵심)

```
bash scripts/launch_prof635_v6_driverb.sh        # 5 datasets, 인자 없이 전부
# 또는
python scripts/train_prof635_side.py --name <run> --epochs 25 --batch-size 512 --lr 3e-4 [--wandb]
```
- 모델 `Prof635SideModel` (main #654 frozen + f_2 side). `--main-ckpt data/ckpts/main_prof654.pth`.
- 데이터(5): sangam_lead_0520/0521, sangam_pertO_v2, sangam_pertX_v2, **DriverB_new**(v6~). 각 = prebake + i2_mem_cache(v7b) + pedal_label. ConcatDataset, val 5%.
- loss = `L1(final_pedal, pedal_label)` (mask0=step0). Adam lr3e-4, ep25, bs512, side 4L. main frozen → trainable ~1.08M.
- **`--pedal-suffix`**: `ttc`(코드 default, closing ACC, legacy) / `distbrake`(거리-onset 곡선, **현행 배포 라벨**). `pedal_label_{ds}_{suffix}.npz` 로드. 주의: default 가 `ttc` 라 배포(distbrake_v1) 재현 시 `--pedal-suffix distbrake --delta-mode brake_only` 명시 필수.
- **`--gate-mode`**: 학습은 `open`(g=1, 전 데이터 학습) 권장 — 추론(node)은 `fixed_thresh` 로 게이팅. `--delta-mode`: `brake_only`(기본, 제동만) / `signed`.
- 진단 옵션 `--inject-gt-vrel`: GT 상대속도(inlane closing)를 f_2 에 주입 → perception vs capacity 격리 실험용.
- 결과: **현행 배포 = `prof635_side_distbrake_v1`**(distbrake 라벨 + brake_only, 5 dataset 975K, val_pedal~0.52[no-lead 지배]; 라벨이 공격적이라 L1 절대값↑ — ttc val 과 직접 비교 불가). legacy: v5 best val_pedal **0.0237**(signed/ttc), steer MAE 3.5°(frozen #654); v6 best **0.0443**(DriverB, 도심 stop&go 분산↑).
- distbrake 예시: `--datasets <5개> --pedal-suffix distbrake --gate-mode open --delta-mode brake_only --name prof635_side_distbrake_v1`. [REPORT §9]

> 주의: gate=fixed_thresh 면 no-lead(gate≈0)는 delta gradient≈0 → f_2 가 lead 샘플(~16%)에서만 학습. lead-upweight 검토 여지(STATE_AND_LESSONS §ACC).

---

## 4. legacy f_1/f_2 — `train_iy.py`

IyModel(§ARCHITECTURE 5) 학습기. close-lead oversample, ttc-augment, side-branch(cross-attn/ControlNet/min-arb) 플래그. v15~v25k 시리즈. prof635 로 대체됨 — 신규 학습은 prof635 사용. (launcher: `launch_v25*.sh`)

---

## 5. 항상 지키는 규칙

- **warm start**: 모든 학습은 prior best ckpt 를 init (from-scratch 금지). f_1 init 기본 = #654.
- **fx=640 only** 데이터 (MultiAgnet fx=320 폐기).
- 신규 데이터 추가 순서: 처리(seg→object→control GT) → 학습(seg pretrain→a_2→f_2). [DATA_PIPELINE.md]
- 합성 가동 시 학습용 CPU 2 스레드 여유 (24-core → synth 22 shard max).
- 학습 출력 `outputs_multi/{name}/` (gitignored). wandb: project=`scc-controller-imitator`, entity=`hmg_e2e_controller_imitator`.

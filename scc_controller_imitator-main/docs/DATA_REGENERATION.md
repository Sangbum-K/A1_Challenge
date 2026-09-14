# 데이터 / ckpt cold-start 재생성 가이드

본선 종료 후 데이터 cache (~190 GB) 삭제. **이 문서 = raw 만 있는 머신에서 처음부터 학습 cache 와
본선 submit ckpt 5 종까지 만드는 순서.**

배경 / 의미 / 포맷 등은 다른 문서에 — 여기는 *"순차 실행 명령"* 만.

| 보강할 것 | 참조 문서 |
|---------|---------|
| 데이터 자산 어디에 얼마나 / 어느 학습에 쓰이나 | [DATA_INVENTORY.md](DATA_INVENTORY.md) |
| 단계별 스크립트 / 입출력 / DAG | [DATA_PIPELINE.md](DATA_PIPELINE.md) |
| 산출물 의미 / 포맷 / suffix 사전 | [DATA_PIPELINE_GLOSSARY.md](DATA_PIPELINE_GLOSSARY.md) |
| 본선 직전 데이터 재감사 결과 | [DATA_STATUS_2026-05-27.md](DATA_STATUS_2026-05-27.md) |
| 모델 4 모듈 구조 | [ARCHITECTURE.md](ARCHITECTURE.md) |
| 본선까지의 9 일 디테일 | [BLOG_READABLE_2026-05-19_to_28.md](BLOG_READABLE_2026-05-19_to_28.md) |

---

## 0. 사전 준비 (한 번만)

```bash
# 환경
conda activate ml_con   # NumPy / PyTorch / timm / pynvml / psutil

# raw 데이터 위치 확인 (SSD 우선)
ls ~/AILabSSD/02_Data/57_A1_E2E/00_raw/       # sangam_track / speedway_track / sangam_qlf
ls ~/AILabDataset/02_Custom_Dataset/57_A1_E2E/00_raw/   # 대체 (NFS, 7× 느림)

# 출력 root
export SCC_DATA_DIR=$HOME/AILabDatasetLocal/scc_controller_imitator/data
mkdir -p $SCC_DATA_DIR/{cache,prebake,raw_local}

# 분리 ckpt (data/ckpts/ 에 git tracked) — 본선 직전 마지막 ckpt 들 보존됨
ls data/ckpts/seg_pretrain_fx640_v3_backbone.pt    # seg backbone (frozen)
ls data/ckpts/a2_mf20_v2s_n20_db_spd.pt            # a_2 + a_2_aux (mf20 21f)
ls data/ckpts/a2_v7b_d128_alpha075_best_loss.pt    # a_2 v7b (예선용 2f)
ls data/ckpts/main_prof654.pth                     # f_1 (Professor #654, frozen)
ls data/ckpts/scc_p3_dbspd_ng_distill_f656.pt      # f_2 (side)
ls data/ckpts/learned_temporal_gate_mlp_{beat,distill}.pt   # gate
```

GPU — RTX 3080 Ti 12 GB 기준. 학습은 batch=64 가 안전.

---

## 1. raw → 학습 cache (cold-start full chain)

각 데이터셋 별로 동일 sequence. 아래는 *sangam_lead_0520* 예시.

### Stage 1 — seg pretrain (글로벌, 1 회성, **신중**)

```bash
# 새 seg backbone 학습 = 모든 i_1 / i_2 cache 무효화 cascade.
# 이미 학습된 seg_pretrain_fx640_v3 있으면 SKIP.

# (a) seg mask 생성 (GroundingDINO + SAM2)
python scripts/generate_image_seg_masks.py \
    --raw-dir ~/AILabSSD/02_Data/57_A1_E2E/00_raw/sangam_track/0520_agent1_pertX_pitX \
    --out-dir $SCC_DATA_DIR/seg_masks_v2/sangam_lead_0520

# (b) seg pretrain (EfficientNetV2-S)
bash scripts/train_seg_pretrain.sh     # ~12h, RTX 3080 Ti
# → data/ckpts/seg_pretrain_fx640_v3_backbone.pt
```

### Stage 2 — manifest 생성

```bash
# 신규 데이터셋이면 manifest 만들기
python scripts/build_manifest.py \
    --track sangam_track --policy 0520_agent1_pertX_pitX \
    --name sangam_lead_0520 \
    --control-gt logged   # 또는 mpc_synth (MPC 합성 GT)
# → manifest_i1cache_sangam_lead_0520.json
```

### Stage 3 — i_1 cache (per-image latent, 학습의 hot path)

```bash
python scripts/compute_i1_tok_cache.py \
    --manifest manifest_i1cache_sangam_lead_0520.json \
    --backbone data/ckpts/seg_pretrain_fx640_v3_backbone.pt \
    --out $SCC_DATA_DIR/cache/i1_tok_cache_sangam_lead_0520_prev20.npz \
    --prev-offsets 5 20
# → (N_frame, 40, 256) — 약 1~3 GB / 데이터셋. GPU 시간 ~30 min / 1 만 frame
```

### Stage 4 — prebake (학습 dataset 직렬화)

```bash
python scripts/prebake_iy_dataset.py \
    --manifest manifest_i1cache_sangam_lead_0520.json \
    --out $SCC_DATA_DIR/prebake/f1_prebake_iy_sangam_lead_0520_prev20.npz
# → traj 40×3 + ego_5d + target steer/ax + mask + d_yaw
# 약 500 MB / 데이터셋. CPU 시간 ~10 min
```

### Stage 5 — i_2 mem cache (a_2 fusion 결과, f_2 cross-attn 입력)

```bash
# 본선 mf20 (21 frame multi-frame)
python scripts/build_i2_cache.py \
    --i1-cache $SCC_DATA_DIR/cache/i1_tok_cache_sangam_lead_0520_prev20.npz \
    --prebake $SCC_DATA_DIR/prebake/f1_prebake_iy_sangam_lead_0520_prev20.npz \
    --a2-ckpt data/ckpts/a2_mf20_v2s_n20_db_spd.pt \
    --multi-frame --n-frames 21 --prev-offsets 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 \
    --out $SCC_DATA_DIR/cache/i2_mem_mf20_sangam_lead_0520.npz \
    --ego-motion 0
# → (N_frame, 40, 128) i2_mem + (40, 7) a2_logits
# 약 1 GB / 데이터셋. GPU 시간 ~15 min
```

**중요** — `--ego-motion 0` 이 필수. closed_loop 추론 시 같은 setting 이라야 i_2 분포 일치 ([STATE_AND_LESSONS.md](STATE_AND_LESSONS.md) i_2 mismatch 사고).

### Stage 6 — inlane GT (in-lane lead 객체 거리 / vL_rel)

```bash
python scripts/build_inlane_gt_cache.py \
    --manifest manifest_i1cache_sangam_lead_0520.json \
    --prebake $SCC_DATA_DIR/prebake/f1_prebake_iy_sangam_lead_0520_prev20.npz \
    --out $SCC_DATA_DIR/cache/inlane_gt_sangam_lead_0520.npz
# → keys + inlane_dist + inlane_closing
```

### Stage 7 — pedal label (brake floor → pedal_map)

```bash
# 본선 distbrake (거리-onset)
BRAKE_MODE=distbrake python scripts/gen_ttc_pedal_labels.py \
    --prebake $SCC_DATA_DIR/prebake/f1_prebake_iy_sangam_lead_0520_prev20.npz \
    --inlane-gt $SCC_DATA_DIR/cache/inlane_gt_sangam_lead_0520.npz \
    --out $SCC_DATA_DIR/cache/pedal_label_sangam_lead_0520_distbrake.npz

# 비교용 ttc (closing 기반)
BRAKE_MODE=ttc python scripts/gen_ttc_pedal_labels.py \
    ... --out $SCC_DATA_DIR/cache/pedal_label_sangam_lead_0520_ttc.npz
```

### Stage 8 — 정합 검증 (gate)

```bash
python scripts/verify_cache_integrity.py \
    --i1 $SCC_DATA_DIR/cache/i1_tok_cache_sangam_lead_0520_prev20.npz \
    --prebake $SCC_DATA_DIR/prebake/f1_prebake_iy_sangam_lead_0520_prev20.npz \
    --i2 $SCC_DATA_DIR/cache/i2_mem_mf20_sangam_lead_0520.npz \
    --inlane $SCC_DATA_DIR/cache/inlane_gt_sangam_lead_0520.npz \
    --pedal $SCC_DATA_DIR/cache/pedal_label_sangam_lead_0520_distbrake.npz
# → 모든 key 가 1:1 매칭, frame_id 같은지 확인
```

---

## 2. 데이터셋 별로 반복

위 Stage 2~8 을 다음 dataset 별로 반복:

| 데이터셋 | track / policy | 용도 |
|---------|---------------|------|
| sangam_lead_0520 | sangam_track / 0520_agent1_pertX_pitX | main / a_2 / f_2 |
| sangam_lead_0521 | sangam_track / 0521_agent1_pertO_pitX | main / a_2 / f_2 |
| sangam_pertO_v2 / pertX_v2 | sangam_track / 0519_pert{O,X}_pitX | a_2 augment |
| speedway_lead_0520 | speedway_track / 0520_agent1_pertX_pitX | speedway 학습 |
| DriverB_new | (별도 driver run) | a_2 OOD 보강 (5/25 의 v6 부터) |
| spd_0522 (speedway 신규) | speedway_track / 260522 | 본선 마지막 합류 (5/27 새벽 6 stage 가공) |

spd_0522 의 6 stage 가공은 `scripts/run_phase2_a2db_rebuild.sh` 가 자동화 — 본선 전날 새벽
05/27 03:00 ~ 04:30 에 만든 chain.

---

## 3. 학습 (cache 다 만든 후)

3-stage 순서, 각 stage `warm-start` (prior best ckpt 로 init):

```bash
# Stage 1 — seg pretrain (위 Stage 1 에서 이미 했으면 skip)

# Stage 2 — a_2 + a_2_aux (mf20 21 frame, 본선용)
bash scripts/train_a2_n20_db_spd.sh
# → data/ckpts/a2_mf20_v2s_n20_db_spd.pt
# 약 3~5h on RTX 3080 Ti, val L1 ~0.025

# Stage 3 — f_1 main (Professor #654 frozen 이라 학습 불필요, ckpt 그대로 사용)
# data/ckpts/main_prof654.pth

# Stage 4 — f_2 side (p3 dbspd 본선)
bash scripts/train_p3_ng_long.sh   # cosine restarts, 140 epoch
# → data/ckpts/scc_p3_dbspd_ng_distill_f656.pt, side loss 0.0344

# Stage 5 — f_2 side (p5 dbspd2 본선 보수)
bash scripts/train_p5_dbspd2.sh
# → data/ckpts/scc_p5_dbspd2.pt, side loss 0.0322

# Stage 6 — gate (TemporalMLP K=16 beat / distill)
bash scripts/train_gate_temporal.sh
# → data/ckpts/learned_temporal_gate_mlp_{beat,distill}.pt
```

---

## 4. 본선 / 예선 submit ckpt 5 종 재생성

분리 ckpt + export 스크립트 → 통합 단일 .pt:

```bash
# 본선 3 종
python scripts/export_submit_weights.py --profile p3 --variant db        --out submit/scc_p3_db.pt
python scripts/export_submit_weights.py --profile p3 --variant dbspd     --out submit/scc_p3_dbspd.pt
python scripts/export_submit_weights.py --profile p3 --variant dbspd2    --out submit/scc_p3_dbspd2.pt
python scripts/export_submit_weights.py --profile p5 --variant dbspd2    --out submit/scc_p5_dbspd2.pt

# 예선 1 종 (qualifier = prof635 + a_2 v7b + beat gate)
python scripts/export_qualifier_weights.py --out submit/qualifier.pt
```

각 파일 ~176 MB. **검증** — 분리 로드와 통합 .pt forward 결과 maxdiff < 1e-6:

```bash
python scripts/verify_e2e_export.py --ckpt submit/scc_p3_dbspd2.pt
# → maxdiff: 1.8e-7 expected
```

---

## 5. closed_loop 실행 (운영)

```bash
# 본선
export E2E_CKPT=submit/scc_p3_dbspd2.pt
bash multicar/run_multicar.sh

# 또는 단독
TRACK=sangam MAP=mild bash scripts/run_closed_loop_prof635.sh
```

자세한 운영 옵션 → [multicar/RUN.md](../multicar/RUN.md).

---

## 6. 시간 / 디스크 estimate (cold-start 전체)

| 단계 | 시간 (RTX 3080 Ti) | 디스크 |
|------|-------------------|--------|
| seg mask 생성 (SAM v2) | ~4 h / 데이터셋 | ~5 GB / 데이터셋 |
| seg pretrain | ~12 h (글로벌 1 회) | (ckpt 77 MB) |
| i_1 cache | ~30 min / 데이터셋 | ~2 GB / 데이터셋 |
| prebake | ~10 min / 데이터셋 | ~500 MB / 데이터셋 |
| i_2 mem cache (mf20) | ~15 min / 데이터셋 | ~1 GB / 데이터셋 |
| inlane GT | ~3 min / 데이터셋 | ~50 MB / 데이터셋 |
| pedal label | ~2 min / 데이터셋 | ~50 MB / 데이터셋 |
| **합 / 데이터셋 (Stage 1 제외)** | **~1 h / 데이터셋** | **~3.5 GB / 데이터셋** |
| 6 데이터셋 합 | ~6 h | ~21 GB |
| a_2 mf20 학습 | ~5 h | (ckpt 84 MB) |
| f_2 side 학습 (p3 + p5) | ~8 h | (ckpt 8.8 MB × N) |
| gate 학습 | ~30 min | (ckpt 8 KB) |
| **전체 cold-start total** | **~30 h** | **~25 GB + raw 원천 별도** |

(seg pretrain 의 12 h 가 critical path. 그게 cached 되어 있으면 ~18 h.)

---

## 7. 검증 — cache 정합

학습 cache 와 closed_loop 추론 의 i_2 가 같은 분포인지 확인 (5/25 / 5/27 의 i_2 mismatch 사고
재발 방지):

```bash
python scripts/verify_i2_live_vs_cache.py --mf20 \
    --ckpt data/ckpts/a2_mf20_v2s_n20_db_spd.pt \
    --cache $SCC_DATA_DIR/cache/i2_mem_mf20_sangam_lead_0520.npz
# → cos similarity > 0.99 expected
```

cos < 0.95 이면 — seg backbone path / ego_motion 설정 / a_2 cache vs 추론 분포 분기 확인.
[STATE_AND_LESSONS.md](STATE_AND_LESSONS.md) 의 i_2 mismatch checklist 참조.

---

## 8. memory cross-reference

본선 9 일 사이클의 회고는 [BLOG_READABLE_2026-05-19_to_28.md](BLOG_READABLE_2026-05-19_to_28.md)
참조. 특히:

- **i_2 mismatch 두 번** (5/25 / 5/27 ★lesson) — § 7 / § 9.4 + 부록 H.0
- **a_2 v7 reg loss bug** (vL_rel pred std = 0 silent) — 부록 H.0
- **a_2 따로 학습해 놓고 안 쓴 사건** (5/25 v25h-A) — 부록 H.0.5
- **SPD 미포함 새벽 1:43 발견** (5/27) — § 9 Step 4.5
- **9 일 진화 표 (v3 → mf20)** — 부록 H.1 + H.2~H.8

---

*마지막 갱신: 2026-05-29 (프로젝트 종료 + 데이터 정리 시점). 본선 ckpt 분리본 (data/ckpts/) 보존,
submit/*.pt 통합본은 위 §4 로 재생성.*

# SCC Controller Imitator — 학습 데이터 인벤토리

최종 갱신: 2026-05-27 (자산 현황 ls 기준 전면 재작성)

이 문서는 "**어떤 데이터 자산이 어디에 얼마나 있고, 어느 학습 단계에 쓰이는가**" 의 현황 지도다.
파이프라인 *어떻게 만드는가* 와 용어 정의는 중복하지 않고 링크한다:

- 단계별 스크립트/입출력/DAG: [DATA_PIPELINE.md](DATA_PIPELINE.md)
- 각 산출물 의미·포맷·naming suffix 사전: [DATA_PIPELINE_GLOSSARY.md](DATA_PIPELINE_GLOSSARY.md)
- 모델 4모듈 (seg/a_2/f_1/f_2) 구조: [ARCHITECTURE.md](ARCHITECTURE.md)
- 모듈별 실험 history: [MODULE_EXPERIMENT_HISTORY.md](MODULE_EXPERIMENT_HISTORY.md)

---

## 0. 신규 합류자용 요약 (먼저 읽기)

이 프로젝트는 카메라 2장(현재+200ms 전) + 자차 state + raceline 으로 SCC(Smart Cruise Control) 제어
명령(조향 steer, 가감속 ax)을 모방학습한다. 핵심 학습 목표 = **두 이미지에서 앞차 상대속도(vL_rel)와
다가오는 차를 구별** 하는 것.

데이터 흐름은 크게 두 갈래:

```
(A) 이미지 갈래:   raw 카메라 ─SAM→ seg_masks ─seg pretrain→ seg backbone(frozen)
                                                              │ forward
                  raw 카메라 ───────────────────────────────► i1_tok_cache (image→token)
                                                              │ + a_2
                                                              ► i2_mem_cache (fusion 결과)
(B) 제어 GT 갈래:  raw EGO/OBJECT ─MPC합성 or logged→ scc_*_bank ─prebake→ prebake npz
                  raw OBJECT ──────────────────────► aux_targets (a_2 GT)
                                                      inlane_gt / pedal_label (f_2 GT)

학습:   seg → a_2 → f_1(main) → f_2(side)  순서 (글로벌 단계인 seg/a_2 먼저)
```

핵심 규약 (꼭 기억):

| 항목 | 값 | 근거 |
|---|---|---|
| 학습 카메라 fx | **fx=640 만 사용.** fx=320(MultiAgnet)은 2026-05-21 폐기 | memory `fx640_only` |
| prev frame 간격 | `_prev20` = 200ms 전 (현재 표준). `_prev5` 는 legacy | glossary §12 |
| seg backbone | `fx640_v3` (EfficientNetV2-S, mIoU 0.921) frozen — **재학습 시 모든 i1/i2 cache cascade** | ARCHITECTURE §1 |
| 진짜 로컬 디스크 | `AILabDatasetLocal` (=NVMe). `AILabSSD`/`AILabDataset` 은 둘 다 NFS(166.104.167.165) | memory `storage_speeds` |
| wheel index | FL=0 FR=1 RL=2 RR=3 (해당 코드/plot) | global 규약 |

이 문서에서 `$D` = `~/AILabDatasetLocal/scc_controller_imitator/data` (로컬 NVMe 작업 루트).

---

## 1. 저장소 위치 / 속도

| 경로 | 시스템 | 속도(체감) | 무엇이 사는가 | 재생성 |
|---|---|---|---|---|
| `$D` (`~/AILabDatasetLocal/scc_controller_imitator/data`) | **로컬 NVMe** (/dev/nvme0n1p2) | **~3300 MB/s** | cache / prebake / aux_targets / synth / bank / meta — 모든 작업 데이터 | 가능 (raw 에서 재생성) |
| `~/AILabSSD/02_Data/57_A1_E2E/00_raw/` | NFS 4.2 (rsize 1MB) | ~14 MB/s | 신규 raw (sangam_track / speedway_track, 0519~0522) | **원천 (재생성 불가)** |
| `~/AILabDataset/02_Custom_Dataset/57_A1_E2E/00_raw/` | NFS 4.1 (rsize 128KB) | ~2 MB/s | 구 raw (DriverB_new, MultiAgentSCC 등) | **원천** |
| `~/AILabDataset/03_Shared_Repository/jiwon/...` | NFS | 느림 | 원천 GT(seg_masks_v2, multiagnet_compact_meta) + 공유 ckpt, rosbag | **원천/공유** |
| `~/.../git/scc_controller_imitator/data/` (repo) | repo 로컬 | — | ckpts(git 추적), control(raceline/pedal_map), multiagnet_compact.npz | ckpt=배포물, control=원천 |

> 이름은 "SSD" 지만 `AILabSSD` 는 NFS 다. 단 `AILabDataset` 보다 7× 빠르다.
> 타 머신엔 `AILabDatasetLocal` 이 없으므로 캐시 출력이 자동으로 `repo/data/local`(gitignored)로 떨어진다
> ([DATA_PIPELINE.md §5](DATA_PIPELINE.md) 경로 portability).

현재 `$D` 디스크 사용: 전체 **~112 GB** (cache 31G, prebake 11G, 나머지 synth/bank/aux_targets/manifest).

---

## 2. 데이터셋 매트릭스

데이터셋 = (track, policy) 한 쌍에 붙인 logical name. manifest 가 logical name → (track, policy, raw_base) 매핑.
이름 규칙: `track` (sangam=상암 3D 곡선 / speedway=spd) + lead 유무 + perturbation(pertO/X) + 특수(40m=원거리, agent3=다중).

| 데이터셋 | track | lead | pert | fx | 용도(seg/a2/f1/f2) | 비고 |
|---|---|---|---|---|---|---|
| `sangam_lead_0520` | sangam | O | X | 640 | a2 / f1 / f2 | synth_v3, brakeaug. ~19.4k frame. **학습 주력** |
| `sangam_lead_0521` | sangam | O | O | 640 | a2 / f1 / f2 | synth_v3. obj 23% (lead인데 낮음 — 합성 재검토 미해결) |
| `sangam_pertO_v2` | sangam | O | O | 640 | a2 / f1 / f2 | v2 재합성. pertO |
| `sangam_pertX_v2` | sangam | O | X | 640 | a2 / f1 / f2 | v2 재합성. pertX (clean) |
| `spd_v2` (speedway) | speedway | O | — | 640 | seg / a2 / f1 / f2 | streaming 1 run, 다량 frame. seg/a2 학습 핵심 |
| `DriverB_new` | sangam | O | — | 640 | a2 / f1 / f2 | 569 run / 135k frame. **logged-only**(MPC 합성 없이 실주행 action). 처리 완료 |
| `spd_nolead_0520` | speedway | X | O | 640 | (seg/a2 보조) | f1 **학습 제외** (obj 85% labeling mismatch 미해결) |
| `spd_nolead_0521` | speedway | X | X | 640 | (seg/a2 보조) | f1 **학습 제외** (obj 1% 정상 nolead) |
| `sangam_0520_agent3` | sangam | O | X | 640 | (대기) | bank/frame_list 만 존재. prebake 미생성 |
| `sangam_0522_40m` | sangam | O(40m) | X | 640 | (대기) | 원거리 lead. synth_v2 폴더만, prebake 미생성 |
| `spd_0519_pertO` | speedway | O | O | 640 | (대기) | bank/frame_list 만 |
| `spd_0522_40m` | speedway | O(40m) | X | 640 | (대기) | 원거리 lead. bank 만 |
| ~~`sangam_v3` / `spd_v3`~~ | — | — | — | — | 폐기 | legacy corrupt synth_batch (ax 비현실, image 없음) |
| ~~MultiAgnet~~ | — | — | — | 320 | 폐기 | fx=320 (2026-05-21 폐기) |

> "학습 주력 5종" = sangam_lead_0520/0521, sangam_pertO_v2, sangam_pertX_v2, DriverB_new (+ spd_v2 는 seg/a2). f_1 은
> track 별 분리 학습 (memory `training_priority_v2`). spd_nolead 2종은 f1 에서 제외하되 seg/a2 GT 로는 활용.

---

## 3. 자산 종류별 상세

각 자산: 무엇 / 어디서 생성 / 어느 학습 단계 / 경로 / 포맷 / 규모 / 커버 데이터셋.

### 3.1 raw / logged / traj bank — 합성 중간 산출물

| 항목 | 내용 |
|---|---|
| 무엇 | raw 주행 로그를 sample 단위 binary 로 묶은 중간물. `scc_traj_bank`=trajectory 후보 bank(K augment용), `scc_logged_bank`=실주행 action sample |
| 생성 | `prepare_new_run_bank.py`, `prepare_logged_trajs_v2.py` (입력: raw, frame_list) |
| 단계 | prebake 직전 (제어 GT 갈래의 중간) |
| 경로 | `$D/scc_traj_bank_<ds>/`, `$D/scc_logged_bank_<ds>/` |
| 규모 | 데이터셋당 수백 KB~수 MB 디렉토리 (sample 파일 다수) |
| 커버 | 학습 5종 + 대기 4종(0520_agent3 / 0522_40m / spd_0519_pertO) 모두 bank 존재 |

### 3.2 synth — MPC+SAM 합성 SCC 제어 쌍

| 항목 | 내용 |
|---|---|
| 무엇 | MPC(HPIPM C++)가 trajectory+ego 로부터 합성한 control target(steer/ax) 40-step. 한 frame당 logged 1 + rand_NNNNN (K augment) |
| 생성 | `run_synth_batch_shards.sh` (env `WARMUP=20` cold-start spike 제거) → `build_pair_manifest.py` |
| 단계 | prebake 입력 (제어 GT) |
| 경로 | `$D/scc_synth_v2_<ds>/` (구), `$D/scc_synth_v3_<ds>/` (**최신, fx640**) — `<run>__t<tick>__{logged,rand_NNNNN}.scc.txt` |
| 규모 | v2: sangam_pertO/pertX/spd 각 ~24MB 디렉토리 메타. v3: sangam_lead_0520/0521, spd_nolead_0520/0521 |
| 커버 | v3 = sangam_lead_0520/0521, spd_nolead_0520/0521. v2 = sangam_pertO/pertX/spd + 40m 대기분 |

> v3 가 fx640 신규. v2 는 그 이전. legacy `_v3` *prebake* npz (corrupt, §3.7) 와 혼동 주의 — 이건 synth 디렉토리.

### 3.3 seg_masks — SAM segmentation GT (seg backbone 학습용)

| 항목 | 내용 |
|---|---|
| 무엇 | 4-class(bg/road/vehicle/wall) per-pixel mask. seg backbone pretrain 의 GT |
| 생성 | `generate_image_seg_masks.py` (GroundingDINO+SAM2) |
| 단계 | seg pretrain (글로벌·1회성) |
| 경로 | `$D/seg_masks_<ds>/{frame}.npz` (5,250,1280). 원천 `seg_masks_v2` 는 NFS shared |
| 규모 | sangam_fx640 ~1.2MB 메타, spd ~840KB, DriverB_new ~6.7MB(최근 생성) |
| 커버 | seg_masks_sangam_fx640, seg_masks_spd, seg_masks_DriverB_new (+ NFS seg_masks_v2 원천) |

> seg backbone 은 frozen 이며 재학습 시 i1/i2 cache 전부 cascade. 신중히. (ARCHITECTURE §1)

### 3.4 prebake — 학습 sample 통합 npz (f_1/f_2 직접 입력)

| 항목 | 내용 |
|---|---|
| 무엇 | 학습 1 sample 의 모든 dict(traj 40×3, ego_5d, target steer/ax, mask, d_yaw, obj_target)를 한 npz 로. mmap random access |
| 생성 | `prebake_iy_dataset.py`(synth) 또는 `prebake_iy_logged_only.py`(DriverB_new 등 logged) |
| 단계 | f_1 / f_2 학습 직접 입력 (`--prebake`, `--prebake-extra`) |
| 경로 | `$D/prebake/f1_prebake_iy_<ds>_*_prev20.npz` |
| 규모 | 데이터셋당 50~230 MB. prebake/ 전체 ~11 GB (legacy 포함) |
| 포맷 | glossary §8 (sample_keys/curr_keys/traj/target/mask/obj_target/...) |

현행 사용 파일 (학습 launcher 가 가리키는 것):

| 파일 | 데이터셋 | tag |
|---|---|---|
| `f1_prebake_iy_sangam_lead_0520_v1_brakeaug_prev20.npz` | sangam_lead_0520 | brakeaug |
| `f1_prebake_iy_sangam_lead_0521_v1_brakeaug_prev20.npz` | sangam_lead_0521 | brakeaug |
| `f1_prebake_iy_sangam_pertO_v2_prev20.npz` / `_prev20_fixed.npz` | sangam_pertO_v2 | synth |
| `f1_prebake_iy_sangam_pertX_v2_prev20.npz` / `_prev20_fixed.npz` | sangam_pertX_v2 | synth |
| `f1_prebake_iy_DriverB_new_logged_fixed_prev20.npz` / `_trajfix.npz` | DriverB_new | logged |
| `f1_prebake_iy_spd_streaming_brakeaug_prev20.npz` | spd_v2 | brakeaug |
| `f1_prebake_iy_<ds>_a2inlane_prev20.npz` | (4종) | a2-inlane GT 주입판 |

> suffix: `_brakeaug`=close-lead 제동 floor 증강, `_smoothbrake`=부드러운 감속 profile, `_a2inlane`=a_2 in-lane GT 합본,
> `_fixed`/`_trajfix`=bug fix 재생성. 단위: target_ax 는 m/s² 정상, 큰 값은 unconstrained MPC (clip 불필요, memory `ci_aug`).

### 3.5 cache — frame-level 미리 계산 (학습 속도 ↑)

frame 단위라 K augment 와 무관하게 frame당 1회 계산. 5 종류가 `$D/cache/` 에 공존:

| cache 종류 | 무엇 | 생성 | 어느 단계 | 포맷 / 규모 |
|---|---|---|---|---|
| `i1_tok_cache_<ds>_prev20.npz` | image→token, seg backbone forward. **proj 전 raw**(C_b=256) | `compute_i1_tok_cache.py` (seg=fx640_v3) | a_2 / f_1 / f_2 입력 | (N,40,256) fp16. ~15MB~2GB |
| `i1_tok_cache_<ds>_hr.npz` | **stride16 = 80-token** 고해상도판 (40 대신 80 strip) | 동일(고해상도 옵션) | 해상도 ablation | (N,80,256) |
| `cache/b0/i1_tok_cache_b0_<ds>_prev20.npz` | **EfficientNetV2-b0** 경량 backbone(192ch) ablation | `build_i1_tok_cache_b0.py` | a_2-on-b0 ablation | (N,40,192). b0 폴더 ~1.1GB. **교체 보류**(vehicle IoU −0.055) |
| `a2_out_cache_<ds>_prev20.npz` / `_prev20_fixed.npz` | 학습된 a_2 의 inference 결과 (per-bin 7ch). f1/f2 학습 시 inject | `build_a2_cache.py` | f_1 / f_2 입력 | (N,40,7) fp16 ~5MB. `_fixed`=2026-05-24 proj 버그 fix |
| `i2_mem_cache_<ds>_<av>_prev20.npz` | a_2 fusion 의 i2_mem(40,128) + a2_logits(40,7). f_2 의 KV | `build_i2_cache.py` (i1+prebake+a2, ego_mot=0) | f_2 입력 | (N,40,128). a2 버전별 분기 |
| `inlane_gt_<ds>.npz` | raw object → in-lane(정면 bin16-25) lead dist/closing GT | `build_inlane_gt_cache.py` | f_2 TTC / pedal label | keys + inlane_dist + inlane_closing |
| `pedal_label_<ds>_{ttc,ttc2,distbrake}.npz` | brake floor → pedal_map 변환된 pedal target(40) | `gen_ttc_pedal_labels.py` (`BRAKE_MODE`) | f_2 pedal GT | keys + pedal(40) |
| `pred_vrel_<ds>.npz` | a_2 의 vL_rel 예측 (진단/gate용) | (a2 inference) | 분석/gate | per-frame vrel |
| `close_lead_meta_v4.npz` | close-lead(dist<30m) sample 표식 + min_dist 분포 | `build_close_lead_meta.py` | f_1/f_2 loss weighting (×3) | v1~v4 누적, v4 최신 |
| `val_sample_DriverB_5k.npz` | DriverB 5k val 고정 sample | — | 평가 | val split |

i2_mem a_2 버전 suffix (`<av>`): `_v6_`, `_v7_`, `_v7b_`, `_v7c_`. **현행 배포·학습 = `v7b`** (a2_v7b_d128_alpha075,
focal α=0.75). v6 은 일부 launcher 가 여전히 참조. closed-loop live i2_mem 은 이 cache 빌더와 **동일 설정**
(seg=fx640_v3, ego_mot=0) 이어야 한다 (memory `closed_loop_i2mem_mismatch` — 어긋나면 SCC/제동 붕괴).

pedal_label 모드: `ttc`=closing 기반 ACC, `distbrake`=거리-onset(closing 비의존, vL_rel 약점 회피 — 현행 실험), `ttc2`=ttc 변형.
세부 파라미터 [DATA_PIPELINE.md §4](DATA_PIPELINE.md).

i1 cache 백업/legacy (cleanup 후보): `_v2sfull`(v2s_full backbone mIoU 0.731), `_combined_v1`(0.892),
`i1_tok_cache_sangam_v3.npz`(1.1GB legacy), `_fx640v3` 접미(현재 standard 와 중복 가능), `spd_nolead_*`(학습 미사용).

### 3.6 aux_targets — a_2 학습 GT (per-frame 디렉토리)

| 항목 | 내용 |
|---|---|
| 무엇 | a_2 의 GT. frame당 (40,7) = `[obj, vL_abs, vY_abs, vL_rel, vY_rel, prox, dist]`. obj=bin별 lead 유무, dist=거리, vL_rel=상대 longitudinal 속도 |
| 생성 | `prepare_a2_aux_targets.py` (raw OBJECT_INFO + camera projection) |
| 단계 | a_2 학습 |
| 경로 | `$D/aux_targets_<ds>/` (frame당 ~500B npz 다수) + `config.json` |
| 규모 | 디렉토리당 ~2~12 MB 메타 (frame 수 만큼 작은 npz). sangam_lead_0520 = 19,400 frame |
| 커버 | DriverB_new / sangam_fx640 / sangam_lead_0520·0521 / spd_fx640 / spd_nolead_0520·0521 |

`_bin80` 변형: **stride16 = 80-bin** 고해상도 GT (40-bin 대신 80, `_hr` i1 cache 와 짝). 신규
(sangam_lead_0520/0521, spd_nolead_0520) — 해상도 ablation 용. `a2_targets/`(구), `aux_targets_spd_smoke_fx640`(smoke).

### 3.7 폐기 / cleanup 후보

| 자산 | 문제 |
|---|---|
| `f1_prebake_iy_{spd,sangam_pertO,sangam}_v3.npz`, `f1_prebake_iy_spd.npz` | legacy synth_batch corrupt — ax mean -31(97% strong brake), range [-172,+284] 비현실, image key 빈 string |
| `i1_tok_cache_{sangam_v3, spd_nolead_*}.npz` | 구 backbone(v2s_full/legacy), 학습 미사용 |
| `i1_tok_cache_*_v2sfull / _combined_v1` | 이전 backbone 백업 |
| MultiAgnet / fx=320 일체 | 폐기 (2026-05-21) |
| smoke / `_v1`(non-prev20) prebake | 초기 테스트물 |

---

## 4. 파이프라인 흐름 (입출력 경로 요약)

상세 DAG·스크립트는 [DATA_PIPELINE.md §2-3](DATA_PIPELINE.md). 여기는 자산 입출력만:

| # | 단계 | 입력 | 출력 | 글로벌? |
|---|---|---|---|---|
| 1 | seg mask | raw 이미지 | `seg_masks_<ds>/` | — |
| 2 | seg pretrain | seg_masks + 이미지 | seg backbone ckpt (fx640_v3) | **글로벌 1회** |
| 3 | i1 token cache | manifest + raw 이미지 + backbone | `cache/i1_tok_cache_<ds>_prev20.npz` | per-ds |
| 4 | aux_targets | raw OBJECT | `aux_targets_<ds>/` | per-ds |
| 5 | a_2 학습 | i1 cache + aux_targets | a2 ckpt (v7b) | **글로벌** |
| 6 | a2 cache / i2_mem | i1 cache + prebake + a2 ckpt | `a2_out_cache`, `i2_mem_cache_<av>` | per-ds |
| 7 | synth/logged bank | raw EGO/object | `scc_*_bank_<ds>/` | per-ds |
| 8 | prebake | bank/manifest + control GT | `prebake/f1_prebake_iy_<ds>_prev20.npz` | per-ds |
| 9 | inlane_gt / pedal_label | raw object + prebake | `cache/inlane_gt`, `cache/pedal_label_<ds>_{ttc,distbrake}` | per-ds |
| 10 | f_1 / f_2 학습 | prebake + i1/i2/a2 cache + pedal_label | iy/prof635 ckpt | — |

신규 데이터 추가 표준 순서: glossary §14 (raw → seg → i1 → aux → a2 cache → bank → prebake → close_lead_meta → 학습).

---

## 5. 학습된 ckpt (repo `data/ckpts/`, git 추적)

배포물이라 git force-add 로 추적. registry = `data/ckpts/CURRENT_BEST.json`.

| 모듈 | 현행 best ckpt | metric | 상태 |
|---|---|---|---|
| seg backbone | `seg_pretrain_fx640_v3_backbone.pt` | mIoU 0.921 | frozen, 배포 |
| a_2 | `a2_v7b_d128_alpha075_best_loss.pt` | obj prec~66%, vL_rel corr~0.40 (약함) | frozen, 배포 |
| f_1 (main) | `main_prof654.pth` | steer corr~0.88 | frozen, 배포 |
| f_2 (side) | `prof635_side_distbrake_v1.pt` | val_pedal~0.52 (brake_only) | 배포 |
| iy(통합) | `iy_v25k_B4_gttc_accgate_best_loss.pt` | val 0.5808(외부보고, [hypothesis]) | 예선용 |

기타 ckpt 계보 (cleanup 안 함, 비교/ablation 보존):

| 그룹 | 파일 예 | 비고 |
|---|---|---|
| seg backbone 백업 | `seg_pretrain_combined_v1_*`(0.892), `_v2s_full_*`(0.731) | 이전 버전 |
| a_2 계보 | `a2_v1/v4/v5_d128/v6_focal/v7~v7c/v8_gaussNLL_lds/v9_lds`, `a2_gate_hardneg`, `a2_mf5f` | v7b 가 best. v8/v9 = LDS/GaussNLL 실험 |
| f_1/main | `main_prof635_raw/654/655/656` | 654 배포 |
| f_2/side | `prof635_side_{distbrake_v1, predvrel_signed/ttc/ttc2, v5}` | distbrake_v1 배포 |
| iy 계보 | `iy_v8 ~ v25k_B4_gttc_accgate` (v15~v25 다수) | v25 = prof635 흡수 redesign. B숫자=side cross-attn layer 수(B1=1L…B4=4L) |
| gate | `learned_temporal_gate_mlp_{beat,distill}.pt` | temporal gate 실험 |

iy 계보 history 와 metric: [MODULE_EXPERIMENT_HISTORY.md §4](MODULE_EXPERIMENT_HISTORY.md).
f_1/f_2 명명: f_1=main waypoint, f_2=Side SCC(cross-attn+a_2+TTC) (memory `f1_f2_naming`).

---

## 6. control / 보조 (repo `data/control/`)

raceline csv + pedal map + pit polygon. closed-loop inference 의 원천(재생성 불가, 내제화됨).

| 파일 | 무엇 |
|---|---|
| `sangam_high/mild_*.csv`, `speedway_high/mild_*.csv` | raceline (semicolon-delim: s;x;y;yaw_rad;curv;speed;pit). s 비단조 → row index sampling (memory `raceline_csv_format`) |
| `pedal_map_a1_f4.csv` | 속도×가속 → accel/brake pedal 변환 LUT |
| `picked_idx_*.json`, `pit_polygon_*.json` | raceline 구간 선택 / pit 영역 |

repo `data/` 기타: `multiagnet_compact.npz`(+meta, fx320 legacy 참고), `v0237_*`(ci_aug 샘플), `scc_synth_*_v1`/`seg_masks_v1`(초기물).

---

## 7. ci_aug — controller_imitator (image-less 보조 sample)

f_1(main_no_image)에 traj+ego→control sample 추가 (waypoint following only, ACC 없음, image token=0 padded).

| 항목 | 내용 |
|---|---|
| 위치 | `~/AILabDatasetLocal/02_Custom_Dataset/35_e2e_morai/controller_imitator/v*` |
| 사용 | v0237(720k) + v0301(4.3k) + v0302(4.7k) = ~730k sample. v0229~0231 미사용 |
| 포맷 | pkl→npz 변환 (mmap), `controller_imitator_aug_dataset.py` ConcatDataset |
| 단위 | target_acceleration m/s² 정상 (unconstrained MPC, clip 불필요) |

---

## 8. 알려진 mismatch / TODO

미해결(추정/확인 구분):

1. [unresolved] `spd_nolead_0520` obj 85% 인데 "nolead" — labeling 모순. f1 학습 제외 중.
2. [unresolved] `sangam_lead_0521` lead 인데 obj 23% (의도 미달) — 합성 재검토 필요.
3. [hypothesis] `iy_v25k_B4_gttc_accgate` val 0.5808 은 외부보고치 — train.log 가 ep01 에서 멈춰 자체 best 미확인 ([MODULE_EXPERIMENT_HISTORY §225](MODULE_EXPERIMENT_HISTORY.md)).
4. [confirmed] a_2 vL_rel corr~0.40 약함 → ACC 한계 근본 원인. distbrake(거리 기반)로 회피 중.
5. cleanup 후보: legacy v3 prebake 4개(~7GB), `i1_tok_cache_{sangam_v3, spd_nolead_*, *_v2sfull, *_combined_v1}`.

이 문서에서 확인 못 한 것 (접근/시간 제약): NFS `seg_masks_v2`·`multiagnet_compact_meta` 내부 규모(NFS find 금지로 1-level만),
ci_aug 디렉토리 실측 크기(별도 마운트), 대기 4종(0520_agent3/0522_40m/spd_0519_pertO)의 정확한 frame 수.

---

## 부록. naming suffix 빠른 참조

| suffix | 의미 |
|---|---|
| `_prev20` / `_prev5` | prev frame 200ms / 50ms(legacy) |
| `_v1~v3`(synth/prebake), `_v6/v7/v7b/v7c`(i2_mem=a2버전) | 데이터/모델 버전 |
| `_hr`, `_bin80` | stride16 = 80-token/bin 고해상도 |
| `_fx640v3` | seg backbone fx640_v3 명시 |
| `_brakeaug` / `_smoothbrake` | 제동 증강 / 부드러운 감속 |
| `_a2inlane` | a_2 in-lane GT 합본 prebake |
| `_logged` | MPC 합성 없이 raw driver action |
| `_fixed` / `_trajfix` / `_clipped` | bug fix 재생성 / traj fix / target clip |
| `{ttc,ttc2,distbrake}` | pedal label brake 모드 |
| `_b0` | EfficientNetV2-b0 경량 backbone(192ch) ablation |
| `pertO` / `pertX` | perturbation 有 / 無 |

전체 사전: [DATA_PIPELINE_GLOSSARY.md §12](DATA_PIPELINE_GLOSSARY.md).

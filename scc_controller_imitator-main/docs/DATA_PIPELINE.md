# 데이터 파이프라인 (raw → 학습 캐시)

raw 주행 로그를 학습 가능한 캐시로 변환하는 단계와 스크립트, 캐시 명명, 경로 portability.

---

## 1. raw 데이터 구조

```
{raw_root}/{track}/{policy}/{run}/
    CAMERA_1 .. CAMERA_5/   {tick}_Intensity.jpeg   (front=CAMERA_1, 1280×720, ROI crop 235:485 → 250×1280)
    EGO_INFO/               {tick}.txt   (자차 pose/속도/가속/조향)
    OBJECT_INFO/            {tick}.txt   (주변 객체 — 상대 위치/속도)
```
- raw_root 후보 (자동 탐지, SSD 우선): `AILabSSD/02_Data/57_A1_E2E/00_raw` (빠름, nfs4.2), `AILabDataset/02_Custom_Dataset/57_A1_E2E/00_raw`.
- **track**: `sangam_track`(상암), `speedway_track`, `sangam_qlf`.
- **policy** 예: `0520_agent1_pertX_pitX`, `0521_agent1_pertO_pitX`, `rule_v2` … (날짜_agent_perturbation_pit 조합).
- **"데이터셋" = (track, policy)**. logical name(예: `sangam_lead_0520`)은 `manifest_i1cache_{name}.json` 이 (track,policy,raw_base) 로 매핑.

---

## 2. 파이프라인 DAG

```
                      (SAM: GroundingDINO+SAM2)
raw 이미지 ───────────────────────────────► seg_masks ──► [seg pretrain] ──► seg backbone (fx640_v3, frozen)
   │                                                                              │
   │ compute_i1_tok_cache (backbone inference, seg mask 불필요)                    │ 사용
   ├──────────────────────────────────────► i1_tok_cache (N_frame, 40, 256)  ◄────┘
   │ prebake_iy_dataset (raw EGO/object + control GT)
   ├──────────────────────────────────────► prebake (traj 40×3, ego_5d, target steer/ax, mask, d_yaw)
   │                                              │
   │ build_i2_cache (i1 + prebake + a_2, ego_mot=0)│
   ├──────────────────────────────────────► i2_mem_cache (N_frame, 40, 128) + a2_logits(40,7)
   │ build_inlane_gt_cache (raw object → in-lane 판정)
   ├──────────────────────────────────────► inlane_gt (keys, inlane_dist, inlane_closing)
   └ gen_ttc_pedal_labels (prebake target_ax + inlane_gt → brake floor → pedal_map; BRAKE_MODE=ttc|distbrake)
                                          ► pedal_label_*_{ttc,distbrake} (keys, pedal 40)
```

의존성 핵심:
- **seg mask / seg pretrain 은 글로벌·1회성** (backbone 학습 GT). i1/i2/prebake 는 학습된 backbone 만 쓰지 seg mask 는 안 씀.
- **seg backbone 재학습 = 모든 i1/i2 캐시 무효화 → 전부 재생성 cascade** (a_2/f_2 까지). 신중히.
- i2_mem 은 **i1_tok_cache + a_2(a2_v7b) + ego_mot=zeros** 로 생성. 이 조합이 학습 분포를 정의 → closed-loop live 계산도 동일해야 함 ([STATE_AND_LESSONS.md](STATE_AND_LESSONS.md) i2_mem 불일치 참조).

---

## 3. 단계별 스크립트 / 입출력

| 단계 | 스크립트 | 입력 | 출력 | 비고 |
|---|---|---|---|---|
| seg mask | `generate_image_seg_masks.py` | raw 이미지 | `seg_masks_v2/{frame}.npz` (5,250,1280) | GroundingDINO+SAM2. RAW_IMG_BASE/OUT_DIR 현재 하드코딩(구 MultiAgnet) — 신규 raw 용 파라미터화 필요 |
| seg pretrain | (train script — TRAINING.md) | seg_masks + 이미지 | seg backbone ckpt | 글로벌. EfficientNetV2-S |
| i1 토큰 | `compute_i1_tok_cache.py` | manifest + raw 이미지 + seg backbone | `i1_tok_cache_{name}_prev20.npz` (tokens 40×256) | adaptive_avg_pool2d(1,40). prev offset(5,20) 캐시 |
| prebake | `prebake_iy_dataset.py` | manifest (sample 목록 + control GT) | `f1_prebake_iy_{name}_*_prev20.npz` | traj/ego/target/mask/d_yaw. control GT = logged 또는 MPC synth |
| i2_mem | `build_i2_cache.py` | i1_cache + prebake + a2 ckpt | `i2_mem_cache_{name}_v7b_prev20.npz` | ego_mot=0. a2.encoder.proj→fusion→fc_proj→head |
| inlane GT | `build_inlane_gt_cache.py` | raw object + prebake | `inlane_gt_{name}.npz` | in-lane 객체 dist/closing(상대속도) GT |
| pedal label | `gen_ttc_pedal_labels.py` | prebake target_ax + inlane_gt | `pedal_label_{name}_{ttc,distbrake}.npz` | brake floor → pedal_map. `BRAKE_MODE`: ttc(closing ACC) / distbrake(거리-onset). raw 라벨은 lead 가속 버그 |

manifest(`manifest_i1cache_{name}.json`): `{track, policy, raw_base, samples[...]}` — 파이프라인의 진입점. 신규 데이터 등록 = manifest 생성(control GT가 logged 인지 synth 인지 결정).

---

## 4. pedal label (brake floor → pedal_map)

`gen_ttc_pedal_labels.py`: 앞차 시 brake floor 를 raw target_ax 에 주입(`aug_ax = min(raw_ax, −sd)`, 첫 KSTEPS=10 step,
in-lane lead 일 때만) 후 pedal_map. **`BRAKE_MODE` 환경변수로 두 모드** (출력: `pedal_label_{name}_{mode}.npz`).

**`ttc` (closing 기반 ACC)** — `sd = max(a_kin, a_gap, a_emerg)`. lead 판정 = dist 유한 & closing>0.1.
- params: `D0=5, HEADWAY=1, KGAP=0.2, KCLO=0.6, TTC_EMERG=2.0, DECAY=5.0, MAXB=20, MARGIN=2.0, KSTEPS=10`.

**`distbrake` (거리-onset 곡선, closing 비의존, 현행 실험)** — a_2 vL_rel 약점 회피, "가까우면 제동":
- `onset(v)=DB_D0+DB_HEADWAY·v`, `u=clip((onset−dist)/(onset−DB_DFULL),0,1)`, `sd=DB_MINB+(MAXB−DB_MINB)·u^DB_P`.
  `dist≤DB_DFULL→MAXB`(full), `dist≥onset→0`. lead = dist 유한 & `dist<onset` (closing 무관).
- params: `DB_D0=10, DB_HEADWAY=1.0, DB_DFULL=7, DB_MINB=1, MAXB=20, DB_P=2`.
- 검증 plot: `viz_distbrake_verify.py` (x=거리 y=속도 color=감속). 생성: `BRAKE_MODE=distbrake python scripts/gen_ttc_pedal_labels.py`.

공통: in-lane 판정 = `inlane_gt`. pedal_map = `data/control/pedal_map_a1_f4.csv` (속도×가속 → accel/brake pedal).
`pedal_label.py` `label()` 은 벡터 bilinear (scalar 678k 호출 시 numpy dispatch segfault 회피, [REPORT §9.4]).
주의: raw(미증강) 라벨은 lead 접근 시 +가속 버그 → 반드시 `_ttc`/`_distbrake` 사용. `_ttc` 와 `_distbrake` 는 공존(별 파일).

---

## 5. 경로 portability (범용성)

- **캐시 출력 디렉토리** `D`: `SCC_DATA_DIR`(env) > `AILabDatasetLocal/scc_controller_imitator/data`(이 머신, 존재 시) > `{repo}/data/local`(타 머신, **gitignored**). 8개 스크립트가 이 규칙 사용.
- **raw_root**: `SCC_RAW_ROOT`(env) > AILabSSD > AILabDataset (자동 탐지).
- 타 컴퓨터엔 AILabDatasetLocal 없음 → 자동으로 `repo/data/local` 사용, 캐시가 repo 안에 생성되고 git 추적 안 됨(`data/*` ignore, `!data/ckpts` 예외).
- 배포 weight 는 `data/ckpts/` 에 force-add 로 git 추적 (clone 만 하면 포함).

---

## 6. 현재 캐시 현황 (이 머신)

`AILabDatasetLocal/scc_controller_imitator/data/` 에: prebake ~37, i2_mem_cache ~29, inlane_gt 6.
학습에 쓰는 5 datasets: sangam_lead_0520/0521, sangam_pertO_v2, sangam_pertX_v2, DriverB_new.
스캔 도구: `python scripts/data_pipeline.py scan` (raw 인벤토리 + 데이터셋별 단계 처리상태 표).

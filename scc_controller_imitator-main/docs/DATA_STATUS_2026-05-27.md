# SCC Controller Imitator — 데이터셋 전수 재실측 (2026-05-27)

목적: 기존 [DATA_INVENTORY.md](DATA_INVENTORY.md) 는 "어디에 뭐가 있나" 만 나열하고 **정합(key overlap) · 라벨품질 · 실제 학습 사용** 을 검증하지 않아,
`spd_v2` 가 `i1_cache`(run 260520) 와 `inlane_gt`(run 260519) 서로 다른 run 에 같은 이름으로 붙은 mismatch 를 놓쳤다.
본 문서는 추측·기존문서 신뢰 없이 **npz key/config/shape 를 직접 읽어** confirmed 로만 기록한다.

- 실측 도구: npz 의 `keys`/`sample_keys`/`curr_keys` 만 로드 (큰 token/target array 는 zip header 로 shape 만, 또는 subsample). GPU 미사용, 읽기 전용.
- `$D` = `/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data`
- **[confirmed]** = 직접 읽어 확인. **[추정]** = 파일명/구조 기반 추론. **[미확인]** = 접근/시간 제약으로 못 봄.
- run prefix: key 앞 6자리 (`260519`=05/19, `260520`=05/20, `260521`=05/21, `260513`=DriverB old). aux 파일명에서 `<track>__<policy>__<run>` 전체 descriptor 확인.

---

## §1. 데이터셋 매트릭스 (logical name × 산출물, run + n) — [confirmed]

각 셀 = `run / n`. **run 이 한 logical name 안에서 어긋나면 ⚠ 표시.** n=frame 수(중복 제거 기준). prebake 의 sample n 은 K-augment 포함이므로 별도 표기.

| logical name | raw (NFS run) | aux_targets (run/n) | i1_tok_cache_prev20 (run/n) | inlane_gt (run/n) | i2_mem v7b (run/n) | prebake curr (run/uniq) | pedal_label (run) |
|---|---|---|---|---|---|---|---|
| sangam_lead_0520 | sangam 0520_agent1_pertX | 0520 / 19,400 | 0520 / 19,399 | 0520 / 19,399 | 0520 / 19,399 | 0520 / 728 | 0520 |
| sangam_lead_0521 | sangam 0521_agent1_pertO | 0521 / 23,698 | 0521 / 23,697 | 0521 / 23,697 | 0521 / 23,697 | 0521 / 985 | 0521 |
| sangam_pertO_v2 | sangam 0519_pertO | (sangam_fx640) 0519 / 63,292 | 0519 / 30,593 | 0519 / 30,593 | 0519 / 30,593 | 0519 / 3,405 | 0519 |
| sangam_pertX_v2 | sangam 0519_pertX | (sangam_fx640) 0519 / 63,292 | 0519 / 32,699 | 0519 / 32,699 | 0519 / 32,699 | 0519 / 3,344 | 0519 |
| DriverB_new | (old NFS) 260513~260519 | 0513/0519 / 135,113 | 0513/0519 / 135,113 | 0513/0519 / 135,113 | 0513/0519 / 135,113 | 0513/0519 / 132,838 | 0513/0519 |
| **spd_v2** ⚠⚠ | spd 0519_pertX **and** 0520_pertO | 0519 `spd_fx640` / 37,041 ⚠ **그리고** 0520 `spd_nolead_0520` / 34,499 ⚠ | **0520 / 34,498** | **0519 / 37,040** ⚠ | **0520 / 34,498** | **0519 / 3,398** (streaming) ⚠ | **0519** |

### spd_v2 = 서로 다른 두 run 이 한 이름에 — [confirmed, 사용자 지적 확정]

`spd_v2` 안에 독립된 두 데이터 세계가 같은 이름으로 공존한다:

| 세계 | run | raw | 붙은 산출물 | n | 내부 정합 |
|---|---|---|---|---|---|
| **A (260519, spd pertX)** | 260519 | speedway 0519_agent1_pertX_pitX | `inlane_gt_spd_v2`, `aux_targets_spd_fx640`, `i1_tok_cache_spd_v2(_streaming)_prev20`(3,407), `pedal_label_spd_v2_*`, `f1_prebake_iy_spd_streaming_brakeaug_prev20` | 37,040 (frame), prebake uniq 3,398 | 100% 일치 |
| **B (260520, spd pertO/nolead)** | 260520 | speedway 0520_agent1_pertO_pitX | `i1_tok_cache_spd_v2_prev20`(34,498), `a2_out_cache_spd_v2`, `i2_mem_cache_spd_v2_v7b`, `aux_targets_spd_nolead_0520` | 34,498 | 100% 일치 |

- **A ↔ B 의 i1 ↔ inlane_gt key 교집합 = 0.0%** (inter=0, na=34498, nb=37040). [confirmed]
- 즉 "spd_v2" 라는 이름 하나가 두 raw run 을 가리킨다. i1/i2/a2 계열은 **260520**, inlane_gt/pedal/prebake 계열은 **260519**.

> 비고: aux_targets_spd_fx640 파일명 = `speedway_track__0519_agent1_pertX_pitX__260519_seg0...`, aux_targets_spd_nolead_0520 = `speedway_track__0520_agent1_pertO_pitX__260520_100...` 로 raw descriptor 까지 확정.

---

## §2. 정합 검증 (key 집합 교집합 %) — [confirmed]

ds 별 `i1 ↔ inlane_gt ↔ aux ↔ prebake` 의 set 교집합. min(na,nb) 기준.

| ds | i1↔inlane | i1↔a2_out | i1↔aux | inlane↔aux | iy join valid (prebake↔i1) | 판정 |
|---|---|---|---|---|---|---|
| sangam_lead_0520 | 100% | 100% | 100% | 100% | 98.9% | 정합 OK |
| sangam_lead_0521 | 100% | 100% | 100% | 100% | 97.7% | 정합 OK |
| sangam_pertO_v2 | 100% | 100% | 100% (i1⊂aux63k) | 100% | frame 100% (sample 10.8%=K-aug) | 정합 OK |
| sangam_pertX_v2 | 100% | 100% | 100% (i1⊂aux63k) | 100% | frame 100% (sample 10.1%=K-aug) | 정합 OK |
| DriverB_new | 100% | 100% | 100% | 100% | 98.3% (INVALID 1.68% drop) | 정합 OK |
| **spd_v2** | **0.0%** ⚠ | 100% | 100%(B) / 0%(A) | 0.0%(B vs A) | **0% (valid 0/339,225)** ⚠ | **f_1/f_2/iy mismatch — 사용불가** |

- spd_v2: i1_cache(260520) ↔ inlane_gt(260519) **0% 교집합**. f_2 의 inlane/pedal GT 가 전부 엉뚱한 frame 에 매칭됨.
- **iy 학습 dataset 은 key-join** (`src/datamodule/scc_synth_iy_dataset.py:78-80`: curr_key+prev_key 둘 다 i1 cache 에 있는 sample 만 valid). spd prebake(260519) ↔ i1_cache_spd_v2_prev20(260520) join → **valid = 0 / 339,225** (직접 재현 확인). 즉 spd 데이터는 iy 학습 시 **에러 없이 전량 silent drop**.
- sangam_pertO/X 의 iy "valid 10%" 는 mismatch 아님: prebake 는 K=100 augment(sample 3M), i1 은 frame당 1개(30k) 라 frame 기준 100% 정합 (set 교집합 100% 확인).

---

## §3. 라벨 품질 (obj/lead presence, closing 분포) — [confirmed, aux 2,500 sample / inlane 전수]

### 3.1 aux_targets obj presence (a_2 GT, target (40,7), ch0=obj ch6=dist)

| aux_targets ds | obj_any % | obj_inlane(16-25) % | dist median(m) | 이상 색출 |
|---|---|---|---|---|
| sangam_lead_0520 | 100.0 | 99.4 | 14.2 | 정상 (lead-rich) |
| sangam_lead_0521 | 32.5 | 28.4 | 32.0 | ⚠ **lead 인데 obj 32%** (의도 미달, 기존 "23%" 와 동급) |
| sangam_fx640 | **0.0** | **0.0** | — | ⚠⚠ **객체 전무**. pertO_v2/pertX_v2 의 a_2/inlane GT 소스인데 obj=0 |
| DriverB_new | 61.4 | 46.1 | 27.4 | 정상 (혼합 lead) |
| spd_fx640 (=spd_v2 세계 A) | 100.0 | 98.6 | 14.7 | lead-rich (단 짝 i1 mismatch) |
| spd_nolead_0520 (=spd_v2 세계 B) | **67.0** | **56.0** | 25.5 | ⚠ **"nolead" 인데 obj 67%** (a_2 학습에 실제 투입됨) |
| spd_nolead_0521 | 0.7 | 0.0 | — | 정상 nolead |

### 3.2 inlane_gt lead/closing (f_2 gate/pedal GT)

| inlane_gt ds | lead_pct | dist med | closing med | closing std | 이상 색출 |
|---|---|---|---|---|---|
| sangam_lead_0520 | 99.3 | 14.2 | 0.06 | 1.69 | 정상 |
| sangam_lead_0521 | 28.0 | 28.9 | 6.54 | 5.81 | ⚠ lead 28% (aux 와 일치) |
| sangam_pertO_v2 | **0.0** | — | — | — | ⚠⚠ **in-lane lead 전무** (aux_sangam_fx640 obj=0 근원) |
| sangam_pertX_v2 | **0.0** | — | — | — | ⚠⚠ **in-lane lead 전무** (동일 근원) |
| DriverB_new | 45.1 | 29.5 | 0.23 | 8.11 | 정상 |
| spd_v2 (260519) | 98.9 | 14.7 | 0.27 | 3.10 | lead-rich 하나 짝 i1=260520 mismatch → 무용 |

핵심:
- **sangam_pertO_v2 / pertX_v2 의 inlane_gt lead_pct = 0%** — [confirmed 신규 발견]. 근원은 aux_targets_sangam_fx640 obj 전무(§3.1). 이 두 ds 는 raw 가 lead(pertO=전방차+perturbation) 인데 OBJECT projection 산출물에 객체가 0 → f_2 의 SCC 제동/gate supervision 기여 0. (closing degenerate 이전 단계에서 lead 자체 부재.)
- closing degenerate(정지객체 상수) 전례는 본 6종 inlane_gt 에서는 재현 안 됨 (closing_std 모두 >1.6). sangam_0520_agent3 는 inlane_gt 미생성이라 [미확인].

### 3.3 prebake target ax 품질 (corrupt 검증)

| prebake | ax range | mean | 판정 |
|---|---|---|---|
| sangam_lead_0520_v1_brakeaug_prev20 | [-25, +25] | -3.98 | 정상 (clip) |
| spd_streaming_brakeaug_prev20 | [-25, +25] | -4.32 | 정상 (단 spd join 0%) |
| **DriverB_new_logged_fixed_prev20_trajfix** | **[-229.5, +60.9]** | 0.12 | ⚠ logged action unclip — 극단값 (-229 m/s²) 잔존. f_2 DATASETS 에 실제 사용 |
| sangam_v3 (legacy) | [0, 0] | 0 | 폐기 (ax 전부 0, image 없음) |
| spd_v3 / spd (legacy) | [-112, +29] | -5.26 | 폐기 (비현실) |
| sangam_pertO_v3 (legacy) | [-169, +266] | -10.15 | 폐기 (비현실) |

---

## §4. 실제 학습 사용 (launcher 대조) — [confirmed]

launcher 가 실제 가리키는 파일을 읽어 대조. "용도" 표기(DATA_INVENTORY §2) 와 다른 셀을 표시.

### 4.1 a_2 학습/eval — `scripts/train_a2_n20.sh`, `eval_a2_n20.sh`
3-set 고정: `sangam_lead_0520`(aux 0520) + `sangam_lead_0521`(aux 0521) + **`spd_v2` i1(260520) + aux_spd_nolead_0520(260520)**.
- spd_v2 가 a_2 에 들어가긴 하나 **i1 과 aux 가 둘 다 260520 (세계 B)** 이라 a_2 관점 정합 OK. 단 GT 가 "nolead" 인데 obj 67% (§3.1) — labeling 모순 데이터를 a_2 가 학습 중.
- DriverB_new 는 a_2 학습에 미사용, **held-out eval 전용**(`eval_a2_expand_driverb.sh`, 40k subset). [confirmed]

### 4.2 f_2(side) 학습 — `scripts/train_prof635_side.py` DATASETS / `train_scc_profiles_g2.sh`
하드코딩 DATASETS = **sangam_lead_0520, sangam_lead_0521, sangam_pertO_v2, sangam_pertX_v2, DriverB_new (5종)**. **spd 없음.**
- ⇒ spd_v2 mismatch 는 f_2 단독 학습에 영향 없음 (애초에 미투입).
- ⚠ pertO_v2/pertX_v2 가 DATASETS 에 있으나 §3.2 에서 inlane_gt lead 0% → gate/pedal GT 가 전부 no-lead. f_2 에서 이 2종은 SCC 제동 신호 0 기여 (steer/waypoint 만 기여).
- pedal_suffix = p1~p5 (profile) 또는 ttc/distbrake. p-series 는 5종 + spd_v2 모두 존재.

### 4.3 f_1/iy 통합 학습 — `train_iy.py` + launch_v25k_B_gttc / v26_Bminacc 등
extra 5-slot = sangam_lead_0521, pertO_v2, pertX_v2, DriverB_new, **spd** (main slot=sangam_lead_0520).
- ⚠⚠ spd slot: prebake=`spd_streaming_brakeaug_prev20`(260519) + i1=`i1_tok_cache_spd_v2_prev20`(260520) + i2=`spd_v2_v7b`(260520) + inlane=`inlane_gt_spd_v2`(260519).
  prebake(260519) ↔ i1(260520) key-join = **valid 0** ⇒ **spd 는 iy 학습에서 0 sample 기여 (silent drop)**. [confirmed]
- 따라서 현재 iy 모델은 사실상 sangam 4종 + DriverB + ci_aug 로만 학습됨. "spd 가 seg/a2/f1/f2 핵심" 이라는 기존 문서 표기와 배치.

---

## §5. 문제 목록 + ds별 판정 — [confirmed 기반]

### 가장 심각한 문제 3

1. **spd_v2 = 두 run(260519/260520) 혼선** ⚠⚠ — i1/i2/a2 는 260520(spd_nolead_0520), inlane_gt/pedal/prebake 는 260519(spd pertX). i1↔inlane 교집합 0%. iy launcher 가 이 둘을 한 slot 에 묶어 → **spd 가 iy 학습에서 valid 0 으로 전량 silent drop**. 에러도 경고도 없이 "학습에 들어간 줄 알았던" 데이터가 0 기여.
2. **sangam_pertO_v2 / pertX_v2 의 객체 라벨 전무** ⚠⚠ — aux_sangam_fx640 obj 0% → inlane_gt lead 0%. f_2 DATASETS 에 포함돼 있으나 SCC 제동/gate GT 0 기여. lead(pertO) 데이터인데 OBJECT projection 산출물이 비어 있음 (raw→aux 단계 의심).
3. **spd_nolead_0520 의 nolead↔obj 모순** ⚠ — "nolead" 명명인데 obj 67%/inlane 56%. 이 데이터가 spd_v2 i1 과 짝지어 **a_2 학습(train_a2_n20.sh)에 실제 투입** 중. a_2 의 obj head 에 모순 신호 주입 가능성.

### 기타 문제
- DriverB_new prebake: INVALID/empty key 1.68% (2,276/135,113) — iy join 에서 drop. trajfix prebake ax [-229, +61] unclip 극단값.
- legacy v3 prebake 4개(sangam_v3 ax=0, spd_v3/sangam_pertO_v3 ax 비현실) — corrupt, 폐기 후보 (DATA_INVENTORY §3.7 과 일치 확인).
- `cache/b0/`, `a2_targets/` 폴더 **비어 있음** — DATA_INVENTORY 가 b0 cache(~1.1GB) 존재로 적었으나 현재 empty. b0 ablation 산출물 삭제됨 (eval_a2_expand_driverb.sh 가 참조하는 b0_40k cache 도 [미확인=없을 가능성]).

### ds별 즉시 판정

| ds | a_2 | f_1/iy | f_2 | 판정 |
|---|---|---|---|---|
| sangam_lead_0520 | OK | OK(98.9%) | OK(lead 99%) | **즉시 학습 가능** (주력) |
| sangam_lead_0521 | obj32% | OK(97.7%) | lead 28% 약함 | **사용 가능 (주의: lead 라벨 부족)** |
| sangam_pertO_v2 | obj 0 | OK(frame100%) | lead 0% → SCC 기여 0 | **재가공 필요** (aux/inlane 객체 재생성) |
| sangam_pertX_v2 | obj 0 | OK(frame100%) | lead 0% → SCC 기여 0 | **재가공 필요** (동일) |
| DriverB_new | (eval만) | OK(98.3%) | OK(lead 45%) | **즉시 학습 가능** (held-out 겸용) |
| spd_v2 세계 A(260519) | i1 짝 없음 | prebake↔i1 0% | inlane OK | **재가공 필요** (i1 cache 260519 로 재생성 후 통일) |
| spd_v2 세계 B(260520) | OK(nolead↔obj 모순) | inlane/prebake 없음 | GT 없음 | **재가공 필요** (이름 분리 + inlane_gt 생성) |
| spd_nolead_0521 | nolead 정상(0.7%) | prebake 미생성 | — | **대기** (정상 nolead, f1 제외 타당) |
| sangam_0520_agent3 | aux 없음 | prebake 없음(bank만) | — | **대기/재가공** (bank/traj 만, 1,489 logged) |
| sangam_0522_40m / spd_0522_40m | aux 없음 | prebake 없음 | — | **대기** (40m 원거리, bank/synth_v2 만) |
| spd_0519_pertO | logged_bank 0 file | traj_bank 47,490 만 | — | **대기/재가공** (logged bank 비어 있음) |

---

## §6. 기존 DATA_INVENTORY.md 대조 — 부정확·누락 지적

| DATA_INVENTORY 기술 | 실측 결과 | 판정 |
|---|---|---|
| §2 `spd_v2` = "streaming 1 run, seg/a2 학습 핵심" | 실제 2개 run(260519+260520) 혼재, iy 학습 valid 0 (전량 drop) | **부정확/치명적 누락** |
| §2 `sangam_pertO_v2/pertX_v2` = "a2/f1/f2" 용도 | inlane_gt lead 0%, aux obj 0% → f_2 SCC 기여 0 | **부정확** (f_2 무용) |
| §2 `spd_nolead_0520` = "f1 제외, obj 85% mismatch" | obj 67%(실측), 단 a_2 학습엔 spd_v2 명목으로 실투입 | 일부 정정 (85%→67%) + 투입 사실 누락 |
| §2 `sangam_lead_0521` obj 23% | obj_any 32.5% / inlane 28% (실측) | 근사 일치 (정정 28~32%) |
| §3.5 `cache/b0/` ~1.1GB 존재 | **빈 폴더** | **부정확** (산출물 없음) |
| §3.5 `i1_tok_cache_*_hr.npz`(80-token) | cache 내 `_hr` 파일 미발견 | **누락/부정확** (현재 없음) |
| §3.6 `a2_targets/`(구) | **빈 폴더** | 정정 (empty) |
| §3.7 legacy v3 prebake corrupt | ax range 직접 확인 (sangam_v3=0, spd_v3 [-112,29], pertO_v3 [-169,266]) | **확정** (기술 정확) |
| §8 mismatch 목록 | spd_v2 run 혼선 / pertO_v2 객체 0 / iy spd drop **모두 미기재** | **누락 3건** |

### 못 본 것 (명시)
- NFS 내부(`seg_masks_v2`, `multiagnet_compact_meta`, old DriverB raw 상세): 1-level ls 만 — 내부 규모 [미확인].
- aux obj % 는 2,500 frame subsample (전수 아님). inlane_gt 는 전수.
- closing degenerate(정지객체 상수) 정밀 검증: 본 6종 inlane_gt 는 std>1.6 으로 비-degenerate, 그 외 대기 ds 는 inlane_gt 미생성 [미확인].
- iy 모델 best val 수치, ckpt 성능: 본 작업 범위 밖.
- `eval_a2_expand_driverb.sh` 가 참조하는 `i1_tok_cache_DriverB_new_40k_subset_prev20.npz` 및 b0 40k cache 실재 여부 [미확인=b0 폴더 빈 점으로 보아 없을 가능성].

---

## 부록. 실측 명령 (재현용)
- npz 메타: `python /tmp/probe_all.py <npz...>` (keys n/first/last/run, shape via zip header)
- 교집합: prebake `curr_keys` ∩ i1 cache `keys` set 비교
- aux obj: `aux_targets_<ds>/*.npz` 의 `target`(40,7) ch0>0.5 빈도 (2,500 subsample)
- inlane: `inlane_gt_<ds>.npz` 의 `inlane_dist` finite&<999 비율, `inlane_closing` 통계

# 경량 seg backbone 평가 — b0 학습 결과 + 교체 판단 보류

> 작성 2026-05-26. 목적 = closed-loop 추론 latency 단축을 위해 seg backbone (현행 `tf_efficientnetv2_s`, ckpt `seg_pretrain_fx640_v3`, mIoU 0.921) 을 더 가벼운 backbone 으로 교체할 가치가 있는지 평가.
>
> seg 는 a_2 의 **frozen feature extractor** (카메라 ROI → 40×256 토큰). 따라서 최종 영향은 seg mIoU 가 아니라 b0 feature 위에서 a_2 의 obj 인지/gate 가 견디는지로 판단해야 한다 (§4 미래 TODO). 본 보고서 시점에는 **b0 seg 학습만 확정**, a_2-on-b0 ablation 은 미수행 → 교체 결정 **보류**.

---

## 1. 한 줄 요약

- 후보 sweep 중 **b0 (`tf_efficientnetv2_b0`) 만 끝까지 학습**, b1/b2 는 중단 (§3 근거).
- b0: i_1 토큰 경로 latency **2.14x 단축** (5.01 → 2.34 ms), seg **mIoU 0.897** (best; baseline 0.921 대비 −0.024).
- 주의: **vehicle class IoU 0.866 → 0.811 (−0.055)** — lead car 인지에 직결되는 최약 클래스. obj 인지/gate 영향은 미측정.
- 결론: latency 이득 vs vehicle IoU 저하의 obj/gate 영향이 미지수 → **a_2-on-b0 ablation 전 교체 보류** (§4).

---

## 2. 확정 결과 (confirmed)

### 2.1 후보 × {latency speedup, seg mIoU, vehicle IoU}

latency 는 `scripts/bench_seg_backbone_latency.py` 실측 (입력 (1,3,250,1280), fp32, CUDA event, warmup 20 + 100 iter; `outputs_multi/seg_backbone_latency.json`).
"i_1 tok speedup" = closed_loop_node 실시간 병목 경로 (features_only `out_indices[-1]` → `adaptive_avg_pool2d(1,40)`) 의 v2_s 대비 배속. mIoU / vehicle IoU 는 b0 만 confirmed, 나머지 후보는 미학습.

| backbone | i_1 tok fp32 (ms) | i_1 tok speedup | full fp32 (ms) | tok GFLOPs | seg mIoU | vehicle IoU |
|---|---|---|---|---|---|---|
| `tf_efficientnetv2_s` (baseline) | 5.01 | 1.00x | 6.95 | 18.52 | 0.921 | 0.866 |
| `tf_efficientnetv2_b0` (학습 완료) | 2.34 | **2.14x** | 3.97 | 4.63 | **0.897** | **0.811** |
| `tf_efficientnet_lite0` | 2.11 | 2.37x | 4.03 | 2.45 | 미학습 | 미학습 |
| `mobilenetv3_large_100` | 1.65 | 3.05x | 3.49 | 1.44 | 미학습 | 미학습 |
| `tf_efficientnetv2_b1` | 3.03 | 1.65x | 4.52 | 6.58 | 미학습 (중단) | 미학습 (중단) |
| `tf_efficientnetv2_b2` | 3.10 | 1.62x | 4.74 | 7.56 | 미학습 (중단) | 미학습 (중단) |
| `tf_efficientnetv2_b3` | 3.77 | 1.33x | 5.60 | 10.42 | 미학습 | 미학습 |

caveat:
- `mobilenetv3_large_100`: tok speedup 3.05x 로 최고이나 last channel = **960** (v2_s 256, b0 192 대비 큼) → downstream a_2 / fc_proj 입력이 무거워져 토큰 경로 이득이 downstream 으로 상쇄될 수 있음. 채택 시 별도 재학습·재측정 필요. (hypothesis)
- fp16 (amp): batch=1 에서 v2_s/b0 모두 fp32 보다 오히려 느림 (b0 tok fp32 2.34 vs fp16 2.92 ms). 실배포 batch=1 이므로 fp16 이득 없음.
- baseline v2_s 의 mIoU 0.921 / vehicle 0.866 은 별도 기록 (`seg_pretrain_fx640_v3`) 수치이며, 이번 latency run / b0 train.log 에는 v2_s class IoU 가 없다.

### 2.2 b0 class별 IoU 비교 (seg 4-class)

b0 = `outputs_multi/seg_pretrain_lite_b0/train.log` 최종 epoch (ep15) 인용. v2_s 는 별도 기록.

| class | v2_s IoU | b0 IoU (ep15) | delta |
|---|---|---|---|
| bg | 0.914* | 0.914 | ~0 |
| road | 0.959* | 0.959 | ~0 |
| **vehicle** | 0.866 | **0.811** | **−0.055** |
| wall | 0.902* | 0.902 | ~0 |
| **mIoU** | 0.921 | 0.897 (best) / 0.896 (ep15) | −0.024 |

(*) bg/road/wall 의 v2_s 값은 b0 와 거의 동일 수준으로 추정 — 본 run 에 v2_s class IoU 가 없어 confirmed 아님. 핵심은 **저하가 vehicle 에 집중**된다는 점.

### 2.3 b0 학습 mIoU 추이 (15 epoch, train.log)

| epoch | val_loss | mIoU | vehicle IoU |
|---|---|---|---|
| 01 | 0.135 | 0.838 | 0.633 |
| 03 | 0.113 | 0.880 | 0.770 |
| 05 | 0.124 | 0.885 | 0.794 |
| 08 | 0.134 | 0.890 | 0.800 |
| 11 | 0.143 | **0.897** | 0.816 |
| 13 | 0.150 | 0.897 | 0.815 |
| 15 | 0.151 | 0.896 | 0.811 |

- best mIoU 0.897 (ep11/ep13), best val_loss 0.1132 (ep03). 산출물: `best_miou_backbone.pt` / `best_miou_full.pt` / `best_valloss_*.pt`.
- vehicle IoU 가 ep11 (0.816) 부터 plateau, 이후 epoch 에서도 v2_s 0.866 에 도달 못함. train_loss 는 0.035 까지 떨어졌으나 vehicle IoU 정체 → 추가 epoch 으로 메워질 gap 으로 보기 어려움 (hypothesis: b0 capacity 한계, 학습 부족 아님).
- 학습 설정: 15 epoch, batch 8, lr 3e-4 cosine, amp, extra meta = speedway_agent1 + sangam_fx640, fx=640 (메모리 규약 `fx640_only` 준수).

---

## 3. b1/b2 중단 근거 (confirmed)

목적이 **경량화(속도)** 인데:

- b1/b2 는 b0 보다 무겁다 (tok GFLOPs b0 4.63 → b1 6.58 → b2 7.56; tok speedup b0 2.14x → b1 1.65x → b2 1.62x). 속도 이득이 b0 보다 작다.
- mIoU 만 b0(0.897)~v2_s(0.921) 사이로 소폭 상승할 뿐, **속도 축에서 b0 에 dominated** (더 느리면서 더 좋은 게 아니라, 속도가 목적인데 더 느림).
- Pareto 의 "경량 끝점" 판단에는 **b0 (가장 빠른 학습 후보) + v2_s (정확도 기준점)** 두 점이면 충분 → b1/b2 학습은 정보 이득 대비 GPU 시간 낭비라 중단.
- b3 도 같은 이유로 미학습 (latency 만 측정, 1.33x 로 이득 미미).

(주: lite0 / mobilenetv3 는 b0 와 다른 family 라 dominated 아님 — 추후 후보로 보존, 단 §2.1 caveat.)

---

## 4. 미래 TODO — backbone 교체 가치 판단 (미결, 가장 중요)

본 보고서는 **seg 단독 품질 + latency** 만 확정했다. 교체 결정은 아래가 풀려야 가능하며, 현재는 **보류**다.

1. **판단 기준은 seg mIoU 가 아니라 a_2-on-b0 품질.**
   seg 는 frozen feature extractor → 최종 영향은 b0 feature 위에서 a_2 의 obj prec/recall · dist · gate 가 무너지지 않는지다. 특히 **vehicle IoU 0.811 저하 (−0.055) 가 obj 인지 / gate 에 주는 영향**을 측정해야 한다. (b0 feature 로 i1/i2 cache 재생성 → a_2 재학습 → obj F1 / closing recall / gate jitter 를 v2_s 기준선과 비교하는 ablation 필요.)

2. **교체 = 전체 cascade 재학습 (비싸고 비가역).**
   seg backbone 변경 시 i1 / i2 cache 전부 무효 → a_2 + f_2 재학습 (STATE_AND_LESSONS §2.5 와 동일 함정). commit 전 반드시 a_2-on-b0 품질 확인. 미확인 상태 교체 금지.

3. **진행 중 gate 안정화 작업과 상충 가능.**
   가벼운 backbone 이 vehicle IoU 를 깎으면 obj 인지가 noisy → gate jitter 증가 → 깜빡임(brake on/off flicker) 악화 우려. gate 는 현재 `fixed_thresh = sigmoid((a2_obj_amax − 0.3)·5)` 로 안정화 중 (STATE_AND_LESSONS §2.3). 순서 = **gate 안정성 매듭 → 그 다음 backbone 교체 판단.** 두 작업을 동시에 흔들지 말 것.

4. **결론 보류.**
   latency **2.14x** 이득 vs vehicle IoU **−0.055** 의 obj/gate 영향이 미지수다. a_2-on-b0 ablation (위 1) 전에는 교체 결정을 내리지 않는다. ablation 에서 obj F1 / closing recall / gate jitter 가 v2_s 대비 무너지지 않음이 확인되면 그때 교체 가치 (2.14x) 를 재평가한다.

---

## 5. 산출물 경로

- b0 학습: `outputs_multi/seg_pretrain_lite_b0/` (`best_miou_backbone.pt`, `best_miou_full.pt`, `best_valloss_*.pt`, `train.log`, `viz_progress/`)
- latency 실측: `outputs_multi/seg_backbone_latency.json`, 측정 스크립트 `scripts/bench_seg_backbone_latency.py`
- baseline seg: `data/ckpts/seg_pretrain_fx640_v3_backbone.pt` (mIoU 0.921)

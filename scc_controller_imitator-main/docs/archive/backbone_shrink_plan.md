# Backbone 경량화 plan (2026-05-25)

## 현재
- `tf_efficientnetv2_s` (timm) — **~21M params**
- input: 720×1280 (jpeg)
- output: 40 bin tokens (256-dim each)
- inference: ~30-50ms per frame on RTX 3080 Ti

## 경량화 후보

| backbone | params | timm name | 비고 |
|---|---|---|---|
| **efficientnetv2_b0** | 7.1M | `tf_efficientnetv2_b0` | 1/3, 가장 가까운 family |
| **efficientnetv2_b1** | 8.1M | `tf_efficientnetv2_b1` | 1/2.5, 약간 더 큼 |
| efficientnet_b0 | 5.3M | `efficientnet_b0` | 구 family, gradient 더 안정 |
| **mobilenetv3_small** | 2.5M | `mobilenetv3_small_100` | 1/8, 모바일 target |
| mobilenetv3_large | 5.5M | `mobilenetv3_large_100` | 1/4 |

## 변경 필요 파일

| 파일 | 변경 |
|---|---|
| `src/model/image_models/segmentation_model.py` | backbone name 인자 |
| `src/model/image_models/a2_model.py` | `LateralStripEncoder` backbone 변경 + `d_token` 조정 |
| 모든 launch 스크립트 | `--img-backbone` |

## 학습 파이프라인 (full)

```
Stage 1: seg pretrain (with new backbone)        ~3-4h
Stage 2: a_2 head (cache build + train)          ~1-2h (cache fast)
Stage 3: f_1 (iy) train (cache 새로 만들어야)      ~2-3h
```

**Total: ~8-10h for full pipeline.** v7b cache 결과 보고 결정.

## 검증 metric (경량화 가치 판단)

| metric | baseline (v2_s) | target |
|---|---|---|
| inference latency | 30-50ms | < 15ms |
| obj rec (DriverB) | 63% (v7b) | ≥ 50% |
| vL_rel corr (DriverB) | +0.40 | ≥ +0.30 |
| iy val loss | TBD (v25j) | 같거나 ≤ 110% |

5% 이내 정확도 손실로 latency 2-3× 개선 → 채택. 그 이상 손실이면 다음 ablation.

## 추천 시작 후보

**efficientnetv2_b0** — same family, 시작점으로 weight transfer 가능 (timm 기본 pretrained ImageNet). seg pretrain 만 새로 학습하면 됨.

## 실행 (morning 시 user 결정)

```bash
# Stage 1: seg pretrain (b0)
python scripts/train_seg_pretrain.py --backbone tf_efficientnetv2_b0 --epochs 30 ...

# Stage 2: a_2 cache + train
python scripts/build_a2_cache.py --backbone tf_efficientnetv2_b0 ...
python scripts/train_a2_v5_from_cache.py --d-out 128 ...

# Stage 3: i_2 cache + iy train (B variant)
bash scripts/launch_v25j_B0.sh   # (수정 필요, 새 backbone path)
```

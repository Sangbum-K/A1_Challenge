# Performance — IyModel forward 소요 시간

`scripts/profile_iy_model.py` 로 측정 (CUDA Event 기반, 30 trial 평균).
GPU: **RTX 3080 Ti (12 GB)**.

## v3 (current, 벡터화 i_y) — 2026-05-21

`yaw_shift_tokens_batch` 를 `torch.gather` 기반 single-kernel 로 벡터화 (이전 Python loop + `.item()` 제거).

### batch 64 (D=128, C_b=256, use_obj_gate=True)

| module | ms | % | samples/sec |
|--------|----|---|-------------|
| i_y (token roll) | 0.05 | 0.6 | 1,500,000+ |
| img_proj (Conv1d k=1) ×2 | 0.07 | 0.9 | 850,000 |
| **i_2 ImageFusion (3 attn blocks)** | 1.31 | 17.4 | 49,000 |
| i_g (obj_head + gate) | 0.03 | 0.4 | 2,700,000 |
| traj+state embed + pos_enc | 0.04 | 0.5 | 1,400,000 |
| concat traj+image | 0.01 | 0.2 | 4,300,000 |
| **transformer (6L self-attn)** | **6.01** | **80.0** | 10,600 |
| fc_out (D→2) | 0.02 | 0.3 | 2,700,000 |
| **TOTAL** | **7.53** | 100 | **8,495** |

### batch 256

| module | ms | % | samples/sec |
|--------|----|---|-------------|
| i_y | 0.05 | 0.1 | 4.9M |
| img_proj | 0.21 | 0.6 | 1.2M |
| **i_2 ImageFusion** | 8.54 | 22.5 | 30,000 |
| i_g | 0.03 | 0.1 | 7.8M |
| traj+state embed | 0.11 | 0.3 | 2.4M |
| concat | 0.05 | 0.1 | 5.4M |
| **transformer (6L)** | **29.01** | **76.3** | 8,800 |
| fc_out | 0.04 | 0.1 | 7.2M |
| **TOTAL** | **38.03** | 100 | **6,731** |

### 시사점
- **transformer (6L) 가 76-80% 차지** — 진짜 main cost (정상)
- i_2 ImageFusion (3 attn) 는 17-22% — 두 번째 큰 비중
- i_y, i_g, img_proj 등은 합쳐서 < 2% — 이제 무시 가능

→ 추가 가속은 transformer (multi-head attn) 또는 i_2 layer 수 줄이는 방향

## v2 (이전, Python loop i_y)

`yaw_shift_tokens_batch` 에서 batch element 마다 `dx[b].item()` GPU→CPU sync + `torch.roll`.

### batch 64

| module | ms | % |
|--------|-----|---|
| **i_y (Python loop, GPU sync)** | **65.22** | **90.3** |
| transformer | 5.31 | 7.4 |
| i_2 | 1.39 | 1.9 |
| 기타 | < 1 | < 1 |
| TOTAL | 72.25 | 100 |

### batch 256

| module | ms | % |
|--------|-----|---|
| **i_y (Python loop)** | **138.45** | **79.3** |
| transformer | 29.00 | 16.6 |
| i_2 | 5.82 | 3.3 |
| TOTAL | 174.50 | 100 |

## v2 → v3 변화

| batch | v2 total | v3 total | speedup | throughput v3 |
|-------|----------|----------|---------|---------------|
| 64    | 72.25 ms | 7.53 ms  | **9.6×** | 8,495 sps |
| 256   | 174.50 ms| 38.03 ms | **4.6×** | 6,731 sps |

i_y 만 보면: 138 ms → 0.05 ms (**~2,700× 가속**)

## 학습 시간 추정 (참고)

학습 시 데이터로더 overhead + backward + optim step 포함 → forward 시간 × 약 2-3배.

| 데이터 | samples | batch | 예상 epoch 시간 | 30 epoch |
|--------|---------|-------|-----------------|----------|
| sangam fx640 + speedway | 10M | 64 | ~30-50 min | 15-25h |
| speedway only (3.7M) | 3.7M | 64 | ~10-15 min | 5-8h |
| sangam fx640 only (6.3M) | 6.3M | 64 | ~20-30 min | 10-15h |

(실측은 dataloader / GPU util 에 좌우. 위는 forward 만 기준 lower bound)

## 재현
```bash
python scripts/profile_iy_model.py --batch 16 64 128 256 --n-trial 30
```

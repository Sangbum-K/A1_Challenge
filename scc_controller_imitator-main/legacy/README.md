# legacy/

v2/v3 refactor 이전 코드. 현재 학습/추론에 사용 안 함.

새 파이프라인 (현재 사용):
- `scripts/train_iy.py` (f_1), `scripts/train_a2_head.py` (a_2), `scripts/train_seg_pretrain.py` (seg)
- `scripts/prebake_iy_dataset.py`, `scripts/compute_i1_tok_cache.py`
- `src/model/image_models/iy_model.py`
- `src/datamodule/scc_synth_iy_dataset.py`, `a2_pair_dataset.py`, `seg_pretrain_dataset.py`

## 이동된 파일 (deprecated)

### scripts/legacy/scripts/

| 파일 | 대체 | 비고 |
|------|------|------|
| `train_scc.py` | `train_iy.py` | 최초 학습 스크립트 (v1) |
| `train_f1.py` | `train_iy.py` | f_1 e2e 학습 (v1) |
| `train_f1_cached.py` | `train_iy.py` | fi_mem cache 기반 f_1 (v2) |
| `eval_f1_cached.py` | `viz_a2_inference.py`, `viz_seg_inference.py` | 옛 eval |
| `eval_open_loop.py` | (TODO: IyModel 기반 새 eval) | scc_dataset 의존 |
| `prebake_f1_dataset.py` | `prebake_iy_dataset.py` | obj_target / fx tensor 없음 |
| `compute_fi_mem_cache.py` | `compute_i1_tok_cache.py` | per-frame fi_mem (v2). 신규는 per-image i_1 token |
| `viz_fi_mem_cache.py` | `viz_a2_inference.py` | |
| `prepare_random_trajs.py` | `prepare_new_run_bank.py` | jiwon_e2e index 의존 → raw 직접 scan |
| `prepare_logged_trajs.py` | `prepare_logged_trajs_v2.py` | parallel + arg-driven |

### legacy/src/datamodule/

| 파일 | 대체 |
|------|------|
| `scc_dataset.py` | `scc_synth_iy_dataset.py` |
| `scc_synth_cached_dataset.py` | `scc_synth_iy_dataset.py` |
| `scc_synth_prebake_dataset.py` | `scc_synth_iy_dataset.py` |

### legacy/src/model/image_models/

| 파일 | 대체 |
|------|------|
| `image_scc_models.py` (SCCControllerImitator v1) | `iy_model.py` |

## 아직 유지 (현재 사용 중)

- `src/datamodule/scc_synth_pair_dataset.py` — utility (DEG2RAD, _parse_ego, _world_to_ego_traj, _parse_scc_txt) 가 `scc_synth_iy_dataset.py` / `prebake_iy_dataset.py` 에서 import
- `src/model/image_models/image_scc_models_v2.py` — `scripts/closed_loop_node.py` 가 import. IyModel 전환 시 같이 legacy 로

## 복원 방법
필요 시 그대로 import path 만 바꾸면 동작:
```python
# 옛: from src.model.image_models.image_scc_models import SCCControllerImitator
# 신: from legacy.src.model.image_models.image_scc_models import SCCControllerImitator
```
또는 git history 에서 복구.

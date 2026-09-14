# Upstream copy manifest

- Copied/corrected on: 2026-09-13
- Upstream root: `../scc_controller_imitator-main/`
- Upstream Git metadata: absent in the delivered tree
- Hash algorithm: SHA-256
- Core scope: physical camera input through 21-frame a₂ output

## Runtime dependency: bundled exact senior files

The real-camera runner imports the byte-identical senior source copied into this project. It does not require the original repository after transfer to another PC.

| Runtime path through `senior_code` | SHA-256 | Use |
|---|---|---|
| `src/model/image_models/a2_model.py` | `6763cfe442ec7be32eb22d0873b2d8b8577073b3d1f435fb6878d99337c2f5a0` | exact `A2ModelMultiFrame` class |
| `data/ckpts/seg_pretrain_fx640_v3_backbone.pt` | `8b9262a5bdd214081b62223f1fee3bb96f50c55b3f3b5bcc3300a5ba30a8fd91` | exact segmentation-pretrained backbone weights |
| `data/ckpts/a2_mf20_v2s_n20.pt` | `c095b16f15cd73614f2f829bba774095e9c65626b895d4cf1558692953f44003` | exact 21-frame a₂ weights |

`senior_code` is a real directory, not a symbolic link. It contains only the exact upstream `a2_model.py` and the two required checkpoints. `senior_code/COPY_MANIFEST.sha256` records all three hashes, and `scripts/verify_portable_bundle.py` validates the bundle using only Python's standard library.

The runner records all three hashes in `summary.json`, so a result proves which original model code and weights were used.

## New real-camera code

| File | Purpose |
|---|---|
| `scripts/live_camera_a2_21fps.py` | physical UVC camera → i₁ → timestamped 21-frame a₂; throughput/latency/history logging |
| `configs/real_camera_mf20.yaml` | real-camera baseline contract and pass criteria |
| `docs/REAL_CAMERA_A2_21FPS_EXPERIMENT.md` | experiment design limited to camera-through-a₂ |

## Required checkpoint artifacts found upstream

These files are duplicated byte-for-byte inside `senior_code/data/ckpts/` so the experiment folder is self-contained.

| Role | Upstream file | Size | SHA-256 |
|---|---|---:|---|
| segmentation backbone | `data/ckpts/seg_pretrain_fx640_v3_backbone.pt` | about 77 MB | `8b9262a5bdd214081b62223f1fee3bb96f50c55b3f3b5bcc3300a5ba30a8fd91` |
| 21-frame a₂ | `data/ckpts/a2_mf20_v2s_n20.pt` | about 83 MB | `c095b16f15cd73614f2f829bba774095e9c65626b895d4cf1558692953f44003` |

Out-of-scope f₁/f₂/controller, training/cache scripts, modified copies and duplicate documents were removed from the portable experiment folder during cleanup on 2026-09-14.

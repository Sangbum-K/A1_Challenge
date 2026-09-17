# A1 Challenge workspace instructions

Before answering or changing anything related to the real-camera perception experiment, read `A1_RealCamera_A2_21FPS/PROJECT_CONTEXT.md` completely. Treat it as the persistent handoff for prior sessions.

Key constraints:

- The current goal stops at a₂. Do not redefine it as an f₂/controller-output experiment.
- The question is whether one physical Hikrobot camera can drive `camera_aravis2 → ROS2 → preprocessing → segmentation-pretrained backbone/i₁ → 21-frame A2ModelMultiFrame → a₂` at sustained 21 FPS.
- Camera transport/acquisition has already been validated separately; do not make camera bandwidth/FPS validation the main experiment again.
- `A1_RealCamera_A2_21FPS/senior_code/` is a self-contained byte-identical copy, not a symlink. Do not edit vendored senior files or checkpoints. Verify them with `python3 A1_RealCamera_A2_21FPS/scripts/verify_portable_bundle.py`.
- The live runner now has a one-camera `camera_aravis2 + ROS2` subscriber adapter, but it has not yet been verified with a connected Hikrobot camera. State this clearly until actual BayerRG8 receipt, debayer, QoS, and sustained runtime are verified. OpenCV UVC remains fallback-only.
- Distinguish 21 FPS from 21-frame input: current token plus 20 historical tokens at target ages 0.05–1.00 seconds.
- Keep throughput success separate from sim-to-real output validity.

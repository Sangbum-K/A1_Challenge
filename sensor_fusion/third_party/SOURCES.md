# Perception references and licenses

- **Directly compiled code:** [KAIST Patchwork++](https://github.com/url-kaist/patchwork-plusplus),
  commit `3e6903a1d5537a4cc2ace897b0bbb98a92d6014c`, BSD-2-Clause (`patchwork-plusplus/LICENSE`).
  Uses upstream `cpp/patchworkpp/src/patchworkpp.cpp` and `cpp/common/src/plane_fit.cpp` unchanged.
  [IROS 2022 paper](https://arxiv.org/abs/2207.11919). RNR disabled because this adapter uses XYZ only.
- **Architecture reference, not copied implementation:** Autoware
  [Euclidean clustering](https://docs.autoware.org/main/tutorials/integrating-autoware/tuning-parameters-and-performance/tuning-parameters/perception-tuning/)
  and [ROI cluster fusion](https://autowarefoundation.github.io/autoware_universe/main/perception/autoware_image_projection_based_fusion/docs/roi-cluster-fusion/).
  PCL VoxelGrid + KD-tree Euclidean clustering are used. Current camera supplies horizontal bins,
  not a YOLO 2D rectangle/class: matching is deliberately weaker horizontal IoU plus model-band overlap.
- **Algorithm reference, independently implemented adapter:**
  [SORT (ICIP 2016)](https://arxiv.org/abs/1602.00763),
  [official repository](https://github.com/abewley/sort).
  Constant-velocity Kalman + global Hungarian association in LiDAR BEV, with distance/Mahalanobis/size gates.
  This is not the original image-box SORT implementation, nor an Autoware EKF package port.
  No SORT source was copied. Detection quality and ego-motion compensation remain separate requirements.

A geometric LiDAR cluster is not a semantic vehicle detector. Unique projected-bin overlap is a
candidate association, not evidence of correct physical identity without calibration and inspection.

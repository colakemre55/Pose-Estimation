
# 3D Human Pose Estimation using MVGFormer  
**Evaluation and Custom Dataset Adaptation**

This project evaluates MVGFormer for multi-view 3D human pose estimation on both a public benchmark (Shelf dataset) and a custom multi-camera dataset captured at the University of Bonn. The work focuses on validating the pipeline, adapting camera calibration to Panoptic format, and assessing accuracy with ArUco-based ground truth.

MVGFormer is evaluated on the Shelf dataset by comparing its 3D outputs to multi-view triangulations derived from MediaPipe 2D joints. The system is then adapted to a custom multi-camera stage at the University of Bonn by converting camera calibrations to Panoptic format and resolving scale/orientation mismatches. Ground truth is established using 12 ArUco markers placed on anatomical landmarks. MPJPE is computed on four sequences. MVGFormer achieves lower error than Faster‑VoxelPose in most sequences but runs significantly slower.

 Comparison against Faster‑VoxelPose.

## Method Overview
MVGFormer combines:
- **Appearance Module (AM)** for refining 2D pose features using attention.
- **Geometry Module (GM)** for triangulating 3D joints with confidence weighting.

The model iteratively refines poses by alternating AM and GM in Transformer decoder layers.

## Datasets
- **Shelf Dataset**: Used to validate the pipeline and compare MediaPipe triangulation vs MVGFormer.
- **Custom Bonn Dataset**: Five synchronized cameras (IDs 19, 25, 30, 31, 39) with converted calibration.

## Calibration & Adaptation
Key fixes for custom data:
- Converted calibration to Panoptic format.
- Fixed inverted extrinsics (Y‑axis up orientation).
- Downscaled images (4608×5328 → 1152×1332) and scaled intrinsics accordingly.

## Evaluation
Metric: **Mean Per Joint Position Error (MPJPE)**

Four sequences were evaluated using ArUco markers as ground truth. Errors were highest at distal joints (wrists, ankles, knees), especially under occlusions or unusual poses.

### Sequence Errors (MPJPE)
- **Seq 1**: 98.94 mm  
- **Seq 2**: 152.24 mm  
- **Seq 3**: 248.17 mm  
- **Seq 4**: 227.89 mm  


## References
- MVGFormer  
- MediaPipe  

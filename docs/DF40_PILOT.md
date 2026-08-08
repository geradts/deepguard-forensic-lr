# DF40 pilot protocol

## Purpose

Establish a reproducible development experiment using DF40 and DeepfakeBench before external validation.

## Current upstream facts

DF40 provides 40 distinct deepfake generation techniques and an official repository with benchmark code. DeepfakeBench provides a unified benchmark with multiple detector implementations.

## Pilot design

1. Select a fixed subset of DF40 generators.
2. Keep subjects grouped across folds.
3. Keep generators grouped when performing cross-generator evaluation.
4. Run Xception and FTCN using the exact configs/checkpoints from the installed DeepfakeBench version.
5. Export one score per video and preserve all provenance metadata.
6. Add DeepGuard classical features only as separate feature dimensions.
7. Calibrate/fuse on development folds only.
8. Freeze the LR system.
9. Evaluate on an untouched external dataset.

## Required score table

```text
video_id,dataset,source_id,subject_id,generator_id,label,detector,score
```

## Important

Do not combine detector confidence scores by simple averaging and call the result an LR. The detector scores are observations. The LR model is calibrated separately.

Do not use Deepfake-Eval-2024, AV-Deepfake1M++ or another external test set during feature selection or calibration.

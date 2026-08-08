# DeepGuard — Forensic Deepfake LR Laboratory

DeepGuard is a research framework for multimodal deepfake detection and forensic likelihood-ratio evaluation.

## Current pipeline

```text
video -> detector adapters -> feature table -> development/calibration -> LiR -> external validation
```

### Detector adapters

- `detectors/heuristic.py` — dependency-light pipeline baseline; **not a validated detector**.
- `detectors/deepfakebench.py` — adapter for an externally installed DeepfakeBench checkout.
- `run_detector.py` — batch runner for the baseline.
- `run_deepfakebench.py` — batch runner for DeepfakeBench.

DeepfakeBench itself is intentionally not vendored into this repository. Install the upstream project and its licensed model weights separately.

## DeepfakeBench adapter

The adapter expects a command template that accepts `{video}` and `{output}` and writes JSON such as:

```json
{
  "score": 0.73,
  "label": 1,
  "features": {"detector_score": 0.73}
}
```

Example shape:

```bash
python run_deepfakebench.py ./data/videos --checkout ./external/DeepfakeBench --command "python YOUR_UPSTREAM_INFERENCE.py --video {video} --output {output}" --output deepfakebench_features.csv
```

The exact upstream inference command depends on the DeepfakeBench version, detector, checkpoint and configuration. Do not copy a command from another version without checking its current documentation.

## Forensic LR principle

Detector confidence is **not** a likelihood ratio. Keep detector outputs as measurement features. Train/calibrate the final LR system on development data and freeze it before independent validation.

Recommended validation:

- development: DF40, AV-Deepfake1M, DeepfakeBench
- external: Deepfake-Eval-2024, AV-Deepfake1M++, MAVOS-DD, GenVidBench, HydraFake

Record dataset versions, generator/subject/source grouping, detector commit, model hash, preprocessing, LiR version and calibration parameters.

## Status

The repository now contains the runnable detector adapter layer and LiR hand-off. Actual DeepfakeBench inference requires the upstream checkout and model weights to be installed locally; those assets are not included here.

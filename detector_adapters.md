# Detector adapters

DeepGuard v10 uses an adapter architecture so detector scores remain separate from evidential LR interpretation.

## DeepfakeBench

Primary detector execution layer. Suggested detector families include Xception, EfficientNet, I3D, FTCN, AltFreezing and supported CLIP-based methods.

Export at minimum:

    video_id
    dataset
    source_id
    subject_id
    generator_id
    label
    detector
    score

## AV-Deepfake1M

For multimodal experiments export visual, audio, audio-visual and temporal/localisation scores where available.

## Fusion

Do not average detector confidences and call the result an LR. Keep detector outputs as separate measurement features and learn the final LR mapping using development data.

## Reproducibility

Record detector repository commit, checkpoint/model hash, preprocessing version and inference configuration for every detector.

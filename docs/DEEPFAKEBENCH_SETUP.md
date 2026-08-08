# DeepfakeBench detector setup

DeepGuard now targets two concrete upstream detectors first:

1. **FTCN** — video detector for temporal coherence.
2. **Xception** — spatial face detector baseline.

The upstream DeepfakeBench repository currently documents 36 detectors, including eight video detectors and recent methods such as TALL, LSDA and Effort. It also provides pretrained weights and evaluates detectors with `training/test.py`. See the upstream README for the exact version-specific setup. 

## 1. Install upstream DeepfakeBench

```bash
git clone https://github.com/SCLBD/DeepfakeBench.git external/DeepfakeBench
cd external/DeepfakeBench
conda create -n DeepfakeBench python=3.7.2
conda activate DeepfakeBench
sh install.sh
```

DeepfakeBench also provides a Dockerfile for a reproducible environment.

## 2. Obtain the data and weights

Do not put datasets or model weights into the DeepGuard repository. Follow the upstream dataset terms and download the required pretrained weights into:

```text
external/DeepfakeBench/training/weights/
```

For FTCN the upstream configuration uses an I3D/3D-R50 pretrained backbone and a video configuration with 16-frame clips and 224-pixel resolution. For Xception the upstream configuration uses 32 frames and 256-pixel resolution.

## 3. Run FTCN

```bash
bash scripts/run_deepfakebench_eval.sh \
  external/DeepfakeBench/training/config/detector/ftcn.yaml \
  external/DeepfakeBench/training/weights/ftcn_best.pth \
  "Celeb-DF-v2 DeepFakeDetection DFDCP UADFV"
```

## 4. Run Xception

```bash
bash scripts/run_deepfakebench_eval.sh \
  external/DeepfakeBench/training/config/detector/xception.yaml \
  external/DeepfakeBench/training/weights/xception_best.pth \
  "Celeb-DF-v2 DeepFakeDetection DFDCP UADFV"
```

## 5. Export scores for DeepGuard

DeepfakeBench's native evaluation output is not assumed to have a stable schema across commits. The DeepGuard adapter therefore requires a small parser that converts the installed DeepfakeBench result/features into:

```text
video_id
label
source_id
subject_id
generator_id
detector
score
```

The resulting score is a **detector measurement**, not an LR.

## 6. LiR experiment

Use the detector scores as separate features:

```text
neural_score_ftcn
neural_score_xception
spatial_score
frequency_score
compression_score
face_consistency_score
av_sync_score
```

Then perform the LR calibration using development data only. Freeze the complete system before evaluating Deepfake-Eval-2024, AV-Deepfake1M++, MAVOS-DD, GenVidBench or other external data.

## Why FTCN + Xception first?

FTCN provides a temporal video signal, while Xception provides a strong spatial face baseline. Their complementarity is scientifically useful for the first multimodal/fusion experiment. DeepfakeBench reports broad cross-dataset evaluations and supports both detectors.

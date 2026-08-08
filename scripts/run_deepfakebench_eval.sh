#!/usr/bin/env bash
set -euo pipefail

DFFB="${DFFB:-$PWD/external/DeepfakeBench}"
DETECTOR_CONFIG="${1:-$DFFB/training/config/detector/ftcn.yaml}"
WEIGHTS="${2:-$DFFB/training/weights/ftcn_best.pth}"
DATASETS="${3:-Celeb-DF-v2 DeepFakeDetection DFDCP UADFV}"

cd "$DFFB"

# DeepfakeBench-v2 officially evaluates detectors through training/test.py.
# Keep the external checkpoint outside DeepGuard and record its SHA-256.
python3 training/test.py \
  --detector_path "$DETECTOR_CONFIG" \
  --test_dataset $DATASETS \
  --weights_path "$WEIGHTS"

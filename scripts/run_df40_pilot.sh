#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_df40_pilot.sh /path/to/DF40 /path/to/DeepfakeBench
#
# This script prepares the pilot workspace and prints the two official
# DeepfakeBench inference commands. It does not download data or weights.

DF40_ROOT="${1:?DF40 root required}"
Dfb_ROOT="${2:?DeepfakeBench root required}"

mkdir -p results/detector_scores results/features results/lir

echo "DF40: ${DF40_ROOT}"
echo "DeepfakeBench: ${Dfb_ROOT}"
echo

echo "Xception configuration:"
echo "${Dfb_ROOT}/training/config/detector/xception.yaml"
echo

echo "FTCN configuration:"
echo "${Dfb_ROOT}/training/config/detector/ftcn.yaml"
echo

echo "Run inference with the exact checkpoint/config combination documented by the installed DeepfakeBench release."
echo "Then export video-level scores with: video_id,dataset,source_id,subject_id,generator_id,label,detector,score"
echo "Finally feed the merged table into the DeepGuard LiR layer."

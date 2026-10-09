#!/bin/bash
# Train one fold of Dataset1001 with the gap-cutout augmentation pilot
# (nnUNetTrainer_onlyMirror01_gapcut: onlyMirror01 + tooth-erasure DA).
# tail-optimization-095 Wave 2 pilot. GPUs 1,2,3 (GPU 0 free), global batch 8.
# Usage: bash scripts/train_fold_gapcut.sh <fold>
set -e

FOLD="${1:?Usage: bash scripts/train_fold_gapcut.sh <fold>}"

source dataset_conversion/setup_nnunet_env.sh
export nnUNet_compile=false
export CUDA_VISIBLE_DEVICES=1,2,3

echo "=========================================="
echo "Training fold $FOLD (Dataset1001 gapcut pilot, GPUs 1,2,3)"
echo "=========================================="

/root/miniconda3/envs/nnunet310/bin/python -m nnunetv2.run.run_training 1001 3d_fullres "$FOLD" \
    -num_gpus 3 \
    --npz \
    --c \
    -tr nnUNetTrainer_onlyMirror01_gapcut

echo "Training fold $FOLD complete!"

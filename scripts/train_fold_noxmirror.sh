#!/bin/bash
# Train one fold of Dataset1001 with x-axis mirroring disabled
# (nnUNetTrainer_onlyMirror01: mirrors z/y only; TTA axes (0,1)).
# GPUs 1,2,3 (GPU 0 free), global batch 8.
# Usage: bash scripts/train_fold_noxmirror.sh <fold>
set -e

FOLD="${1:?Usage: bash scripts/train_fold_noxmirror.sh <fold>}"

source dataset_conversion/setup_nnunet_env.sh
export nnUNet_compile=false
export CUDA_VISIBLE_DEVICES=1,2,3

echo "=========================================="
echo "Training fold $FOLD (Dataset1001 no-x-mirror, GPUs 1,2,3)"
echo "=========================================="

/root/miniconda3/envs/nnunet310/bin/python -m nnunetv2.run.run_training 1001 3d_fullres "$FOLD" \
    -num_gpus 3 \
    --npz \
    --c \
    -tr nnUNetTrainer_onlyMirror01

echo "Training fold $FOLD complete!"

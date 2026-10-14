#!/bin/bash
# Combo pilot: gap-cutout DA (p=0.3, same as the plain-UNet gapcut pilot) on the
# ResEncL base (nnUNetResEncUNetLPlans, patch 112x224x256). tail-optimization-095
# final training-track hypothesis: the two failed pilots are each other's cure —
# gapcut suppresses ResEncL's phantom/drift family, ResEncL capacity keeps the
# recall family. Single-variable A/B vs both pilots (base architecture only).
# GPUs 1,2,3 (GPU 0 free), DDP local batch 1 -> global 3
# (plans batch manually overridden 2->3, see train_fold_resenc_l.sh).
# Usage: bash scripts/train_fold_resenc_l_gapcut.sh <fold>
set -e

FOLD="${1:?Usage: bash scripts/train_fold_resenc_l_gapcut.sh <fold>}"

source dataset_conversion/setup_nnunet_env.sh
export nnUNet_compile=false
export CUDA_VISIBLE_DEVICES=1,2,3

echo "=========================================="
echo "Training fold $FOLD (Dataset1001 gapcut x ResEncL combo pilot, GPUs 1,2,3)"
echo "=========================================="

/root/miniconda3/envs/nnunet310/bin/python -m nnunetv2.run.run_training 1001 3d_fullres "$FOLD" \
    -num_gpus 3 \
    -p nnUNetResEncUNetLPlans \
    -tr nnUNetTrainer_onlyMirror01_gapcut \
    --npz \
    --c

echo "Training fold $FOLD complete!"

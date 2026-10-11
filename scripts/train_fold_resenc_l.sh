#!/bin/bash
# Train one fold of Dataset1001 with the Residual-Encoder UNet (ResEncL plans,
# 24GB target: patch 112x224x256 = 3x plain-UNet context) + onlyMirror01
# (mirror axes (0,1)). tail-optimization-095 Wave 3 pilot.
# GPUs 1,2,3 (GPU 0 free), DDP local batch 1 -> global 3
# (plans batch manually overridden 2->3: DDP requires global batch >= world_size;
#  per-GPU VRAM ~12GB, well within A10 22.5GB).
# Usage: bash scripts/train_fold_resenc_l.sh <fold>
set -e

FOLD="${1:?Usage: bash scripts/train_fold_resenc_l.sh <fold>}"

source dataset_conversion/setup_nnunet_env.sh
export nnUNet_compile=false
export CUDA_VISIBLE_DEVICES=1,2,3

echo "=========================================="
echo "Training fold $FOLD (Dataset1001 ResEncL pilot, GPUs 1,2,3)"
echo "=========================================="

/root/miniconda3/envs/nnunet310/bin/python -m nnunetv2.run.run_training 1001 3d_fullres "$FOLD" \
    -num_gpus 3 \
    -p nnUNetResEncUNetLPlans \
    -tr nnUNetTrainer_onlyMirror01 \
    --npz \
    --c

echo "Training fold $FOLD complete!"

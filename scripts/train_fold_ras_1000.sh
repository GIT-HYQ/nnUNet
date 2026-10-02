#!/bin/bash
# Train a single fold of the RAS-frame model (Dataset1000_ToothSemanticRAS, built
# from tooth_fairy2_m). Uses GPUs 1,2,3 only (GPU 0 left free).
# Global batch 8 (from plans) -> 3/3/2 per GPU on 3-way DDP.
# Archived Dataset1000-era script (renamed with 1000 tag on 2026-10-02).
# Usage: bash scripts/train_fold_ras_1000.sh <fold>

set -e

FOLD="${1:?Usage: bash scripts/train_fold_ras_1000.sh <fold>}"

# Source environment
source dataset_conversion/setup_nnunet_env.sh

# Disable torch.compile (Triton CUDA compatibility issue on this hardware)
export nnUNet_compile=false

# Use the last 3 of the 4 GPUs
export CUDA_VISIBLE_DEVICES=1,2,3

echo "=========================================="
echo "Training fold $FOLD (Dataset1000 RAS, GPUs 1,2,3)"
echo "=========================================="

# --c: resume from checkpoint_final -> checkpoint_latest -> checkpoint_best if any
# exists in the fold's output folder; a fresh fold (no checkpoint) only prints a
# warning and starts from scratch. To force a fresh start, delete the fold's
# results dir first.
/root/miniconda3/envs/nnunet310/bin/python -m nnunetv2.run.run_training 1000 3d_fullres "$FOLD" \
    -num_gpus 3 \
    --npz \
    --c

echo "Training fold $FOLD complete!"

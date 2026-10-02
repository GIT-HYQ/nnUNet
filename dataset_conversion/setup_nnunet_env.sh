#!/bin/bash
# nnUNet environment setup script
# Usage: source dataset_conversion/setup_nnunet_env.sh

export nnUNet_raw="/home/share/clr/share/work/CBCT/nnUNet/tooth_3d_semantic/raw"
export nnUNet_preprocessed="/home/share/clr/share/work/CBCT/nnUNet/tooth_3d_semantic/preprocessed"
export nnUNet_results="/home/share/clr/share/work/CBCT/nnUNet/tooth_3d_semantic/results"

echo "nnUNet environment configured:"
echo "  nnUNet_raw=$nnUNet_raw"
echo "  nnUNet_preprocessed=$nnUNet_preprocessed"
echo "  nnUNet_results=$nnUNet_results"

#!/bin/bash
# Sequential fold 1-4 for Dataset1001 (start after fold_0 gate PASS).
# Stops on first failure. Resume-safe via --c in the per-fold script.
set -e
cd /home/share/clr/share/work/CBCT/nnUNet
for F in 1 2 3 4; do
  if ! bash scripts/train_fold_noxmirror.sh "$F"; then
    echo "FOLD $F FAILED -- stopping sequence" >&2
    exit 1
  fi
done
echo "ALL FOLDS 1-4 COMPLETE"

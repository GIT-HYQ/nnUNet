#!/bin/bash
# Train all 5 folds of Dataset1000_ToothSemanticRAS sequentially, GPUs 1,2,3,
# global batch 8 (3/3/2 per GPU).
# Each fold gets its own log: tooth_3d_semantic/logs/train_ras_fold<N>.log
# Stops at the first failing fold (check its log before re-running).
# Archived Dataset1000-era script (renamed with 1000 tag on 2026-10-02).
set -o pipefail

cd /home/share/clr/share/work/CBCT/nnUNet
LOGDIR=tooth_3d_semantic/logs
mkdir -p "$LOGDIR"

for FOLD in 0 1 2 3 4; do
    echo "=== [$(date '+%F %T')] START fold $FOLD ==="
    if bash scripts/train_fold_ras_1000.sh "$FOLD" 2>&1 | tee "$LOGDIR/train_ras_fold${FOLD}.log"; then
        echo "=== [$(date '+%F %T')] fold $FOLD DONE ==="
    else
        echo "=== [$(date '+%F %T')] fold $FOLD FAILED — stopping, see $LOGDIR/train_ras_fold${FOLD}.log ==="
        exit 1
    fi
done

echo "=== [$(date '+%F %T')] ALL 5 FOLDS (Dataset1000 RAS) COMPLETE ==="

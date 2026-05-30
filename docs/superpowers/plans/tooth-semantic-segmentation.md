# 牙齿3D语义分割 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 使用 nnUNetv2 官方工具链训练 32 类牙齿语义分割模型，支持对新的 CBCT 图像进行推理。

**Architecture:** 数据转换脚本将现有扁平文件组织为 nnUNetv2 标准格式 → nnUNetv2 PlanAndPreprocess（spacing分析/实验规划/预处理）→ nnUNetv2 Train（4卡DDP）→ 独立 inference.py 用于新图像预测。

**Tech Stack:** Python 3.10+, nnUNetv2 2.7.0, PyTorch, SimpleITK, numpy, PyYAML

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `dataset_conversion/setup_nnunet_env.sh` | 设置 nnUNet 环境变量的一键脚本 |
| `dataset_conversion/ToothFairy2Semantic.py` | 将扁平数据转为 nnUNetv2 标准格式（imagesTr/labelsTr/dataset.json/splits_final.json） |
| `inference/tooth_predict.py` | 对新 CBCT 做推理，输出每颗牙的 mask |
| `scripts/train_fold.sh` | 训练单折的封装脚本 |

---

### Task 1: 创建环境变量设置脚本

**Files:**
- Create: `dataset_conversion/setup_nnunet_env.sh`

- [ ] **Step 1: 编写环境设置脚本**

```bash
#!/bin/bash
# nnUNet environment setup script
# Usage: source dataset_conversion/setup_nnunet_env.sh

export nnUNet_raw="/home/share/clr/share/data/CBCT/tooth_3d_semantic/raw"
export nnUNet_preprocessed="/home/share/clr/share/data/CBCT/tooth_3d_semantic/preprocessed"
export nnUNet_results="/home/share/clr/share/data/CBCT/tooth_3d_semantic/results"

echo "nnUNet environment configured:"
echo "  nnUNet_raw=$nnUNet_raw"
echo "  nnUNet_preprocessed=$nnUNet_preprocessed"
echo "  nnUNet_results=$nnUNet_results"
```

- [ ] **Step 2: 添加执行权限**

```bash
chmod +x dataset_conversion/setup_nnunet_env.sh
```

---

### Task 2: 创建数据转换脚本 `ToothFairy2Semantic.py`

**Files:**
- Create: `dataset_conversion/ToothFairy2Semantic.py`

- [ ] **Step 1: 编写数据转换脚本**

```python
#!/usr/bin/env python3
"""
Convert ToothFairy2 flat data to nnUNetv2 standard format.

Reads:
  - /home/share/clr/share/data/CBCT/tooth_fairy2/*.nii.gz (images)
  - /home/share/clr/share/data/CBCT/tooth_fairy2/*_gt.nii.gz (labels)
  - /home/share/clr/share/data/CBCT/tooth_fairy2/list/dataset.yaml (label_map)
  - /home/share/clr/share/data/CBCT/tooth_fairy2/list/tooth_split_k5_seed0.yaml (5-fold split)

Writes:
  - /home/share/clr/share/data/CBCT/tooth_3d_semantic/raw/Dataset999_ToothSemantic/dataset.json
  - /home/share/clr/share/data/CBCT/tooth_3d_semantic/raw/Dataset999_ToothSemantic/splits_final.json
  - /home/share/clr/share/data/CBCT/tooth_3d_semantic/raw/Dataset999_ToothSemantic/imagesTr/*.nii.gz (named case_0000.nii.gz)
  - /home/share/clr/share/data/CBCT/tooth_3d_semantic/raw/Dataset999_ToothSemantic/labelsTr/*.nii.gz
"""

import os
import sys
import shutil
import argparse
import yaml
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json, isfile, maybe_mkdir_p


# Base paths
TOOTH_FAIRY2_DIR = Path("/home/share/clr/share/data/CBCT/tooth_fairy2")
LIST_DIR = TOOTH_FAIRY2_DIR / "list"
OUTPUT_BASE = Path("/home/share/clr/share/data/CBCT/tooth_3d_semantic")
RAW_DIR = OUTPUT_BASE / "raw"
DATASET_NAME = "Dataset999_ToothSemantic"
DATASET_DIR = RAW_DIR / DATASET_NAME


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_label_map() -> dict:
    """Load label mapping from dataset.yaml."""
    data = load_yaml(str(LIST_DIR / "dataset.yaml"))
    return data["raw_to_train"]


def get_case_names() -> list:
    """Get list of all case names from dataset.yaml raw_to_train keys."""
    label_map = get_label_map()
    cases = []
    for raw_id in sorted(label_map.keys()):
        train_id = label_map[raw_id]
        if train_id == 0:
            continue
        # Determine prefix and numeric part
        # Raw IDs: 11-18 -> F (upper right), 21-28 -> P (upper left)
        #           31-38 -> F (lower left), 41-48 -> P (lower right)
        if 11 <= raw_id <= 18:
            prefix = "ToothFairy2F"
            num = raw_id - 10
        elif 21 <= raw_id <= 28:
            prefix = "ToothFairy2P"
            num = raw_id - 20
        elif 31 <= raw_id <= 38:
            prefix = "ToothFairy2F"
            num = raw_id - 30
        elif 41 <= raw_id <= 48:
            prefix = "ToothFairy2P"
            num = raw_id - 40
        else:
            continue
        case_name = f"{prefix}_{num:03d}"
        cases.append(case_name)
    return sorted(set(cases))


def get_split_data() -> list:
    """Load 5-fold split and convert to nnUNetv2 format."""
    split_data = load_yaml(str(LIST_DIR / "tooth_split_k5_seed0.yaml"))
    folds = split_data["folds"]
    splits = []
    for fold_info in folds:
        fold_idx = fold_info["fold"]
        train_cases = sorted(fold_info["train"])
        val_cases = sorted(fold_info["val"])
        splits.append({
            "train": train_cases,
            "val": val_cases,
        })
    return splits


def convert_label_to_nnunet_format(raw_label_map: dict) -> dict:
    """Convert label mapping to nnUNetv2 dataset.json labels format."""
    # Build reverse map: train_id -> class_name
    train_to_names = {}
    for raw_id, train_id in raw_label_map.items():
        if train_id == 0:
            continue
        name = f"tooth_{train_id}"
        train_to_names.setdefault(train_id, []).append(name)

    labels = {"background": 0}
    for tid in sorted(train_to_names.keys()):
        # Use the first name for this class
        labels[f"tooth_{tid}"] = tid
    return labels


def setup_directories():
    """Create output directories."""
    maybe_mkdir_p(str(DATASET_DIR / "imagesTr"))
    maybe_mkdir_p(str(DATASET_DIR / "labelsTr"))


def copy_file_with_rename(src: Path, dst: Path, rename_suffix: str = "") -> bool:
    """Copy a file, optionally renaming it."""
    if not src.exists():
        print(f"  WARNING: {src} not found, skipping")
        return False
    dst_path = dst
    if rename_suffix:
        dst_path = Path(str(dst) + rename_suffix)
    shutil.copy2(str(src), str(dst_path))
    return True


def convert_dataset():
    """Main conversion logic."""
    print("=" * 60)
    print("ToothFairy2 -> nnUNetv2 Semantic Dataset Converter")
    print("=" * 60)

    # Step 1: Load configs
    print("\n[1/6] Loading configurations...")
    label_map = get_label_map()
    case_names = get_case_names()
    splits = get_split_data()
    print(f"  Label map: {len(label_map)} entries")
    print(f"  Cases with teeth (label > 0): {len(case_names)}")
    print(f"  Folds: {len(splits)}")

    # Step 2: Create directories
    print("\n[2/6] Creating output directories...")
    setup_directories()
    print(f"  Output: {DATASET_DIR}")

    # Step 3: Copy images and labels
    print("\n[3/6] Copying image and label files...")
    images_copied = 0
    labels_copied = 0
    skipped = []

    for case_name in case_names:
        img_src = TOOTH_FAIRY2_DIR / f"{case_name}.nii.gz"
        lbl_src = TOOTH_FAIRY2_DIR / f"{case_name}_gt.nii.gz"

        # Check both files exist
        if not img_src.exists() or not lbl_src.exists():
            skipped.append(case_name)
            continue

        # Copy image as case_0000.nii.gz (nnUNetv2 requirement)
        copy_file_with_rename(img_src, DATASET_DIR / "imagesTr" / f"{case_name}", "_0000.nii.gz")
        images_copied += 1

        # Copy label as case.nii.gz (no suffix)
        copy_file_with_rename(lbl_src, DATASET_DIR / "labelsTr" / f"{case_name}")
        labels_copied += 1

    if skipped:
        print(f"  Skipped {len(skipped)} cases with missing files:")
        for s in skipped[:5]:
            print(f"    - {s}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")
    else:
        print(f"  All {images_copied} cases copied successfully")

    # Step 4: Generate dataset.json
    print("\n[4/6] Generating dataset.json...")
    labels = convert_label_to_nnunet_format(label_map)
    num_classes = max(labels.values())

    dataset_json = {
        "channel_names": {
            "0": "CBCT"
        },
        "labels": labels,
        "numTraining": images_copied,
        "file_ending": ".nii.gz",
        "name": DATASET_NAME,
        "description": "ToothFairy2 3D semantic tooth segmentation (32 classes + background)",
    }

    save_json(dataset_json, str(DATASET_DIR / "dataset.json"))
    print(f"  Classes: {num_classes} teeth + background = {num_classes + 1}")
    print(f"  Labels: {labels}")

    # Step 5: Generate splits_final.json
    print("\n[5/6] Generating splits_final.json...")
    save_json(splits, str(DATASET_DIR / "splits_final.json"))
    print(f"  {len(splits)} folds written")

    # Step 6: Summary
    print("\n[6/6] Conversion complete!")
    print(f"  Dataset directory: {DATASET_DIR}")
    print(f"  Images: {images_copied} files in imagesTr/")
    print(f"  Labels: {labels_copied} files in labelsTr/")
    print(f"  dataset.json: {DATASET_DIR / 'dataset.json'}")
    print(f"  splits_final.json: {DATASET_DIR / 'splits_final.json'}")

    # Print the nnUNet dataset ID for reference
    print(f"\nTo use with nnUNetv2, this is Dataset ID 999.")
    print(f"Set environment variables:")
    print(f"  export nnUNet_raw='{RAW_DIR}'")
    print(f"  export nnUNet_preprocessed='{OUTPUT_BASE / 'preprocessed'}'")
    print(f"  export nnUNet_results='{OUTPUT_BASE / 'results'}'")

    return images_copied


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ToothFairy2 to nnUNetv2 format")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be done")
    args, remaining = parser.parse_known_args()

    if not args.dry_run:
        count = convert_dataset()
        print(f"\nCopied {count} cases to nnUNetv2 format.")
    else:
        label_map = get_label_map()
        case_names = get_case_names()
        splits = get_split_data()
        print(f"Would process {len(case_names)} cases across {len(splits)} folds.")
```

- [ ] **Step 2: 添加执行权限**

```bash
chmod +x dataset_conversion/ToothFairy2Semantic.py
```

---

### Task 3: 创建推理脚本 `tooth_predict.py`

**Files:**
- Create: `inference/tooth_predict.py`
- Create: `inference/__init__.py` (empty)

- [ ] **Step 1: 编写推理脚本**

```python
#!/usr/bin/env python3
"""
Predict tooth segmentation on a new CBCT image using a trained nnUNetv2 model.

Usage:
    python inference/tooth_predict.py \
        --model_dir /home/share/clr/share/data/CBCT/tooth_3d_semantic/results/Dataset999_ToothSemantic/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0 \
        --input /path/to/new_scan.nii.gz \
        --output /path/to/output.nii.gz

The model directory should contain: checkpoint_final.pth, plans.json, dataset.json
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import join, isfile, maybe_mkdir_p

# Ensure nnUNet paths are set
def setup_nnunet_env():
    """Set up nnUNet environment variables if not already set."""
    base = "/home/share/clr/share/data/CBCT/tooth_3d_semantic"
    os.environ.setdefault("nnUNet_raw", join(base, "raw"))
    os.environ.setdefault("nnUNet_preprocessed", join(base, "preprocessed"))
    os.environ.setdefault("nnUNet_results", join(base, "results"))


def predict_single_image(model_dir: str, input_path: str, output_path: str):
    """Run prediction on a single CBCT image."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    setup_nnunet_env()

    print(f"Loading model from: {model_dir}")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device('cuda'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )

    # Prepare input/output directories
    input_dir = str(Path(input_path).parent)
    input_filename = Path(input_path).name
    output_dir = str(Path(output_path).parent)
    maybe_mkdir_p(output_dir)

    print(f"Predicting: {input_path}")
    print(f"Output: {output_path}")

    # Run prediction
    predictor.predict_from_files(
        [join(input_dir, input_filename)],
        [output_path],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )

    print(f"Prediction saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Predict tooth segmentation on new CBCT")
    parser.add_argument('--model_dir', required=True, help='Model training output directory')
    parser.add_argument('--input', required=True, help='Input CBCT .nii.gz file')
    parser.add_argument('--output', required=True, help='Output segmentation .nii.gz file')
    args = parser.parse_args()

    if not isfile(args.model_dir + '/checkpoint_final.pth'):
        print(f"ERROR: checkpoint not found at {args.model_dir}/checkpoint_final.pth")
        sys.exit(1)
    if not isfile(args.input):
        print(f"ERROR: input file not found: {args.input}")
        sys.exit(1)

    predict_single_image(args.model_dir, args.input, args.output)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 添加执行权限**

```bash
chmod +x inference/tooth_predict.py
```

---

### Task 4: 创建训练封装脚本 `train_fold.sh`

**Files:**
- Create: `scripts/train_fold.sh`

- [ ] **Step 1: 编写训练脚本**

```bash
#!/bin/bash
# Train a single fold of the tooth semantic segmentation model.
# Usage: bash scripts/train_fold.sh <fold> [epochs]
# Example: bash scripts/train_fold.sh 0 1000

set -e

FOLD="${1:?Usage: bash scripts/train_fold.sh <fold> [epochs]}"
EPOCHS="${2:-1000}"

# Source environment
source dataset_conversion/setup_nnunet_env.sh

echo "=========================================="
echo "Training fold $FOLD for $EPOCHS epochs"
echo "=========================================="

nnUNetv2_train 999 3d_fullres "$FOLD" \
    -tr nnUNetTrainer_1000epochs \
    -num_gpus 4 \
    --npz

echo "Training fold $FOLD complete!"
```

- [ ] **Step 2: 添加执行权限**

```bash
chmod +x scripts/train_fold.sh
```

---

### Task 5: 运行转换并验证数据格式

**Files:**
- Modify: (none, uses scripts from Tasks 1-4)

- [ ] **Step 1: 设置环境变量**

Run in terminal:
```bash
source dataset_conversion/setup_nnunet_env.sh
```

- [ ] **Step 2: 运行数据转换**

Run in conda environment `nnunet`:
```bash
cd /home/share/clr/share/work/CBCT/nnUNet
python dataset_conversion/ToothFairy2Semantic.py
```

Expected output:
```
============================================================
ToothFairy2 -> nnUNetv2 Semantic Dataset Converter
============================================================
[1/6] Loading configurations...
  Label map: XX entries
  Cases with teeth (label > 0): 481
  Folds: 5
[2/6] Creating output directories...
[3/6] Copying image and label files...
  All 481 cases copied successfully
[4/6] Generating dataset.json...
  Classes: 32 teeth + background = 33
...
```

- [ ] **Step 3: 验证输出文件结构**

Run:
```bash
ls /home/share/clr/share/data/CBCT/tooth_3d_semantic/raw/Dataset999_ToothSemantic/
# Should show: dataset.json, imagesTr/, labelsTr/, splits_final.json

ls /home/share/clr/share/data/CBCT/tooth_3d_semantic/raw/Dataset999_ToothSemantic/imagesTr/ | head -5
# Should show files like: ToothFairy2F_001_0000.nii.gz

ls /home/share/clr/share/data/CBCT/tooth_3d_semantic/raw/Dataset999_ToothSemantic/labelsTr/ | head -5
# Should show files like: ToothFairy2F_001.nii.gz

cat /home/share/clr/share/data/CBCT/tooth_3d_semantic/raw/Dataset999_ToothSemantic/dataset.json | python -m json.tool | head -20
```

- [ ] **Step 4: 验证 dataset.json 关键字段**

Run:
```bash
python -c "
import json
with open('/home/share/clr/share/data/CBCT/tooth_3d_semantic/raw/Dataset999_ToothSemantic/dataset.json') as f:
    d = json.load(f)
assert d['numTraining'] == 481, f'Expected 481 training cases, got {d[\"numTraining\"]}'
assert len(d['labels']) == 33, f'Expected 33 labels (background + 32 teeth), got {len(d[\"labels\"])}'
assert d['file_ending'] == '.nii.gz', 'Wrong file ending'
print('dataset.json validation PASSED')
print(f'  numTraining: {d[\"numTraining\"]}')
print(f'  labels count: {len(d[\"labels\"])}')
print(f'  file_ending: {d[\"file_ending\"]}')
"
```

---

### Task 6: 运行 nnUNetv2 PlanAndPreprocess

**Files:**
- (none, uses nnUNetv2 CLI)

- [ ] **Step 1: 运行 plan_and_preprocess**

Run in conda environment `nnunet`:
```bash
cd /home/share/clr/share/work/CBCT/nnUNet
source dataset_conversion/setup_nnunet_env.sh

nnUNetv2_plan_and_preprocess -d 999 --verify_dataset_integrity -c 3d_fullres -npfp 8 -np 4
```

This will:
1. Extract dataset fingerprint (spacing, intensity stats)
2. Run experiment planning (determine patch/batch size)
3. Preprocess images (resampling to target spacing, normalization)

- [ ] **Step 2: 验证预处理输出**

Run:
```bash
ls /home/share/clr/share/data/CBCT/tooth_3d_semantic/preprocessed/Dataset999_ToothSemantic/nnUNetPlans_3d_fullres/data*.npz | wc -l
# Should show 481 files (one per training case)
```

---

### Task 7: 训练模型

**Files:**
- Modify: `scripts/train_fold.sh` (may need to adjust trainer class name)

- [ ] **Step 1: 调试用 — 训练 fold 0，少量 epoch**

Run in conda environment `nnunet`:
```bash
cd /home/share/clr/share/work/CBCT/nnUNet
source dataset_conversion/setup_nnunet_env.sh

nnUNetv2_train 999 3d_fullres 0 \
    -tr nnUNetTrainer_1000epochs \
    -num_gpus 4 \
    --disable_checkpointing
```

This runs a quick sanity check (no checkpoints saved, just verifies the pipeline works end-to-end).

- [ ] **Step 2: 正式训练 — fold 0，1000 epochs**

Run in conda environment `nnunet`:
```bash
nnUNetv2_train 999 3d_fullres 0 \
    -tr nnUNetTrainer_1000epochs \
    -num_gpus 4 \
    --npz
```

- [ ] **Step 3: 训练其余 folds**

Run in conda environment `nnunet`:
```bash
nnUNetv2_train 999 3d_fullres 1 -tr nnUNetTrainer_1000epochs -num_gpus 4 --npz
nnUNetv2_train 999 3d_fullres 2 -tr nnUNetTrainer_1000epochs -num_gpus 4 --npz
nnUNetv2_train 999 3d_fullres 3 -tr nnUNetTrainer_1000epochs -num_gpus 4 --npz
nnUNetv2_train 999 3d_fullres 4 -tr nnUNetTrainer_1000epochs -num_gpus 4 --npz
```

- [ ] **Step 4: 验证训练结果**

Run:
```bash
# Check that all folds produced checkpoint_final.pth
for fold in 0 1 2 3 4; do
    ckpt="/home/share/clr/share/data/CBCT/tooth_3d_semantic/results/Dataset999_ToothSemantic/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_${fold}/checkpoint_final.pth"
    if [ -f "$ckpt" ]; then
        echo "Fold $fold: OK ($(du -h "$ckpt" | cut -f1))"
    else
        echo "Fold $fold: MISSING"
    fi
done
```

---

### Task 8: 推理测试

**Files:**
- (uses inference/tooth_predict.py from Task 3)

- [ ] **Step 1: 对单个测试图像做预测**

Run in conda environment `nnunet`:
```bash
cd /home/share/clr/share/work/CBCT/nnUNet

# Pick a test case not used in training (from holdout_test set)
python inference/tooth_predict.py \
    --model_dir /home/share/clr/share/data/CBCT/tooth_3d_semantic/results/Dataset999_ToothSemantic/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0 \
    --input /home/share/clr/share/data/CBCT/tooth_fairy2/ToothFairy2P_084.nii.gz \
    --output /home/share/clr/share/data/CBCT/tooth_3d_semantic/pred_test.nii.gz
```

- [ ] **Step 2: 验证预测结果**

Run:
```bash
python -c "
import numpy as np
import SimpleITK as sitk
pred = sitk.GetArrayFromImage(sitk.ReadImage('/home/share/clr/share/data/CBCT/tooth_3d_semantic/pred_test.nii.gz'))
print(f'Shape: {pred.shape}')
print(f'Unique values: {np.unique(pred)}')
print(f'Non-zero voxels: {np.sum(pred > 0)}')
# Values should be in [0, 32] range
assert pred.min() >= 0 and pred.max() <= 32, f'Invalid label range: [{pred.min()}, {pred.max()}]'
print('Prediction validation PASSED')
"
```

---

## 完整使用流程总结

```bash
# 1. 设置环境
source dataset_conversion/setup_nnunet_env.sh

# 2. 转换数据（一次性）
python dataset_conversion/ToothFairy2Semantic.py

# 3. 预处理（一次性）
nnUNetv2_plan_and_preprocess -d 999 --verify_dataset_integrity -c 3d_fullres -npfp 8 -np 4

# 4. 训练（5折，每折约1-2天）
nnUNetv2_train 999 3d_fullres 0 -tr nnUNetTrainer_1000epochs -num_gpus 4 --npz
nnUNetv2_train 999 3d_fullres 1 -tr nnUNetTrainer_1000epochs -num_gpus 4 --npz
nnUNetv2_train 999 3d_fullres 2 -tr nnUNetTrainer_1000epochs -num_gpus 4 --npz
nnUNetv2_train 999 3d_fullres 3 -tr nnUNetTrainer_1000epochs -num_gpus 4 --npz
nnUNetv2_train 999 3d_fullres 4 -tr nnUNetTrainer_1000epochs -num_gpus 4 --npz

# 5. 推理新图像
python inference/tooth_predict.py \
    --model_dir .../fold_0 \
    --input new_scan.nii.gz \
    --output pred.nii.gz
```

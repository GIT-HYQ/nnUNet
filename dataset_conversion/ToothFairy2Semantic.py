#!/usr/bin/env python3
"""
Convert ToothFairy2 flat data to nnUNetv2 standard format.

Reads:
  - /home/share/clr/share/data/CBCT/tooth_fairy2/*.nii.gz (images)
  - /home/share/clr/share/data/CBCT/tooth_fairy2/*_gt.nii.gz (labels)
  - /home/share/clr/share/data/CBCT/tooth_fairy2/list/dataset.yaml (label_map)
  - /home/share/clr/share/data/CBCT/tooth_fairy2/list/tooth_split_k5_seed0.yaml (5-fold split)

Writes:
  - /home/share/clr/share/work/CBCT/nnUNet/tooth_3d_semantic/raw/Dataset999_ToothSemantic/dataset.json
  - /home/share/clr/share/work/CBCT/nnUNet/tooth_3d_semantic/raw/Dataset999_ToothSemantic/splits_final.json
  - imagesTr/*.nii.gz (named case_0000.nii.gz)
  - labelsTr/*.nii.gz
"""

import os
import sys
import shutil
import argparse
import yaml
import numpy as np
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import save_json, maybe_mkdir_p
import SimpleITK as sitk


# Base paths
TOOTH_FAIRY2_DIR = Path("/home/share/clr/share/data/CBCT/tooth_fairy2")
LIST_DIR = TOOTH_FAIRY2_DIR / "list"
OUTPUT_BASE = Path("/home/share/clr/share/work/CBCT/nnUNet/tooth_3d_semantic")
RAW_DIR = OUTPUT_BASE / "raw"
DATASET_NAME = "Dataset999_ToothSemantic"
DATASET_DIR = RAW_DIR / DATASET_NAME


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_label_map() -> dict:
    """Load label mapping from label_map.yaml."""
    data = load_yaml(str(LIST_DIR / "label_map.yaml"))
    return data["raw_to_train"]


def get_case_names() -> list:
    """Get list of all case names from dataset.yaml (plain list)."""
    data = load_yaml(str(LIST_DIR / "dataset.yaml"))
    # Filter out non-string entries and sort
    return sorted([c for c in data if isinstance(c, str)])


def get_split_data() -> list:
    """Load 5-fold split and convert to nnUNetv2 format."""
    split_data = load_yaml(str(LIST_DIR / "tooth_split_k5_seed0.yaml"))
    folds = split_data["folds"]
    splits = []
    for fold_info in folds:
        train_cases = sorted(fold_info["train"])
        val_cases = sorted(fold_info["val"])
        splits.append({
            "train": train_cases,
            "val": val_cases,
        })
    return splits


def convert_label_to_nnunet_format(raw_label_map: dict) -> dict:
    """Convert label mapping to nnUNetv2 dataset.json labels format."""
    train_to_names = {}
    for raw_id, train_id in raw_label_map.items():
        if train_id == 0:
            continue
        name = f"tooth_{train_id}"
        train_to_names.setdefault(train_id, []).append(name)

    labels = {"background": 0}
    for tid in sorted(train_to_names.keys()):
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
    dst_path = Path(str(dst) + rename_suffix)
    shutil.copy2(str(src), str(dst_path))
    return True


# GT masks are ALREADY in the 1..32 training label space (verified on all 480
# cases: union of non-zero values is exactly {1..32}). Applying the FDI
# raw_to_train map on top of them would zero out teeth 1-10 and shift the rest.
MAX_TRAIN_LABEL = 32


def copy_label_file(src: Path, dst: Path) -> bool:
    """Copy label file as-is (already in 1..32 space). Aborts on out-of-range values."""
    if not src.exists():
        print(f"  WARNING: {src} not found, skipping")
        return False
    label_sitk = sitk.ReadImage(str(src))
    label_np = sitk.GetArrayFromImage(label_sitk)
    uniq = np.unique(label_np)
    if int(uniq.max()) > MAX_TRAIN_LABEL:
        raise ValueError(
            f"{src}: label values up to {int(uniq.max())} exceed expected 0..{MAX_TRAIN_LABEL}. "
            f"GT is expected to be in 1..32 training space, not FDI. "
            f"Unique values: {uniq.tolist()}"
        )
    label_sitk_out = sitk.GetImageFromArray(label_np.astype(np.uint8))
    label_sitk_out.CopyInformation(label_sitk)
    sitk.WriteImage(label_sitk_out, str(dst))
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

        if not img_src.exists() or not lbl_src.exists():
            skipped.append(case_name)
            continue

        copy_file_with_rename(img_src, DATASET_DIR / "imagesTr" / f"{case_name}", "_0000.nii.gz")
        images_copied += 1

        copy_label_file(lbl_src, DATASET_DIR / "labelsTr" / f"{case_name}.nii.gz")
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
        "channel_names": {"0": "CBCT"},
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

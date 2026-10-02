#!/usr/bin/env python3
"""
Convert the RAS-corrected ToothFairy2 frame (tooth_fairy2_m) to nnUNetv2 standard
format, as Dataset1000_ToothSemanticRAS.

This is the same case set, crops, splits and holdout as Dataset999_ToothSemantic
(legacy RPI-array frame), but built from the z/y-corrected arrays stored as true
RAS (viewer-correct orientation). Do NOT mix the two: models trained on one frame
are not compatible with data of the other.

Reads:
  - /home/share/clr/share/data/CBCT/tooth_fairy2_m/*.nii.gz (images)
  - /home/share/clr/share/data/CBCT/tooth_fairy2_m/*_gt.nii.gz (labels)
  - /home/share/clr/share/data/CBCT/tooth_fairy2_m/list/dataset.yaml (case list)
  - /home/share/clr/share/data/CBCT/tooth_fairy2_m/list/label_map.yaml
  - splits: copied byte-identical from raw/Dataset999_ToothSemantic/splits_final.json

Writes:
  - tooth_3d_semantic/raw/Dataset1000_ToothSemanticRAS/dataset.json
  - .../splits_final.json
  - .../imagesTr/*.nii.gz (named case_0000.nii.gz)
  - .../labelsTr/*.nii.gz
"""

import json
import shutil
import yaml
import numpy as np
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import save_json, maybe_mkdir_p
import SimpleITK as sitk


TOOTH_FAIRY2_M_DIR = Path("/home/share/clr/share/data/CBCT/tooth_fairy2_m")
LIST_DIR = TOOTH_FAIRY2_M_DIR / "list"
OUTPUT_BASE = Path("/home/share/clr/share/work/CBCT/nnUNet/tooth_3d_semantic")
RAW_DIR = OUTPUT_BASE / "raw"
DATASET_NAME = "Dataset1000_ToothSemanticRAS"
DATASET_DIR = RAW_DIR / DATASET_NAME
SPLIT_SRC = RAW_DIR / "Dataset999_ToothSemantic" / "splits_final.json"

MAX_TRAIN_LABEL = 32  # GT is already in the 1..32 training label space


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_label_map() -> dict:
    return load_yaml(str(LIST_DIR / "label_map.yaml"))["raw_to_train"]


def get_case_names() -> list:
    data = load_yaml(str(LIST_DIR / "dataset.yaml"))
    return sorted([c for c in data if isinstance(c, str)])


def convert_label_to_nnunet_format(raw_label_map: dict) -> dict:
    labels = {"background": 0}
    for raw_id, train_id in raw_label_map.items():
        if train_id == 0:
            continue
        labels[f"tooth_{train_id}"] = train_id
    return labels


def copy_label_file(src: Path, dst: Path) -> None:
    """Validate label range/dtype; byte-copy if already uint8, else rewrite as uint8."""
    label_sitk = sitk.ReadImage(str(src))
    label_np = sitk.GetArrayFromImage(label_sitk)
    uniq = np.unique(label_np)
    if int(uniq.max()) > MAX_TRAIN_LABEL:
        raise ValueError(
            f"{src}: label values up to {int(uniq.max())} exceed expected 0..{MAX_TRAIN_LABEL}. "
            f"Unique values: {uniq.tolist()}"
        )
    if label_np.dtype == np.uint8:
        shutil.copy2(str(src), str(dst))
    else:
        out = sitk.GetImageFromArray(label_np.astype(np.uint8))
        out.CopyInformation(label_sitk)
        sitk.WriteImage(out, str(dst), compressionLevel=4)


def convert_dataset() -> int:
    print("=" * 60)
    print(f"{TOOTH_FAIRY2_M_DIR.name} -> {DATASET_NAME} converter")
    print("=" * 60)

    print("\n[1/6] Loading configurations...")
    label_map = get_label_map()
    case_names = get_case_names()
    print(f"  Label map: {len(label_map)} entries")
    print(f"  Cases: {len(case_names)}")

    if not SPLIT_SRC.is_file():
        raise FileNotFoundError(f"source split file missing: {SPLIT_SRC}")
    with open(str(SPLIT_SRC)) as f:
        splits = json.load(f)
    split_cases = set()
    for i, s in enumerate(splits):
        split_cases.update(s["train"])
        split_cases.update(s["val"])
    unknown = split_cases - set(case_names)
    if unknown:
        raise ValueError(f"{len(unknown)} split cases not in case list, e.g. {sorted(unknown)[:5]}")
    print(f"  Splits copied from Dataset999: {len(splits)} folds, {len(split_cases)} cases in folds")

    print("\n[2/6] Creating output directories...")
    maybe_mkdir_p(str(DATASET_DIR / "imagesTr"))
    maybe_mkdir_p(str(DATASET_DIR / "labelsTr"))

    print("\n[3/6] Copying image and label files...")
    images_copied = labels_copied = 0
    skipped = []
    for n, case_name in enumerate(case_names, 1):
        img_src = TOOTH_FAIRY2_M_DIR / f"{case_name}.nii.gz"
        lbl_src = TOOTH_FAIRY2_M_DIR / f"{case_name}_gt.nii.gz"
        if not img_src.exists() or not lbl_src.exists():
            skipped.append(case_name)
            continue
        shutil.copy2(str(img_src), str(DATASET_DIR / "imagesTr" / f"{case_name}_0000.nii.gz"))
        images_copied += 1
        copy_label_file(lbl_src, DATASET_DIR / "labelsTr" / f"{case_name}.nii.gz")
        labels_copied += 1
        if n % 50 == 0:
            print(f"  {n}/{len(case_names)} copied", flush=True)

    if skipped:
        raise RuntimeError(f"{len(skipped)} cases with missing files: {skipped[:5]}")
    print(f"  All {images_copied} cases copied (images + validated labels)")

    print("\n[4/6] Generating dataset.json...")
    dataset_json = {
        "channel_names": {"0": "CBCT"},
        "labels": convert_label_to_nnunet_format(label_map),
        "numTraining": images_copied,
        "file_ending": ".nii.gz",
        "name": DATASET_NAME,
        "description": "ToothFairy2 3D semantic tooth segmentation (32 classes + background), "
                       "RAS-corrected frame (tooth_fairy2_m, z/y flip of legacy arrays)",
    }
    save_json(dataset_json, str(DATASET_DIR / "dataset.json"))
    print(f"  Classes: {max(dataset_json['labels'].values())} teeth + background")

    print("\n[5/6] Writing splits_final.json (byte-identical to Dataset999)...")
    shutil.copy2(str(SPLIT_SRC), str(DATASET_DIR / "splits_final.json"))
    print(f"  {len(splits)} folds written")

    print("\n[6/6] Summary")
    n_img = len(list((DATASET_DIR / "imagesTr").glob("*_0000.nii.gz")))
    n_lbl = len(list((DATASET_DIR / "labelsTr").glob("*.nii.gz")))
    src_bytes = sum(p.stat().st_size for p in TOOTH_FAIRY2_M_DIR.glob("*.nii.gz"))
    dst_bytes = sum(p.stat().st_size for p in (DATASET_DIR / "imagesTr").glob("*.nii.gz")) + \
                sum(p.stat().st_size for p in (DATASET_DIR / "labelsTr").glob("*.nii.gz"))
    print(f"  imagesTr: {n_img} files ({dst_bytes / 1e9:.2f} GB of {src_bytes / 1e9:.2f} GB source)")
    print(f"  labelsTr: {n_lbl} files")
    if n_img != images_copied or n_lbl != labels_copied:
        raise RuntimeError("file count mismatch after copy")
    print(f"\nDone. Dataset ID 1000. Set nnUNet_raw/preprocessed/results as before.")
    return images_copied


if __name__ == "__main__":
    convert_dataset()

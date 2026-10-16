#!/usr/bin/env python3
"""Subset single-fold eval for the fold_0 gate.
Usage: python scripts/eval_subset.py --model_dir <trainer_dir> --folds 0 \
        --out <per_case.jsonl> [--id_assign on|off]
Evaluates the 40-case subset (20 worst + 20 best by Task-3 dice_before)."""
import argparse
import json
import os
import sys

import numpy as np
import SimpleITK as sitk
import torch

sys.path.insert(0, "/home/share/clr/share/work/CBCT/nnUNet")
from inference.id_assigner import apply_assigner

BASE = "/home/share/clr/share/work/CBCT/nnUNet"
CENSUS = f"{BASE}/tooth_3d_semantic/logs/lr_census_66.jsonl"
GT = f"{BASE}/tooth_3d_semantic/preprocessed/Dataset1000_ToothSemanticRAS/gt_segmentations"

os.environ.setdefault("nnUNet_raw", f"{BASE}/tooth_3d_semantic/raw")
os.environ.setdefault("nnUNet_preprocessed", f"{BASE}/tooth_3d_semantic/preprocessed")
os.environ.setdefault("nnUNet_results", f"{BASE}/tooth_3d_semantic/results")


def case_dice(seg, gt):
    d = []
    for c in np.unique(gt):
        if c == 0:
            continue
        g = gt == c
        p = seg == c
        d.append(2 * np.count_nonzero(p & g) / max(1, p.sum() + g.sum()))
    return float(np.mean(d)) if d else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--id_assign", choices=["on", "off"], default="off")
    a = ap.parse_args()

    with open(CENSUS) as f:
        rows = [json.loads(l) for l in f]
    worst20 = {r["case"] for r in sorted(rows, key=lambda r: r["dice_before"])[:20]}
    best20 = {r["case"] for r in sorted(rows, key=lambda r: -r["dice_before"])[:20]}
    cases = sorted(worst20 | best20)

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
                           use_mirroring=True, perform_everything_on_device=True,
                           device=torch.device("cuda"), verbose=False,
                           allow_tqdm=False)
    pred.initialize_from_trained_model_folder(
        a.model_dir, use_folds=[a.folds], checkpoint_name="checkpoint_final.pth")

    out_rows = []
    for i, c in enumerate(cases, 1):
        # read raw image for this case from Dataset1000 imagesTr
        raw = sitk.ReadImage(
            f"{BASE}/tooth_3d_semantic/raw/Dataset1000_ToothSemanticRAS/imagesTr/{c}_0000.nii.gz")
        arr = sitk.GetArrayFromImage(raw)
        seg = pred.predict_single_npy_array(
            arr[None], {"spacing": raw.GetSpacing()},
            save_or_return_probabilities=False)
        if a.id_assign == "on":
            seg, _ = apply_assigner(seg)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(f"{GT}/{c}.nii.gz"))
        d = case_dice(seg, gt)
        out_rows.append({"case": c, "group": "worst" if c in worst20 else "best",
                         "dice": round(float(d), 4)})
        print(f"[{i}/40] {c} {d:.4f}", flush=True)
    with open(a.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    w = [r["dice"] for r in out_rows if r["group"] == "worst"]
    b = [r["dice"] for r in out_rows if r["group"] == "best"]
    print(f"worst20 mean={np.mean(w):.4f}  best20 mean={np.mean(b):.4f}")


if __name__ == "__main__":
    main()

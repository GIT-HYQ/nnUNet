#!/usr/bin/env python3
"""Zero-training complementarity test: softmax-ensemble 1001 fold_0 (plain UNet)
with ResEncL fold_0 on the 40 gate cases, per gate protocol (checkpoint_final,
mirroring per trainer, tile step 0.5). Writes per-case dice for the 2-model
ensemble next to the single-model numbers.

Usage: python scripts/ensemble_gate_1001f0_resenc.py --out <jsonl>
"""
import argparse
import json
import os
import sys

import numpy as np
import SimpleITK as sitk
import torch

BASE = "/home/share/clr/share/work/CBCT/nnUNet"
GATE = f"{BASE}/tooth_3d_semantic/logs/gate_resenc_l_fold0_raw.jsonl"
GT = f"{BASE}/tooth_3d_semantic/preprocessed/Dataset1000_ToothSemanticRAS/gt_segmentations"
RES = f"{BASE}/tooth_3d_semantic/results/Dataset1001_ToothSemanticNoXMirror"
M1001 = f"{RES}/nnUNetTrainer_onlyMirror01__nnUNetPlans__3d_fullres"
MRESENC = f"{RES}/nnUNetTrainer_onlyMirror01__nnUNetResEncUNetLPlans__3d_fullres"

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


def mkpred(model_dir):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
                        use_mirroring=True, perform_everything_on_device=True,
                        device=torch.device("cuda"), verbose=False,
                        allow_tqdm=False)
    p.initialize_from_trained_model_folder(
        model_dir, use_folds=[0], checkpoint_name="checkpoint_final.pth")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(GATE)]
    p1 = mkpred(M1001)
    p2 = mkpred(MRESENC)

    out = []
    for i, r in enumerate(rows, 1):
        c = r["case"]
        img = sitk.ReadImage(f"{BASE}/tooth_3d_semantic/raw/Dataset1000_ToothSemanticRAS/imagesTr/{c}_0000.nii.gz")
        arr = sitk.GetArrayFromImage(img)
        props = {"spacing": img.GetSpacing()}
        _, prob1 = p1.predict_single_npy_array(arr[None], props, save_or_return_probabilities=True)
        _, prob2 = p2.predict_single_npy_array(arr[None], props, save_or_return_probabilities=True)
        if isinstance(prob1, torch.Tensor):
            prob1 = prob1.cpu().numpy()
        if isinstance(prob2, torch.Tensor):
            prob2 = prob2.cpu().numpy()
        seg = np.argmax(prob1 + prob2, axis=0).astype(np.uint8)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(f"{GT}/{c}.nii.gz"))
        d = case_dice(seg, gt)
        out.append({"case": c, "group": r["group"], "dice": round(float(d), 4)})
        print(f"[{i}/40] {c} ens={d:.4f} (1001f0={r['dice']:.4f})", flush=True)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("done ->", a.out)


if __name__ == "__main__":
    main()

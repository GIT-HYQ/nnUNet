#!/usr/bin/env python3
"""Census of L/R ID swaps on the 66-case strict holdout set (1000 ensemble
predictions), and measure the id-assigner's achievable mean (ceiling)."""
import json
import sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, "/home/share/clr/share/work/CBCT/nnUNet")
from inference.id_assigner import apply_assigner

BASE = "/home/share/clr/share/work/CBCT/nnUNet"
PRED = f"{BASE}/tooth_3d_semantic/eval/holdout_ensemble_ras/predictions"
GT = f"{BASE}/tooth_3d_semantic/preprocessed/Dataset1000_ToothSemanticRAS/gt_segmentations"
PER_CASE = f"{BASE}/tooth_3d_semantic/eval/holdout_ensemble_ras/per_case_metrics.jsonl"
OUT = f"{BASE}/tooth_3d_semantic/logs/lr_census_66.jsonl"

EXCLUDE = {"ToothFairy2P_113", "ToothFairy2P_136", "ToothFairy2P_356",
           "ToothFairy2F_049", "ToothFairy2P_049", "ToothFairy2P_105"}


def case_dice(seg, gt):
    d = {}
    for c in np.unique(gt):
        if c == 0:
            continue
        g = gt == c
        p = seg == c
        d[int(c)] = float(2 * np.count_nonzero(p & g) / max(1, p.sum() + g.sum()))
    return d


def main():
    with open(PER_CASE) as f:
        cases = sorted(json.loads(l)["case"] for l in f)
    cases = [c for c in cases if c not in EXCLUDE]
    assert len(cases) == 66, f"expected 66 cases, got {len(cases)}"
    rows = []
    for i, c in enumerate(cases, 1):
        pred = sitk.GetArrayFromImage(sitk.ReadImage(f"{PRED}/{c}.nii.gz"))
        gt = sitk.GetArrayFromImage(sitk.ReadImage(f"{GT}/{c}.nii.gz"))
        before = case_dice(pred, gt)
        new, log = apply_assigner(pred)
        after = case_dice(new, gt)
        fb = float(np.mean(list(before.values())))
        fa = float(np.mean(list(after.values())))
        rows.append({"case": c, "n_remaps": max(0, len(log) - 1),
                     "dice_before": round(fb, 4), "dice_after": round(fa, 4)})
        print(f"[{i}/66] {c} remaps={rows[-1]['n_remaps']} {fb:.4f} -> {fa:.4f}",
              flush=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    mb = float(np.mean([r["dice_before"] for r in rows]))
    ma = float(np.mean([r["dice_after"] for r in rows]))
    healthy = [r for r in rows if r["dice_before"] >= 0.9]
    bad = [r["case"] for r in healthy if r["n_remaps"] > 0]
    print(f"\n66-case mean: before={mb:.4f}  after={ma:.4f}")
    print(f"healthy (before>=0.9): {len(healthy)} cases; re-mapped: {bad or 'none'}")
    gains = sorted(rows, key=lambda r: -(r["dice_after"] - r["dice_before"]))
    print("top gains: " + ", ".join(
        f"{r['case']} +{r['dice_after']-r['dice_before']:.3f}" for r in gains[:5]))


if __name__ == "__main__":
    main()

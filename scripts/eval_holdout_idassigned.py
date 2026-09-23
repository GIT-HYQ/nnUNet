#!/usr/bin/env python3
"""Full holdout metrics with the L/R id-assigner applied to the saved
1000 5-fold ensemble predictions. Outputs before/after per case and the
66/69/72 aggregates + per-class table (1-32 with FDI annotation)."""
import json

import numpy as np
import SimpleITK as sitk
import sys

sys.path.insert(0, "/home/share/clr/share/work/CBCT/nnUNet")
from inference.id_assigner import apply_assigner

BASE = "/home/share/clr/share/work/CBCT/nnUNet"
PRED = f"{BASE}/tooth_3d_semantic/eval/holdout_ensemble_ras/predictions"
GT = f"{BASE}/tooth_3d_semantic/preprocessed/Dataset1000_ToothSemanticRAS/gt_segmentations"
OUT = f"{BASE}/tooth_3d_semantic/eval/holdout_ensemble_ras_idassigned"

EMPTY_GT = {"ToothFairy2P_113", "ToothFairy2P_136", "ToothFairy2P_356"}
PATHO = {"ToothFairy2F_049", "ToothFairy2P_049", "ToothFairy2P_105"}


def fdi(i):
    # closed form, verified 32/32 vs tooth_fairy2/list/label_map.yaml
    return 10 * (1 + (i - 1) // 8) + ((i - 1) % 8 + 1)


def case_dice(seg, gt):
    d = {}
    for c in np.unique(gt):
        if c == 0:
            continue
        g = gt == c
        p = seg == c
        d[int(c)] = float(2 * np.count_nonzero(p & g) / max(1, p.sum() + g.sum()))
    return d


def mean_of(d):
    return float(np.mean(list(d.values()))) if d else 0.0


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for c in sorted(os.listdir(PRED)):
        if not c.endswith(".nii.gz"):
            continue
        case = c[:-len(".nii.gz")]
        pred = sitk.GetArrayFromImage(sitk.ReadImage(f"{PRED}/{c}"))
        gt = sitk.GetArrayFromImage(sitk.ReadImage(f"{GT}/{case}.nii.gz"))
        new, log = apply_assigner(pred)
        b, a = case_dice(pred, gt), case_dice(new, gt)
        rows.append({"case": case,
                     "n_remaps": max(0, len(log) - 1),
                     "fg_before": round(mean_of(b), 4), "fg_after": round(mean_of(a), 4),
                     "per_class_after": {str(k): round(v, 4) for k, v in a.items()}})
        print(f"[{len(rows)}/72] {case} {rows[-1]['fg_before']} -> {rows[-1]['fg_after']} "
              f"({rows[-1]['n_remaps']} remaps)", flush=True)

    with open(f"{OUT}/per_case_metrics.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    def subset(preds):
        vals = [r["fg_after"] for r in rows
                if r["case"] in preds or
                (preds is None and True)]
        return vals

    n72 = [r["fg_after"] for r in rows]
    n69 = [r["fg_after"] for r in rows if r["case"] not in EMPTY_GT]
    n66 = [r["fg_after"] for r in rows if r["case"] not in EMPTY_GT and r["case"] not in PATHO]
    cls_sum, cls_cnt = np.zeros(33), np.zeros(33)
    for r in rows:
        for c, v in r["per_class_after"].items():
            cls_sum[int(c)] += v
            cls_cnt[int(c)] += 1
    per_class = {str(c): round(float(cls_sum[c] / cls_cnt[c]), 4) if cls_cnt[c] else None
                 for c in range(1, 33)}
    upper = [per_class[str(c)] for c in range(1, 17) if per_class[str(c)] is not None]
    lower = [per_class[str(c)] for c in range(17, 33) if per_class[str(c)] is not None]
    weak_now = {c: per_class[c] for c in ("4", "6", "7", "8", "13", "14",
                                          "15", "16", "23", "32")}
    metrics = {
        "ensemble": "1000 5-fold, checkpoint_final, mirroring+gaussian, + L/R id-assigner",
        "mean_fg_dice_66": round(float(np.mean(n66)), 4),
        "mean_fg_dice_69": round(float(np.mean(n69)), 4),
        "mean_fg_dice_72": round(float(np.mean(n72)), 4),
        "mean_upper_jaw_1_16": round(float(np.mean(upper)), 4),
        "mean_lower_jaw_17_32": round(float(np.mean(lower)), 4),
        "per_class_dice": per_class,
        "per_class_fdi": {str(c): fdi(int(c)) for c in per_class},
        "weak_classes_focus": weak_now,
        "n_remaps_total": int(sum(r["n_remaps"] for r in rows)),
    }
    with open(f"{OUT}/holdout_metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)
    print(json.dumps({k: metrics[k] for k in
                      ("mean_fg_dice_66", "mean_fg_dice_69", "mean_fg_dice_72",
                       "mean_upper_jaw_1_16", "mean_lower_jaw_17_32")}, indent=1))
    print("weak classes (after assigner):",
          {k: f"{k}=FDI{fdi(int(k))}:{v}" for k, v in weak_now.items()})


if __name__ == "__main__":
    main()

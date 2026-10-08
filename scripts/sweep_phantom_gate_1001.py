#!/usr/bin/env python3
"""Wave 1: sweep phantom probability-gating rules offline (CPU).

Uses the per-component confidence stats exported by export_probs_1001.py.
For each candidate rule (threshold on component mean/min maxprob, optionally
combined with a size cap), removes matching components from the prediction
and recomputes 66-case foreground dice. Gate: no case may regress >0.005.

Dice is recomputed exactly per case from seg + GT (vectorized), so no
approximation. Prints the Pareto of (cases_fixed, mean dice, worst delta).

Usage:
  python scripts/sweep_phantom_gate_1001.py          # full sweep
  python scripts/sweep_phantom_gate_1001.py --stat margin
"""
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndi

BASE = Path("/home/share/clr/share/work/CBCT/nnUNet")
GT_DIR = BASE / "tooth_3d_semantic/preprocessed/Dataset1000_ToothSemanticRAS/gt_segmentations"
PRED = BASE / "tooth_3d_semantic/eval/holdout_ensemble_noxmirror/predictions"
PROBS = BASE / "tooth_3d_semantic/eval/holdout_ensemble_noxmirror_probs"
EMPTY_GT = {"ToothFairy2P_113", "ToothFairy2P_136", "ToothFairy2P_356"}
PATHO = {"ToothFairy2F_049", "ToothFairy2P_049", "ToothFairy2P_105"}
N_CLASSES = 32


def load_cases():
    cases = sorted(p.name.replace(".nii.gz", "") for p in PRED.glob("*.nii.gz"))
    return [c for c in cases if c not in EMPTY_GT and c not in PATHO]


def dice_table(seg, gt):
    """Per-class dice dict for GT-present classes."""
    out = {}
    for c in range(1, N_CLASSES + 1):
        g = gt == c
        if not g.any():
            continue
        p = seg == c
        s = p.sum() + g.sum()
        out[c] = 2.0 * np.count_nonzero(p & g) / s if s else 0.0
    return out


def sweep_exact(stat="mean_maxprob", thresholds=None, size_cap=None, verbose=True):
    """Exact sweep: per tau, remove matching comps and recompute dice from
    the modified seg. Slower but unambiguous."""
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0.10, 0.96, 0.05), 2)
    cases = load_cases()
    # preload
    data = {}
    for c in cases:
        with open(PROBS / "comps" / f"{c}.json") as f:
            comps = json.load(f)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(GT_DIR / f"{c}.nii.gz")))
        seg = sitk.GetArrayFromImage(sitk.ReadImage(str(PRED / f"{c}.nii.gz")))
        lab, _ = ndi.label(seg > 0)
        eligible = []
        for comp in comps:
            if size_cap is not None and comp["n_vox"] > size_cap:
                continue
            eligible.append((comp["comp"], comp[stat]))
        data[c] = (seg, gt, lab, eligible, dice_table(seg, gt))

    out = []
    for tau in thresholds:
        case_dice_new, deltas = {}, {}
        for c in cases:
            seg, gt, lab, eligible, base = data[c]
            drop_ids = [cid for cid, t in eligible if t < tau]
            nd_ = 0.0
            if drop_ids:
                s2 = seg.copy()
                mask = np.isin(lab, drop_ids)
                s2[mask] = 0
                new = dice_table(s2, gt)
            else:
                new = base
            bd = float(np.mean(list(base.values())))
            ndv = float(np.mean(list(new.values())))
            case_dice_new[c] = ndv
            deltas[c] = ndv - bd
        mean_new = float(np.mean(list(case_dice_new.values())))
        worst = min(deltas.values())
        n_fixed = sum(1 for v in deltas.values() if v > 0.005)
        n_hurt = sum(1 for v in deltas.values() if v < -0.005)
        out.append({"tau": float(tau), "mean66": round(mean_new, 4),
                    "worst_delta": round(worst, 4), "n_fixed": n_fixed,
                    "n_hurt": n_hurt, "deltas": {k: round(v, 4) for k, v in deltas.items()}})
        if verbose:
            print(f"tau={tau:.2f}  mean66={mean_new:.4f}  worst={worst:+.4f}  "
                  f"fixed(>0.005)={n_fixed}  hurt(<-0.005)={n_hurt}")
    return out


if __name__ == "__main__":
    stat = "mean_margin" if "--stat" in sys.argv and "margin" in sys.argv else "mean_maxprob"
    print(f"# sweep stat={stat} (66-case scope; baseline mean66=0.9281)")
    res = sweep_exact(stat=stat)
    ok = [r for r in res if r["worst_delta"] >= -0.005]
    if ok:
        best = max(ok, key=lambda r: r["mean66"])
        print(f"\nBEST zero-regression gate: tau={best['tau']} mean66={best['mean66']} "
              f"worst={best['worst_delta']} fixed={best['n_fixed']} hurt={best['n_hurt']}")

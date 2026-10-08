#!/usr/bin/env python
"""Residual-error 3-family decomposition on Dataset1001 holdout predictions.

Repeats the 1000-era confusion-matrix decomposition (Task 8 leftover) so the
1001 residual structure can be compared against the 1000 baseline:
  missed  : GT foreground -> pred background
  phantom : GT background -> pred foreground
  idwrong : GT foreground -> pred foreground, different class

Instance level: GT connected components matched to pred by majority voxel
overlap -> OK / ID_WRONG / MISSING, plus phantom pred components.

66-case strict scope: 72 holdout minus 3 empty-GT minus 3 pathological.
Voxel-pair keys use int32 (g*33+s) -- uint8 overflows (documented trap).

Output: tooth_3d_semantic/logs/err_decompose_1001.json + per-case jsonl.
"""
import json
import time
from collections import Counter

import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk

BASE = "/home/share/clr/share/work/CBCT/nnUNet"
GT = f"{BASE}/tooth_3d_semantic/preprocessed/Dataset1000_ToothSemanticRAS/gt_segmentations"
PRED = f"{BASE}/tooth_3d_semantic/eval/holdout_ensemble_noxmirror/predictions"
OUT_JSON = f"{BASE}/tooth_3d_semantic/logs/err_decompose_1001.json"
OUT_JSONL = f"{BASE}/tooth_3d_semantic/logs/err_decompose_1001_cases.jsonl"

EMPTY_GT = {"ToothFairy2P_113", "ToothFairy2P_136", "ToothFairy2P_356"}
PATHO = {"ToothFairy2F_049", "ToothFairy2P_049", "ToothFairy2P_105"}
# P_036..P_323 tail from per_case_metrics.jsonl fg_dice < 0.85 (66-case scope)
TAIL = {
    "ToothFairy2P_036", "ToothFairy2P_341", "ToothFairy2P_067",
    "ToothFairy2P_368", "ToothFairy2P_527", "ToothFairy2P_378",
    "ToothFairy2P_342", "ToothFairy2P_413", "ToothFairy2P_090",
    "ToothFairy2P_134", "ToothFairy2P_323",
}

MIN_COMP_VOX = 20  # match assigner's connectivity threshold


def is_twin(a, b):
    """±8 lower-arch L/R twin pair (FDI 3x/4x), the mirror-DA failure family."""
    return 17 <= a <= 32 and 17 <= b <= 32 and abs(a - b) == 8


def decompose(case):
    gt = sitk.GetArrayFromImage(sitk.ReadImage(f"{GT}/{case}.nii.gz")).astype(np.int32)
    pred = sitk.GetArrayFromImage(sitk.ReadImage(f"{PRED}/{case}.nii.gz")).astype(np.int32)
    gfg, pfg = gt > 0, pred > 0
    res = {
        "case": case,
        "gt_fg_vox": int(gfg.sum()),
        "pred_fg_vox": int(pfg.sum()),
        "missed": int((gfg & ~pfg).sum()),
        "phantom": int((~gfg & pfg).sum()),
        "idwrong_vox": 0,
        "pair_counter": {},          # "g->p" family buckets
        "pair_examples": {},         # exact class pairs, top only
        "twin8_vox": 0,              # ±8 subset of idwrong
        "drift_vox": 0,              # |g-p| in {1} adjacent-tooth drift
        "inst_ok": 0, "inst_idwrong": 0, "inst_missing": 0,
        "phantom_inst": 0,           # pred comps with no GT support
        "phantom_inst_vox": 0,
    }
    pairs = Counter()
    # ---- ID-wrong family: both foreground, different label ----
    both = gfg & pfg
    if both.any():
        key = gt[both] * 33 + pred[both]          # int32 arrays: no overflow
        same = key % 33 == key // 33
        diff_keys, counts = np.unique(key[~same], return_counts=True)
        for k, c in zip(diff_keys.tolist(), counts.tolist()):
            g, p = k // 33, k % 33
            res["idwrong_vox"] += c
            if is_twin(g, p):
                res["twin8_vox"] += c
                pairs[f"twin {g}->{p}"] += c
            elif abs(g - p) == 1:
                res["drift_vox"] += c
                pairs[f"drift {g}->{p}"] += c
            else:
                pairs[f"other {g}->{p}"] += c
    top = pairs.most_common(8)
    res["pair_examples"] = {k: v for k, v in top}
    # family buckets over exact pairs
    fam = Counter()
    for k, v in pairs.items():
        fam[k.split()[0]] += v
    res["pair_counter"] = dict(fam)

    # ---- instance level: GT comps matched by majority overlap ----
    lab_gt, n_gt = ndi.label(gfg)
    lab_pd, n_pd = ndi.label(pfg)
    for i in range(1, n_gt + 1):
        m = lab_gt == i
        n = int(m.sum())
        if n < MIN_COMP_VOX:
            continue
        vals, cnts = np.unique(pred[m & pfg], return_counts=True)
        if len(vals) == 0:
            res["inst_missing"] += 1
        elif vals[np.argmax(cnts)] != int(round(np.unique(gt[m])[0])):
            res["inst_idwrong"] += 1
        else:
            res["inst_ok"] += 1
    for j in range(1, n_pd + 1):
        m = lab_pd == j
        n = int(m.sum())
        if n < MIN_COMP_VOX:
            continue
        ov = gt[m & gfg]
        supported = len(ov) > 0 and np.bincount(ov).argmax() > 0
        if not supported:
            res["phantom_inst"] += 1
            res["phantom_inst_vox"] += n
    return res


def main():
    import os
    cases = sorted(f[:-7] for f in os.listdir(PRED) if f.endswith(".nii.gz"))
    scope66 = [c for c in cases if c not in EMPTY_GT and c not in PATHO]
    t0 = time.time()
    rows = []
    for i, c in enumerate(scope66, 1):
        rows.append(decompose(c))
        print(f"[{i:2d}/{len(scope66)}] {c} idwrong={rows[-1]['idwrong_vox']:>8} "
              f"missed={rows[-1]['missed']:>8} phantom={rows[-1]['phantom']:>8} "
              f"tw={rows[-1]['twin8_vox']:>7} dr={rows[-1]['drift_vox']:>7} "
              f"inst ok/idw/miss={rows[-1]['inst_ok']}/{rows[-1]['inst_idwrong']}/{rows[-1]['inst_missing']} "
              f"ph_inst={rows[-1]['phantom_inst']}", flush=True)
    agg = {k: sum(r[k] for r in rows) for k in
           ("gt_fg_vox", "pred_fg_vox", "missed", "phantom", "idwrong_vox",
            "twin8_vox", "drift_vox", "inst_ok", "inst_idwrong", "inst_missing",
            "phantom_inst", "phantom_inst_vox")}
    total_err = agg["missed"] + agg["phantom"] + agg["idwrong_vox"]
    summary = {
        "n_cases": len(rows), "scope": "66-case strict",
        "aggregate": agg,
        "total_err_vox": total_err,
        "family_frac": {
            "missed": agg["missed"] / total_err,
            "phantom": agg["phantom"] / total_err,
            "idwrong": agg["idwrong_vox"] / total_err,
        },
        "idwrong_sub": {
            "twin8": agg["twin8_vox"], "drift1": agg["drift_vox"],
            "other": agg["idwrong_vox"] - agg["twin8_vox"] - agg["drift_vox"],
        },
        "tail_rows": {r["case"]: {k: r[k] for k in
                     ("fg_missed",) if False} for r in []},  # placeholder
        "elapsed_min": (time.time() - t0) / 60,
    }
    tail = {r["case"]: {k: r[k] for k in
            ("missed", "phantom", "idwrong_vox", "twin8_vox", "drift_vox",
             "inst_missing", "inst_idwrong", "phantom_inst")} for r in rows
            if r["case"] in TAIL}
    summary["tail_rows"] = tail
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    with open(OUT_JSONL, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "tail_rows"},
                     indent=2))


if __name__ == "__main__":
    main()

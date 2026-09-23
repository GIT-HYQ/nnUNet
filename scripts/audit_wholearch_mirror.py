#!/usr/bin/env python3
"""Whole-arch L/R mirrored-GT detector (blind spot of the 08-27 midpoint
audit: a fully mirrored label set is still symmetric, so midpoint checks
pass). Direct check: each tooth ID must sit on its anatomical side in the
true-RAS frame (right = high x). Audit only - no label modification."""
import glob
import json
import os
import sys

import numpy as np
import SimpleITK as sitk
import yaml
from scipy import ndimage

sys.path.insert(0, "/home/share/clr/share/work/CBCT/nnUNet")
from inference.id_assigner import SIDE_LUT, MIN_COMP_VOX, MARGIN_MIN_VOX, MARGIN_FRAC

RAW = "/home/share/clr/share/work/CBCT/nnUNet/tooth_3d_semantic/raw/Dataset1000_ToothSemanticRAS/labelsTr"
SPLIT_YAML = "/home/share/clr/share/data/CBCT/tooth_fairy2/list/tooth_split_k5_seed0.yaml"
OUT = "/home/share/clr/share/work/CBCT/nnUNet/tooth_3d_semantic/logs/wholearch_audit_all480.json"


def audit_case(seg):
    lab, n = ndimage.label(seg > 0)
    sizes = np.bincount(lab.ravel())
    cxs, ids = [], []
    for i in range(1, n + 1):
        if sizes[i] < MIN_COMP_VOX:
            continue
        m = lab == i
        vals, counts = np.unique(seg[m], return_counts=True)
        dom = int(vals[np.argmax(counts)])
        if dom == 0:
            continue
        _, _, cx = ndimage.center_of_mass(m)
        cxs.append(float(cx))
        ids.append(dom)
    if not cxs:
        return {"status": "EMPTY"}
    mid = (min(cxs) + max(cxs)) / 2.0
    margin = max(MARGIN_MIN_VOX, MARGIN_FRAC * (max(cxs) - min(cxs)))
    det = [(cx, i) for cx, i in zip(cxs, ids) if abs(cx - mid) > margin]
    if len(det) < 6:
        return {"status": "INSUFFICIENT", "n_det": len(det)}
    wrong = sum(1 for cx, i in det if SIDE_LUT[i] != ("R" if cx > mid else "L"))
    score = wrong / len(det)
    status = "MIRROR" if score >= 0.6 else "SUSPECT" if score >= 0.3 else "OK"
    return {"status": status, "score": round(score, 3), "n_det": len(det)}


def main():
    with open(SPLIT_YAML) as f:
        split = yaml.safe_load(f)
    holdout = set(split["holdout_test"])
    files = sorted(glob.glob(f"{RAW}/*.nii.gz"))
    assert len(files) == 480, f"expected 480 GTs, got {len(files)}"
    results = []
    for i, fp in enumerate(files, 1):
        case = os.path.basename(fp)[:-len(".nii.gz")]
        seg = sitk.GetArrayFromImage(sitk.ReadImage(fp))
        r = audit_case(seg)
        r["case"] = case
        r["split"] = "holdout" if case in holdout else "train"
        results.append(r)
        if r["status"] in ("MIRROR", "SUSPECT"):
            print(f"FLAG {case} [{r['split']}] {r}", flush=True)
        if i % 20 == 0:
            print(f"[{i}/480]", flush=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    from collections import Counter
    cnt = Counter((r["split"], r["status"]) for r in results)
    print(json.dumps({f"{k[0]}/{k[1]}": v for k, v in sorted(cnt.items())}, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Deterministic left-right consistency assigner for 1-32 tooth ID maps.

The model occasionally assigns a tooth its left-right homolog's ID
(arch-level +/-8 swap) while segmenting its position correctly. This module
re-maps such IDs from the geometric side of each connected component in the
true-RAS array frame (patient right = high x, array axis 2).

Side convention (verified on tooth_fairy2_m, Slicer-confirmed, see
memory orientation-investigation):
    1-8  = FDI 11-18 upper right  -> patient RIGHT (high x)
    9-16 = FDI 21-28 upper left   -> patient LEFT  (low x)
    17-24 = FDI 31-38 lower left  -> patient LEFT  (low x)
    25-32 = FDI 41-48 lower right -> patient RIGHT (high x)
(FDI lower quadrants are reversed: 3x = left, 4x = right.)
"""
import argparse
import json

import numpy as np
from scipy import ndimage

MIN_COMP_VOX = 20       # ignore specks
MARGIN_MIN_VOX = 10     # ~3 mm at 0.3 mm/voxel
MARGIN_FRAC = 0.08      # of arch x-extent


def build_side_lut():
    lut = {}
    for i in range(1, 9):
        lut[i] = "R"
    for i in range(9, 17):
        lut[i] = "L"
    for i in range(17, 25):
        lut[i] = "L"
    for i in range(25, 33):
        lut[i] = "R"
    return lut


SIDE_LUT = build_side_lut()


def swap_lr(i):
    if not 1 <= i <= 32:
        raise ValueError(f"label out of range: {i}")
    return i + 8 if (i <= 8 or 17 <= i <= 24) else i - 8


def decide_remaps(components):
    """components: [{"id": int, "cx": float}, ...].
    Returns (per-component (old, new) pairs, {"midline": float, "margin": float})."""
    cxs = [c["cx"] for c in components]
    mid = (min(cxs) + max(cxs)) / 2.0
    margin = max(MARGIN_MIN_VOX, MARGIN_FRAC * (max(cxs) - min(cxs)))
    remaps = []
    for c in components:
        new = c["id"]
        if c["cx"] > mid + margin and SIDE_LUT[c["id"]] == "L":
            new = swap_lr(c["id"])
        elif c["cx"] < mid - margin and SIDE_LUT[c["id"]] == "R":
            new = swap_lr(c["id"])
        remaps.append((c["id"], new))
    return remaps, {"midline": mid, "margin": margin}


def _components(seg):
    """Yield (comp_index, dominant_label, cx_vox, n_vox) for seg>0 components."""
    lab, n = ndimage.label(seg > 0)
    sizes = np.bincount(lab.ravel())
    out = []
    for i in range(1, n + 1):
        if sizes[i] < MIN_COMP_VOX:
            continue
        m = lab == i
        vals, counts = np.unique(seg[m], return_counts=True)
        dom = int(vals[np.argmax(counts)])
        if dom == 0:
            continue
        _, _, cx = ndimage.center_of_mass(m)
        out.append((i, dom, float(cx), int(sizes[i])))
    return lab, out


def apply_assigner(seg):
    """seg: 3D uint8 (z, y, x), labels 0..32 -> (remapped seg, log).
    Only the dominant label of a re-mapped component is rewritten; other
    labels inside merged components are left untouched."""
    lab, comps = _components(seg)
    out = seg.copy()
    log = []
    if not comps:
        return out, log
    remaps, meta = decide_remaps([{"id": d, "cx": cx} for _, d, cx, _ in comps])
    log.append({"midline": round(meta["midline"], 1),
                "margin": round(meta["margin"], 1)})
    for (i, old, cx, nvox), (_, new) in zip(comps, remaps):
        if old != new:
            out[(lab == i) & (seg == old)] = new
            log.append({"comp": i, "id_before": old, "id_after": new,
                        "cx": round(cx, 1), "n_vox": nvox})
    return out, log


def assign_file(input_path, output_path, log_path=None):
    import SimpleITK as sitk
    img = sitk.ReadImage(str(input_path))
    arr = sitk.GetArrayFromImage(img)
    out, log = apply_assigner(arr)
    out_img = sitk.GetImageFromArray(out.astype(np.uint8))
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, str(output_path))
    n = max(0, len(log) - 1)
    if log_path is not None:
        with open(log_path, "w") as f:
            json.dump(log, f, indent=1)
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="1-32 label map .nii.gz")
    p.add_argument("--output", required=True, help="output .nii.gz")
    p.add_argument("--log", default=None, help="remap log .json path")
    a = p.parse_args()
    n = assign_file(a.input, a.output, a.log)
    print(f"re-mapped components: {n}")


if __name__ == "__main__":
    main()

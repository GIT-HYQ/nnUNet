#!/usr/bin/env python3
"""Convert nnUNet tooth prediction labels (sequential 1-32) back to FDI numbering.

Dataset999_ToothSemantic outputs sequential class ids 1-32 (see
dataset_conversion/tooth_3d.py: auto_build_label_map). Clinical display and
delivery use FDI numbering: 11-18 (upper right), 21-28 (upper left),
31-38 (lower left), 41-48 (lower right). FDI quadrants run clockwise from
the front, so the lower arch is 3x = left, 4x = right.

The authoritative map is label_map.yaml (train_to_raw section); the closed
form fdi(i) = 10*(1+(i-1)//8) + ((i-1)%8+1) is cross-checked against it at
startup (they coincide because this dataset's GT covers all 32 teeth).

Usage:
    python inference/tooth_relabel_fdi.py --input pred.nii.gz --output pred_fdi.nii.gz
    python inference/tooth_relabel_fdi.py --input_dir preds/ --output_dir preds_fdi/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

DEFAULT_LABEL_MAP = "/home/share/clr/share/data/CBCT/tooth_fairy2/list/label_map.yaml"


def fdi_from_seq(i: int) -> int:
    if not 1 <= i <= 32:
        raise ValueError(f"sequential id out of range: {i}")
    return 10 * (1 + (i - 1) // 8) + ((i - 1) % 8 + 1)


def build_lut(label_map_path: str) -> np.ndarray:
    import yaml

    with open(label_map_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    t2r = data.get("train_to_raw", data)

    lut = np.zeros(33, dtype=np.uint8)
    for train_id in range(1, 33):
        raws = t2r.get(train_id, t2r.get(str(train_id)))
        if raws is None or len(raws) != 1:
            sys.exit(f"ERROR: train id {train_id} does not map to exactly one FDI id in {label_map_path}: {raws}")
        lut[train_id] = int(raws[0])

    for i in range(1, 33):
        if lut[i] != fdi_from_seq(i):
            sys.exit(f"ERROR: label_map.yaml disagrees with closed form at id {i}: {lut[i]} vs {fdi_from_seq(i)}")
    return lut


def convert_file(fin: str, fout: str, lut: np.ndarray) -> int:
    img = sitk.ReadImage(fin)
    arr = sitk.GetArrayFromImage(img).astype(np.int32)
    if arr.size and (int(arr.min()) < 0 or int(arr.max()) > 32):
        sys.exit(f"ERROR: {fin} contains labels outside 0-32: {sorted(np.unique(arr).tolist())}")

    before = np.bincount(arr.ravel(), minlength=33)
    out = sitk.GetImageFromArray(lut[arr])
    out.CopyInformation(img)
    sitk.WriteImage(out, fout, useCompression=True)

    after = np.bincount(sitk.GetArrayFromImage(sitk.ReadImage(fout)).ravel(), minlength=49)
    for i in range(33):
        if before[i] != after[lut[i]]:
            sys.exit(f"ERROR: voxel count mismatch for label {i} -> {int(lut[i])} in {fout}")
    if img.GetSize() != out.GetSize() or tuple(img.GetSpacing()) != tuple(out.GetSpacing()):
        sys.exit(f"ERROR: header mismatch in {fout}")
    return int((arr > 0).sum())


def main():
    p = argparse.ArgumentParser(description="Relabel tooth prediction maps from sequential 1-32 to FDI")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="single prediction .nii.gz with labels 1-32")
    src.add_argument("--input_dir", help="directory of prediction .nii.gz files")
    dst = p.add_mutually_exclusive_group(required=True)
    dst.add_argument("--output", help="output .nii.gz with FDI labels")
    dst.add_argument("--output_dir", help="output directory (file names preserved)")
    p.add_argument("--label_map", default=DEFAULT_LABEL_MAP,
                   help="label_map.yaml containing train_to_raw (default: %(default)s)")
    args = p.parse_args()

    lut = build_lut(args.label_map)
    print(f"Label map: {args.label_map} (closed-form cross-check OK)")

    if args.input:
        pairs = [(args.input, args.output)]
    else:
        ins = sorted(Path(args.input_dir).glob("*.nii.gz"))
        if not ins:
            sys.exit(f"ERROR: no .nii.gz files in {args.input_dir}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        pairs = [(str(x), str(Path(args.output_dir) / x.name)) for x in ins]

    for fin, fout in pairs:
        fg = convert_file(fin, fout, lut)
        print(f"{Path(fin).name}: {fg} fg voxels -> {fout}")
    print(f"Done: {len(pairs)} file(s)")


if __name__ == "__main__":
    main()

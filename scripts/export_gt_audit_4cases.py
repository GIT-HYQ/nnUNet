#!/usr/bin/env python3
"""Export viewer-ready bundles for the Wave 0.2 GT audit (4 tail cases).

Per case writes eval/gt_audit_4cases/<case>/:
  image.nii.gz         raw CBCT (Dataset1000 imagesTr, original grid)
  gt.nii.gz            ground truth labels
  pred.nii.gz          Dataset1001 5-fold ensemble prediction
  pred_only_mask.nii.gz  1 = model predicted tooth, GT says background (phantom)
  gt_only_mask.nii.gz    1 = GT tooth, model predicted background (missed)
P_368 additionally gets a zoom crop around the two off-arch small blobs.

All 5 files share the original image grid -> drag into 3D Slicer / ITK-SNAP.
"""
from pathlib import Path

import numpy as np
import SimpleITK as sitk

BASE = Path("/home/share/clr/share/work/CBCT/nnUNet")
IMG = BASE / "tooth_3d_semantic/raw/Dataset1000_ToothSemanticRAS/imagesTr"
GT = BASE / "tooth_3d_semantic/preprocessed/Dataset1000_ToothSemanticRAS/gt_segmentations"
PRED = BASE / "tooth_3d_semantic/eval/holdout_ensemble_noxmirror/predictions"
OUT = BASE / "tooth_3d_semantic/eval/gt_audit_4cases"
CASES = ("ToothFairy2P_036", "ToothFairy2P_368", "ToothFairy2P_378", "ToothFairy2P_527")
ZOOM = {"ToothFairy2P_368": [(123, 85, 50), (126, 45, 40)]}  # blob centers (z,y,x)
MARGIN = 40


def save(arr, img, path):
    o = sitk.GetImageFromArray(arr)
    o.CopyInformation(img)
    sitk.WriteImage(o, str(path))


for case in CASES:
    d = OUT / case
    d.mkdir(parents=True, exist_ok=True)
    img_o = sitk.ReadImage(str(IMG / f"{case}_0000.nii.gz"))
    image = sitk.GetArrayFromImage(img_o)
    gt = sitk.GetArrayFromImage(sitk.ReadImage(str(GT / f"{case}.nii.gz")))
    pred = sitk.GetArrayFromImage(sitk.ReadImage(str(PRED / f"{case}.nii.gz")))
    save(image, img_o, d / "image.nii.gz")
    save(gt, img_o, d / "gt.nii.gz")
    save(pred, img_o, d / "pred.nii.gz")
    save(((pred > 0) & (gt == 0)).astype(np.uint8), img_o, d / "pred_only_mask.nii.gz")
    save(((gt > 0) & (pred == 0)).astype(np.uint8), img_o, d / "gt_only_mask.nii.gz")
    print(f"{case}: exported 5 files")

    if case in ZOOM:
        cd = d / "zoom_crop"
        cd.mkdir(exist_ok=True)
        z0 = min(c[0] for c in ZOOM[case]) - MARGIN
        z1 = max(c[0] for c in ZOOM[case]) + MARGIN
        y0 = min(c[1] for c in ZOOM[case]) - MARGIN
        y1 = max(c[1] for c in ZOOM[case]) + MARGIN
        x0 = min(c[2] for c in ZOOM[case]) - MARGIN
        x1 = max(c[2] for c in ZOOM[case]) + MARGIN
        z0, y0, x0 = max(z0, 0), max(y0, 0), max(x0, 0)
        z1 = min(z1, image.shape[0]); y1 = min(y1, image.shape[1]); x1 = min(x1, image.shape[2])
        sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
        crop_o = sitk.GetImageFromArray(image[sl])
        # inherit spacing/direction from the full image; shift origin so the
        # crop lands at the correct world position
        idx = sitk.Image(img_o.GetSize(), sitk.sitkUInt8)
        idx.CopyInformation(img_o)
        pos = idx.TransformContinuousIndexToPhysicalPoint((float(x0), float(y0), float(z0)))
        crop_o.SetSpacing(img_o.GetSpacing())
        crop_o.SetDirection(img_o.GetDirection())
        crop_o.SetOrigin(pos)
        sitk.WriteImage(crop_o, str(cd / "image_crop.nii.gz"))
        for name, arr in (("gt", gt), ("pred", pred)):
            o = sitk.GetImageFromArray(arr[sl])
            o.CopyInformation(crop_o)
            sitk.WriteImage(o, str(cd / f"{name}_crop.nii.gz"))
        print(f"  zoom crop {sl} -> {cd}")
print("done ->", OUT)

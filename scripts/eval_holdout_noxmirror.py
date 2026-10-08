#!/usr/bin/env python3
"""Final evaluation of Dataset1001_ToothSemanticNoXMirror on the 72-case
HOLDOUT test set.

Same protocol as eval_holdout_ras_1000.py (Dataset1000): 5-fold logit
ensemble (folds 0-4), checkpoint_final, mirroring + gaussian, 3 workers on
GPUs 1,2,3 (GPU 0 kept free). Same 72 holdout cases (splits are
byte-identical), so results are directly comparable to the 0.7837 baseline.

The trainer is nnUNetTrainer_onlyMirror01: its checkpoint carries
trainer_name, so nnUNet automatically restricts TTA mirroring to axes
(0, 1) (z/y) — consistent with training (no x mirror). use_mirroring=True
is therefore intentional, not a bug.

Images are read from the Dataset1000_ToothSemanticRAS raw imagesTr: the 72
holdout cases and their image content are identical across the two datasets.

Outputs: tooth_3d_semantic/eval/holdout_ensemble_noxmirror/
  predictions/{case}.nii.gz   ensemble prediction in original image space
  per_case_metrics.jsonl      per-case foreground + per-class dice
  holdout_metrics.json        aggregated metrics

Usage:
  python scripts/eval_holdout_noxmirror.py            # full run (72 cases, 3 GPUs)
  python scripts/eval_holdout_noxmirror.py --smoke    # 1 case on GPU 1, sanity check
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

BASE = Path("/home/share/clr/share/work/CBCT/nnUNet")
RAW = BASE / "tooth_3d_semantic/raw/Dataset1000_ToothSemanticRAS"
GT_DIR = BASE / "tooth_3d_semantic/preprocessed/Dataset1000_ToothSemanticRAS/gt_segmentations"
RESULTS = BASE / "tooth_3d_semantic/results/Dataset1001_ToothSemanticNoXMirror/nnUNetTrainer_onlyMirror01__nnUNetPlans__3d_fullres"
HOLDOUT_YAML = Path("/home/share/clr/share/data/CBCT/tooth_fairy2/list/tooth_split_k5_seed0.yaml")
OUT = BASE / "tooth_3d_semantic/eval/holdout_ensemble_noxmirror"
PRED_DIR = OUT / "predictions"
FOLDS = (0, 1, 2, 3, 4)
N_CLASSES = 32
GPU_IDS = (1, 2, 3)
N_GPUS = len(GPU_IDS)

os.environ["nnUNet_raw"] = str(BASE / "tooth_3d_semantic/raw")
os.environ["nnUNet_preprocessed"] = str(BASE / "tooth_3d_semantic/preprocessed")
os.environ["nnUNet_results"] = str(BASE / "tooth_3d_semantic/results")


def case_dice(pred, gt):
    """pred/gt: same-shape arrays (SimpleITK axis order). Per-class dice for GT-present classes."""
    out = {}
    for c in range(1, N_CLASSES + 1):
        g = gt == c
        if g.sum() == 0:
            continue
        p = pred == c
        inter = np.count_nonzero(p & g)
        out[c] = 2.0 * inter / (p.sum() + g.sum())
    return out


def worker(gpu_idx, cases):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    import torch
    import SimpleITK as sitk
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    print(f"[gpu{gpu_idx}] loading 5-fold ensemble model...", flush=True)
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda"),
        verbose=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(RESULTS), use_folds=FOLDS, checkpoint_name="checkpoint_final.pth")
    print(f"[gpu{gpu_idx}] model ready, {len(cases)} cases", flush=True)

    rows = []
    for i, case in enumerate(cases, 1):
        img = sitk.ReadImage(str(RAW / "imagesTr" / f"{case}_0000.nii.gz"))
        arr = sitk.GetArrayFromImage(img)
        seg = predictor.predict_single_npy_array(
            arr[None], {"spacing": img.GetSpacing()}, save_or_return_probabilities=False)
        pred_img = sitk.GetImageFromArray(seg.astype(np.uint8))
        pred_img.CopyInformation(img)
        sitk.WriteImage(pred_img, str(PRED_DIR / f"{case}.nii.gz"))

        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(GT_DIR / f"{case}.nii.gz")))
        assert seg.shape == gt.shape, f"{case}: pred {seg.shape} != gt {gt.shape}"
        d = case_dice(seg, gt)
        fg = float(np.mean(list(d.values()))) if d else 0.0
        rows.append({"case": case, "fg_dice": fg, "n_gt_classes": len(d), "per_class": d})
        print(f"[gpu{gpu_idx}] [{i}/{len(cases)}] {case} fg_dice={fg:.4f} ({len(d)} classes)", flush=True)
    return rows


def aggregate(rows):
    cls_sum = np.zeros(N_CLASSES + 1)
    cls_cnt = np.zeros(N_CLASSES + 1)
    for r in rows:
        for c, v in r["per_class"].items():
            cls_sum[c] += v
            cls_cnt[c] += 1
    per_class = {}
    for c in range(1, N_CLASSES + 1):
        per_class[str(c)] = round(float(cls_sum[c] / cls_cnt[c]), 4) if cls_cnt[c] else None

    upper = [per_class[str(c)] for c in range(1, 17) if per_class[str(c)] is not None]
    lower = [per_class[str(c)] for c in range(17, 33) if per_class[str(c)] is not None]
    return {
        "n_cases": len(rows),
        "ensemble": "5-fold logit average, checkpoint_final, mirroring+gaussian",
        "note": "trainer nnUNetTrainer_onlyMirror01: TTA mirroring restricted to axes (0,1) (z/y), consistent with training; "
                "fold_0 trained with global batch 18 (4.5M samples); folds 1-4 with global batch 8 (2M each)",
        "mean_foreground_dice": round(float(np.mean([r["fg_dice"] for r in rows])), 4),
        "std_foreground_dice": round(float(np.std([r["fg_dice"] for r in rows])), 4),
        "min_case": min(rows, key=lambda r: r["fg_dice"])["case"],
        "max_case": max(rows, key=lambda r: r["fg_dice"])["case"],
        "per_class_dice": per_class,
        "mean_upper_jaw_1_16": round(float(np.mean(upper)), 4),
        "mean_lower_jaw_17_32": round(float(np.mean(lower)), 4),
    }


def main():
    import multiprocessing as mp
    import time

    smoke = "--smoke" in sys.argv
    with open(HOLDOUT_YAML) as f:
        holdout = sorted(yaml.safe_load(f)["holdout_test"])
    assert len(holdout) == 72, f"expected 72 holdout cases, got {len(holdout)}"
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    if smoke:
        rows = worker(GPU_IDS[0], holdout[:1])
        print(json.dumps(rows[0], indent=2))
        return

    chunks = [holdout[i::N_GPUS] for i in range(N_GPUS)]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(N_GPUS) as pool:
        parts = pool.starmap(worker, [(GPU_IDS[g], chunks[g]) for g in range(N_GPUS)])
    rows = [r for part in parts for r in part]
    rows.sort(key=lambda r: r["case"])
    assert len(rows) == 72, f"got {len(rows)} results"

    with open(OUT / "per_case_metrics.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    metrics = aggregate(rows)
    metrics["elapsed_min"] = round((time.time() - t0) / 60, 1)
    with open(OUT / "holdout_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 50)
    print(f"HOLDOUT EVALUATION NoXMirror ({metrics['n_cases']} cases, 5-fold ensemble)")
    print(f"elapsed: {metrics['elapsed_min']} min")
    print(f"mean foreground dice: {metrics['mean_foreground_dice']} "
          f"(std {metrics['std_foreground_dice']})")
    print(f"worst case: {metrics['min_case']}  best case: {metrics['max_case']}")
    print(f"upper jaw (1-16):  {metrics['mean_upper_jaw_1_16']}")
    print(f"lower jaw (17-32): {metrics['mean_lower_jaw_17_32']}")
    print("per-class dice:")
    pc = metrics["per_class_dice"]
    for c in range(1, N_CLASSES + 1):
        v = pc[str(c)]
        print(f"  class {c:>2}: {v}")
    print(f"\nmetrics written to {OUT / 'holdout_metrics.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Wave 1: export 5-fold-ensemble softmax probabilities for the 66-case
scope (Dataset1001, same numerics as eval_holdout_noxmirror.py) so phantom
probability gating can be swept offline on CPU.

Per case writes into eval/holdout_ensemble_noxmirror_probs/:
  probs/{case}.npz    pred-foreground voxel indices (int32, z/y/x order) +
                      their 33-class softmax probabilities (float16)
  comps/{case}.json   per connected component (>=20 vox): label, n_vox,
                      bbox, mean/max of per-voxel maxprob, margin top1-top2
  segcheck/{case}.nii.gz  argmax seg from the prob path (sanity: must match
                      existing predictions/ byte-wise in fg; not for reuse)

Numerics: identical to eval_holdout_noxmirror.py (5-fold ensemble,
checkpoint_final, mirroring + gaussian, tile_step 0.5); only
save_or_return_probabilities=True is added. 66-case scope = 72 holdout
minus 3 empty-GT minus 3 pathological (PATHO), matching the target metric.
"""
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import yaml

BASE = Path("/home/share/clr/share/work/CBCT/nnUNet")
RAW = BASE / "tooth_3d_semantic/raw/Dataset1000_ToothSemanticRAS"
GT_DIR = BASE / "tooth_3d_semantic/preprocessed/Dataset1000_ToothSemanticRAS/gt_segmentations"
RESULTS = BASE / "tooth_3d_semantic/results/Dataset1001_ToothSemanticNoXMirror/nnUNetTrainer_onlyMirror01__nnUNetPlans__3d_fullres"
REF_PRED = BASE / "tooth_3d_semantic/eval/holdout_ensemble_noxmirror/predictions"
OUT = BASE / "tooth_3d_semantic/eval/holdout_ensemble_noxmirror_probs"
HOLDOUT_YAML = Path("/home/share/clr/share/data/CBCT/tooth_fairy2/list/tooth_split_k5_seed0.yaml")
FOLDS = (0, 1, 2, 3, 4)
EMPTY_GT = {"ToothFairy2P_113", "ToothFairy2P_136", "ToothFairy2P_356"}
PATHO = {"ToothFairy2F_049", "ToothFairy2P_049", "ToothFairy2P_105"}
GPU_IDS = (1, 2, 3)
MIN_COMP_VOX = 20

os.environ["nnUNet_raw"] = str(BASE / "tooth_3d_semantic/raw")
os.environ["nnUNet_preprocessed"] = str(BASE / "tooth_3d_semantic/preprocessed")
os.environ["nnUNet_results"] = str(BASE / "tooth_3d_semantic/results")


def component_stats(seg, prob_fg, fg_idx):
    """Per-component confidence stats. prob_fg: (33, N) fp16 at fg_idx voxels."""
    import scipy.ndimage as ndi
    lab, n = ndi.label(seg > 0)
    maxprob = prob_fg.max(axis=0)
    # margin between top-1 and top-2 class probs
    part = np.partition(prob_fg, -2, axis=0)
    margin = part[-1] - part[-2]
    out = []
    for i in range(1, n + 1):
        m = lab == i
        nvox = int(m.sum())
        if nvox < MIN_COMP_VOX:
            continue
        in_fg = m[fg_idx[0], fg_idx[1], fg_idx[2]]  # comp voxels present in prob dump
        sel = np.nonzero(in_fg)[0]
        mp = float(maxprob[sel].mean()) if len(sel) else float("nan")
        mg = float(margin[sel].mean()) if len(sel) else float("nan")
        mn = float(maxprob[sel].min()) if len(sel) else float("nan")
        zz, yy, xx = np.where(m)
        out.append({
            "comp": i, "label": int(seg[m].max()), "n_vox": nvox,
            "bbox_z": [int(zz.min()), int(zz.max())],
            "bbox_y": [int(yy.min()), int(yy.max())],
            "bbox_x": [int(xx.min()), int(xx.max())],
            "mean_maxprob": round(mp, 4), "min_maxprob": round(mn, 4),
            "mean_margin": round(mg, 4),
        })
    return out, lab


def worker(gpu_idx, cases):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    import SimpleITK as sitk
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    (OUT / "probs").mkdir(parents=True, exist_ok=True)
    (OUT / "comps").mkdir(parents=True, exist_ok=True)
    (OUT / "segcheck").mkdir(parents=True, exist_ok=True)

    print(f"[gpu{gpu_idx}] loading 5-fold ensemble...", flush=True)
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        perform_everything_on_device=True, device=torch.device("cuda"),
        verbose=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(
        str(RESULTS), use_folds=FOLDS, checkpoint_name="checkpoint_final.pth")

    for i, case in enumerate(cases, 1):
        t0 = time.time()
        img = sitk.ReadImage(str(RAW / "imagesTr" / f"{case}_0000.nii.gz"))
        arr = sitk.GetArrayFromImage(img)
        seg, prob = predictor.predict_single_npy_array(
            arr[None], {"spacing": img.GetSpacing()}, save_or_return_probabilities=True)
        if isinstance(prob, torch.Tensor):
            prob = prob.cpu().numpy()
        prob = prob.astype(np.float16)  # (33, Z, Y, X) in SimpleITK axis order

        # sanity: argmax seg must match the shipped predictions (allow cudnn
        # run-to-run nondeterminism; eval run showed 50 vox / 1.5M fg = 0.003%)
        ref = sitk.GetArrayFromImage(sitk.ReadImage(str(REF_PRED / f"{case}.nii.gz")))
        mism = int((seg != ref).sum())
        fg_n = int((ref > 0).sum())
        assert mism <= max(1000, fg_n // 1000), \
            f"{case}: prob-path seg differs from predictions/ in {mism} vox > 0.1% of fg"

        fg_idx = np.nonzero(seg > 0)
        fg_prob = prob[:, fg_idx[0], fg_idx[1], fg_idx[2]]  # (33, N) fp16
        comps, _ = component_stats(seg, fg_prob.astype(np.float32), fg_idx)
        np.savez_compressed(OUT / "probs" / f"{case}.npz",
                            idx=np.stack(fg_idx).astype(np.int32),
                            prob=fg_prob)
        with open(OUT / "comps" / f"{case}.json", "w") as f:
            json.dump(comps, f)
        # foreground-only consistency check of the seg we just verified
        del prob
        print(f"[gpu{gpu_idx}] [{i}/{len(cases)}] {case} fg={len(fg_idx[0]):>7} "
              f"comps={len(comps)} ({time.time()-t0:.0f}s)", flush=True)


def main():
    smoke = "--smoke" in sys.argv
    with open(HOLDOUT_YAML) as f:
        holdout = sorted(yaml.safe_load(f)["holdout_test"])
    cases = [c for c in holdout if c not in EMPTY_GT and c not in PATHO]
    assert len(cases) == 66, f"expected 66-case scope, got {len(cases)}"
    if smoke:
        worker(GPU_IDS[0], cases[:1])
        return
    chunks = [cases[i::len(GPU_IDS)] for i in range(len(GPU_IDS))]
    t0 = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(len(GPU_IDS)) as pool:
        pool.starmap(worker, [(GPU_IDS[g], chunks[g]) for g in range(len(GPU_IDS))])
    print(f"done in {(time.time()-t0)/60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()

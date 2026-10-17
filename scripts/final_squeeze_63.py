#!/usr/bin/env python3
"""Final squeeze before the endgame report: exact in-memory merges on the 63-case
census. Per case computes dice for:
  final  : production 1001 5-fold, checkpoint_final (baseline sanity)
  fb     : mean(final_probs, best_probs)          — 牌3 checkpoint ensemble
  ov     : 5-fold, tile_step_size=0.75            — 牌3 overlap
  a6     : (5*final + ResEncL_f0) / 6             — 牌2 six-model
  c6     : (5*final + combo_f0) / 6               — 牌2 six-model (combo)
  all7   : (5*final + ResEncL_f0 + combo_f0) / 7  — bonus
Sharded: argv[1] = shard idx, argv[2] = n_shards, argv[3] = gpu. Writes jsonl.
"""
import json
import os
import sys

import numpy as np
import SimpleITK as sitk
import torch

BASE = "/home/share/clr/share/work/CBCT/nnUNet"
C63 = f"{BASE}/tooth_3d_semantic/logs/lr_census_63.jsonl"
GT = f"{BASE}/tooth_3d_semantic/preprocessed/Dataset1000_ToothSemanticRAS/gt_segmentations"
RAW = f"{BASE}/tooth_3d_semantic/raw/Dataset1000_ToothSemanticRAS/imagesTr"
RES = f"{BASE}/tooth_3d_semantic/results/Dataset1001_ToothSemanticNoXMirror"
M1001 = f"{RES}/nnUNetTrainer_onlyMirror01__nnUNetPlans__3d_fullres"
MRESENC = f"{RES}/nnUNetTrainer_onlyMirror01__nnUNetResEncUNetLPlans__3d_fullres"
MCOMBO = f"{RES}/nnUNetTrainer_onlyMirror01_gapcut__nnUNetResEncUNetLPlans__3d_fullres"

os.environ.setdefault("nnUNet_raw", f"{BASE}/tooth_3d_semantic/raw")
os.environ.setdefault("nnUNet_preprocessed", f"{BASE}/tooth_3d_semantic/preprocessed")
os.environ.setdefault("nnUNet_results", f"{BASE}/tooth_3d_semantic/results")

sys.path.insert(0, BASE)


def case_dice(seg, gt):
    d = []
    for c in np.unique(gt):
        if c == 0:
            continue
        g = gt == c
        p = seg == c
        d.append(2 * np.count_nonzero(p & g) / max(1, p.sum() + g.sum()))
    return float(np.mean(d)) if d else 0.0


def mkpred(model_dir, folds, ckpt="checkpoint_final.pth", step=0.5):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    p = nnUNetPredictor(tile_step_size=step, use_gaussian=True,
                        use_mirroring=True, perform_everything_on_device=True,
                        device=torch.device("cuda"), verbose=False,
                        allow_tqdm=False)
    p.initialize_from_trained_model_folder(model_dir, use_folds=folds, checkpoint_name=ckpt)
    return p


def probs_of(pred, arr, props):
    _, pb = pred.predict_single_npy_array(arr[None], props, save_or_return_probabilities=True)
    if isinstance(pb, torch.Tensor):
        pb = pb.cpu().numpy()
    return pb.astype(np.float32)


def main():
    shard, n_shards, gpu = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    cases = [json.loads(l)["case"] for l in open(C63)]
    cases = cases[shard::n_shards]

    p_final = mkpred(M1001, [0, 1, 2, 3, 4])
    p_best = mkpred(M1001, [0, 1, 2, 3, 4], ckpt="checkpoint_best.pth")
    p_ov = mkpred(M1001, [0, 1, 2, 3, 4], step=0.75)
    p_a = mkpred(MRESENC, [0])
    p_c = mkpred(MCOMBO, [0])

    out_path = f"{BASE}/tooth_3d_semantic/logs/final_squeeze_shard{shard}.jsonl"
    with open(out_path, "w") as f:
        for i, c in enumerate(cases, 1):
            img = sitk.ReadImage(f"{RAW}/{c}_0000.nii.gz")
            arr = sitk.GetArrayFromImage(img)
            props = {"spacing": img.GetSpacing()}
            gt = sitk.GetArrayFromImage(sitk.ReadImage(f"{GT}/{c}.nii.gz"))

            pf = probs_of(p_final, arr, props)
            d_final = case_dice(np.argmax(pf, 0).astype(np.uint8), gt)
            pb = probs_of(p_best, arr, props)
            d_fb = case_dice(np.argmax(pf + pb, 0).astype(np.uint8), gt)
            del pb
            po = probs_of(p_ov, arr, props)
            d_ov = case_dice(np.argmax(po, 0).astype(np.uint8), gt)
            del po
            pa = probs_of(p_a, arr, props)
            d_a6 = case_dice(np.argmax(5 * pf + pa, 0).astype(np.uint8), gt)
            pc = probs_of(p_c, arr, props)
            d_c6 = case_dice(np.argmax(5 * pf + pc, 0).astype(np.uint8), gt)
            d_all7 = case_dice(np.argmax(5 * pf + pa + pc, 0).astype(np.uint8), gt)
            del pa, pc, pf

            f.write(json.dumps({"case": c, "final": round(d_final, 4), "fb": round(d_fb, 4),
                                "ov": round(d_ov, 4), "a6": round(d_a6, 4),
                                "c6": round(d_c6, 4), "all7": round(d_all7, 4)}) + "\n")
            f.flush()
            print(f"[shard{shard} {i}/{len(cases)}] {c} final={d_final:.4f} fb={d_fb:.4f} "
                  f"ov={d_ov:.4f} a6={d_a6:.4f} c6={d_c6:.4f} all7={d_all7:.4f}", flush=True)
    print("shard done ->", out_path, flush=True)


if __name__ == "__main__":
    main()

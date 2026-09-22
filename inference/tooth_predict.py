#!/usr/bin/env python3
"""
Predict tooth segmentation on a new CBCT image using a trained nnUNetv2 model.

Default production model: Dataset1000_ToothSemanticRAS (true-RAS frame,
tooth_fairy2_m), 5-fold logit ensemble — the configuration behind the
0.8177 holdout result (see tooth_3d_quickstart.md §8).

Usage:
    source dataset_conversion/setup_nnunet_env.sh
    python inference/tooth_predict.py \
        --model_dir tooth_3d_semantic/results/Dataset1000_ToothSemanticRAS/nnUNetTrainer__nnUNetPlans__3d_fullres \
        --input /path/to/new_scan.nii.gz \
        --output /path/to/output.nii.gz

Notes:
    --model_dir must be the TRAINER directory (contains plans.json and fold_N/
    subdirs), NOT a single fold_N directory.
    --folds: folds to average (logit ensemble). Default: auto-detect all
    available folds (5-fold ensemble). Pass e.g. --folds 0 for single-fold.
    Inference uses mirroring + gaussian, matching the holdout evaluation.
    The input scan should be in RAS orientation (same as the training data).
    Output labels are 1-32 (1-8 upper right, 9-16 upper left, 17-24 lower
    left, 25-32 lower right); convert to clinical FDI numbering with
    inference/tooth_relabel_fdi.py.
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import join, isfile, maybe_mkdir_p

# make sibling modules (e.g. id_assigner) importable when run as
# `python inference/tooth_predict.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))


def setup_nnunet_env():
    """Set up nnUNet environment variables if not already set."""
    base = "/home/share/clr/share/work/CBCT/nnUNet/tooth_3d_semantic"
    os.environ.setdefault("nnUNet_raw", join(base, "raw"))
    os.environ.setdefault("nnUNet_preprocessed", join(base, "preprocessed"))
    os.environ.setdefault("nnUNet_results", join(base, "results"))


def predict_single_image(model_dir: str, input_path: str, output_path: str,
                         folds=None, checkpoint_name: str = 'checkpoint_final.pth',
                         id_assign: str = 'on'):
    """Run prediction on a single CBCT image. folds=None auto-detects all folds."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    setup_nnunet_env()

    print(f"Loading model from: {model_dir} (folds: {folds if folds else 'auto-detect all'})")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds=folds,
        checkpoint_name=checkpoint_name,
    )

    # nnUNet expects a list of per-case file lists, and writes the output as
    # <truncated name> + dataset file_ending (.nii.gz)
    input_path = str(Path(input_path).resolve())
    output_truncated = str(Path(output_path).resolve())
    if output_truncated.endswith('.nii.gz'):
        output_truncated = output_truncated[:-len('.nii.gz')]
    final_output = output_truncated + '.nii.gz'
    maybe_mkdir_p(os.path.dirname(final_output))

    print(f"Predicting: {input_path}")
    print(f"Output: {final_output}")

    predictor.predict_from_files(
        [[input_path]],
        [output_truncated],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )

    if id_assign == 'on':
        import json as _json
        from id_assigner import apply_assigner
        import SimpleITK as _sitk
        seg_img = _sitk.ReadImage(final_output)
        seg_arr = _sitk.GetArrayFromImage(seg_img)
        seg_arr, alog = apply_assigner(seg_arr)
        seg_img = _sitk.GetImageFromArray(seg_arr.astype('uint8'))
        seg_img.CopyInformation(_sitk.ReadImage(final_output))
        _sitk.WriteImage(seg_img, final_output)
        log_path = str(Path(final_output).with_suffix('')) + '_idassign_log.json'
        with open(log_path, 'w') as f:
            _json.dump(alog, f, indent=1)
        print(f"ID assigner: {max(0, len(alog)-1)} component(s) re-mapped "
              f"(log: {log_path})")

    print(f"Prediction saved to: {final_output}")


def main():
    parser = argparse.ArgumentParser(description="Predict tooth segmentation on new CBCT")
    parser.add_argument('--model_dir', required=True,
                        help='TRAINER output directory (parent of fold_N), e.g. '
                             'tooth_3d_semantic/results/Dataset1000_ToothSemanticRAS/nnUNetTrainer__nnUNetPlans__3d_fullres')
    parser.add_argument('--input', required=True, help='Input CBCT .nii.gz file')
    parser.add_argument('--output', required=True, help='Output segmentation .nii.gz file')
    parser.add_argument('--folds', nargs='*', type=int, default=None,
                        help='Folds for logit ensemble (default: auto-detect all)')
    parser.add_argument('--id_assign', choices=['on', 'off'], default='on',
                        help='deterministic L/R ID consistency post-processor '
                             '(default on)')
    args = parser.parse_args()

    if not isfile(join(args.model_dir, 'plans.json')):
        print(f"ERROR: {args.model_dir}/plans.json not found — pass the TRAINER directory "
              f"(parent of fold_N), not a fold_N directory")
        sys.exit(1)
    if args.folds:
        for f in args.folds:
            cp = join(args.model_dir, f'fold_{f}', 'checkpoint_final.pth')
            if not isfile(cp):
                print(f"ERROR: checkpoint not found at {cp}")
                sys.exit(1)
    if not isfile(args.input):
        print(f"ERROR: input file not found: {args.input}")
        sys.exit(1)
    if not args.input.endswith('_0000.nii.gz'):
        print(f"ERROR: input filename must end with '_0000.nii.gz' (nnUNet convention), got: {Path(args.input).name}")
        sys.exit(1)

    predict_single_image(args.model_dir, args.input, args.output,
                         folds=args.folds, id_assign=args.id_assign)


if __name__ == '__main__':
    main()

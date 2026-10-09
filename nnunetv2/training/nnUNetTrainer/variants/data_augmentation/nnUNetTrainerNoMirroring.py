from typing import Union, Tuple, List

import numpy as np
import scipy.ndimage as ndi
import torch
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerNoMirroring(nnUNetTrainer):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes


class nnUNetTrainer_onlyMirror01(nnUNetTrainer):
    """
    Only mirrors along spatial axes 0 and 1 for 3D and 0 for 2D
    """
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        if dim == 2:
            mirror_axes = (0, )
        else:
            mirror_axes = (0, 1)
        self.inference_allowed_mirroring_axes = mirror_axes
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes


class nnUNetTrainer_onlyMirror01_1500ep(nnUNetTrainer_onlyMirror01):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1500


class GapCutoutTransform(BasicTransform):
    """Erase one complete tooth from the patch: its image voxels are replaced
    with same-patch background texture and its seg label is set to 0.

    Motivation (tail-optimization-095 Wave 2): the model fills edentulous
    gaps with phantom teeth because patches only see local arch context, so
    a gap reads as 'missing evidence' rather than 'missing tooth'. This
    transform synthesizes gaps-with-hole-chains so 'gap = no tooth' and
    non-consecutive ID chains are learned directly.

    Selection guarantees: victim bbox fully inside the patch with `margin`
    clearance, no -1 (out-of-crop padding) in the affected region, victim
    >= min_tooth_vox voxels. Image fill = dilated(mask) restricted to
    background-or-victim voxels (neighbor teeth are never overwritten);
    donor texture is a same-shape window elsewhere in the patch that is
    entirely background. If no candidate/donor is found the sample passes
    through unchanged.
    """

    def __init__(self, min_tooth_vox: int = 200, margin: int = 2,
                 fill_dilate: int = 1, max_donor_try: int = 64):
        super().__init__()
        self.min_tooth_vox = min_tooth_vox
        self.margin = margin
        self.fill_dilate = fill_dilate
        self.max_donor_try = max_donor_try

    def get_parameters(self, **data_dict) -> dict:
        seg = data_dict.get('segmentation')
        if seg is None or seg.dim() != 4:
            return {'skip': True}
        seg0 = seg[0].detach().cpu().numpy()
        fg = (seg0 > 0)
        candidates = []
        for lab in np.unique(seg0):
            if lab <= 0 or lab > 32:
                continue
            mask = seg0 == lab
            n = int(mask.sum())
            if n < self.min_tooth_vox:
                continue
            zz, yy, xx = np.where(mask)
            pad = self.margin + self.fill_dilate
            z0, z1 = zz.min() - pad, zz.max() + pad + 1
            y0, y1 = yy.min() - pad, yy.max() + pad + 1
            x0, x1 = xx.min() - pad, xx.max() + pad + 1
            if min(z0, y0, x0) < 0 or z1 > seg0.shape[0] or \
                    y1 > seg0.shape[1] or x1 > seg0.shape[2]:
                continue
            # complete tooth + no out-of-crop padding anywhere near it
            win = seg0[z0:z1, y0:y1, x0:x1]
            if (win == -1).any():
                continue
            candidates.append((int(lab), mask, (z0, z1, y0, y1, x0, x1)))
        if not candidates:
            return {'skip': True}
        lab, mask, bbox = candidates[np.random.randint(len(candidates))]
        halo = ndi.binary_dilation(mask, iterations=self.fill_dilate)
        fill = halo & ((seg0 == 0) | (seg0 == lab))
        fz, fy, fx = np.where(fill)
        if len(fz) == 0:
            return {'skip': True}
        # donor: entirely-background window with the fill bbox shape
        fb = (fz.min(), fz.max() + 1, fy.min(), fy.max() + 1, fx.min(), fx.max() + 1)
        dz, dy, dx = fb[1] - fb[0], fb[3] - fb[2], fb[5] - fb[4]
        Z, Y, X = seg0.shape
        for _ in range(self.max_donor_try):
            z0 = np.random.randint(0, Z - dz + 1)
            y0 = np.random.randint(0, Y - dy + 1)
            x0 = np.random.randint(0, X - dx + 1)
            win = seg0[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx]
            if not (win != 0).any():
                donor_win = (z0, z0 + dz, y0, y0 + dy, x0, x0 + dx)
                params = {'skip': False, 'victim_label': lab,
                          'fill_idx': np.stack([fz, fy, fx]),
                          'donor_bbox': donor_win, 'fill_bbox': fb}
                return params
        return {'skip': True}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if params.get('skip', True):
            return img
        img = img.clone()  # functional semantics: never mutate the caller's tensor
        fb, db = params['fill_bbox'], params['donor_bbox']
        fz, fy, fx = params['fill_idx']
        win = img[0, fb[0]:fb[1], fb[2]:fb[3], fb[4]:fb[5]]  # basic slicing -> view
        mask_win = torch.zeros_like(win, dtype=torch.bool)
        mask_win[fz - fb[0], fy - fb[2], fx - fb[4]] = True
        src = img[0, db[0]:db[1], db[2]:db[3], db[4]:db[5]]
        win[mask_win] = src[mask_win]  # view write propagates to the clone
        return img

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        if params.get('skip', True):
            return segmentation
        segmentation = segmentation.clone()
        fz, fy, fx = params['fill_idx']
        lab = params['victim_label']
        seg0 = segmentation[0]
        vals = seg0[fz, fy, fx]
        vals[vals == lab] = 0
        seg0[fz, fy, fx] = vals
        return segmentation


class nnUNetTrainer_onlyMirror01_gapcut(nnUNetTrainer_onlyMirror01):
    """onlyMirror01 + GapCutoutTransform (tail-optimization-095 Wave 2 pilot):
    with p=0.3 per patch, one complete tooth is erased (image + label) to
    synthesize edentulous gaps. Pipeline otherwise identical to the base
    nnUNetTrainer (mirror axes restricted to (0,1) by the parent class).
    """
    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,
                border_mode_seg='constant',
                padding_value_seg=-1,
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GapCutoutTransform(min_tooth_vox=200, margin=2, fill_dilate=1),
            apply_probability=0.3
        ))

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)


class nnUNetTrainer_onlyMirror01_DASegOrd0(nnUNetTrainer_onlyMirror01):
    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,
                mode_seg='nearest',
                border_mode_seg='constant',
                padding_value_seg=-1,
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig

from dinov2.data.transforms import *

logger = logging.getLogger("dinov2")


class DataAugmentationDINO:
    """
    Data augmentation class for DINO, supporting 2D and 3D crops.

    This class sets up and applies a series of transformations to generate
    global and local views from an input image, which can be either 2D or 3D.
    """

    def __init__(self, config: DictConfig, mean: float, std: float) -> None:
        self.config = config
        self.norm_cfg = {"mean": mean, "std": std}

        crops_cfg = config.crops
        self.enable_3d = crops_cfg.views.enable_3d
        self.enable_2d = crops_cfg.views.enable_2d
        self.num_global_2d = crops_cfg.crops_number.global_2d
        self.num_local_3d = crops_cfg.crops_number.local_3d
        self.num_local_2d = crops_cfg.crops_number.local_2d
        self.global_view_multiple = crops_cfg.crops_number.global_view_multiple

        self.transforms = self._create_transforms()

    def _create_base_augmentations(self, skip_first: bool) -> ImageTransforms:
        """Creates a base augmentation pipeline with Flip and Permute."""
        augment = ImageTransforms()
        augment += Flip(skip_first=skip_first)
        augment += Permute(skip_first=skip_first)
        return augment

    def _create_transforms(self) -> Dict[str, Callable]:
        """Builds the dictionary of transformation functions based on config."""
        transforms: Dict[str, Callable] = {}
        crop_sizes = self.config.crops.crop_sizes

        if self.enable_3d:
            if self.enable_2d:
                g_3d_size = crop_sizes.global_2d
            else:
                g_3d_size = crop_sizes.global_3d
            l_3d_size = crop_sizes.local_3d

            general_crop_3d = ImageTransforms()
            general_crop_3d += RandomCrop3D(size=g_3d_size, scale=(0.3, 1.0))
            general_crop_3d += Norm(**self.norm_cfg)
            transforms["general_crop_3d"] = general_crop_3d

            transforms["global_3d"] = self._create_base_augmentations(skip_first=False)

            local_3d_augment = ImageTransforms()
            local_3d_augment += RandomCrop3D(size=l_3d_size, scale=(0.1, 0.5))
            local_3d_augment += self._create_base_augmentations(skip_first=False)
            local_3d_augment += GaussianBlur()
            transforms["local_3d"] = local_3d_augment

        if self.enable_2d:
            g_2d_size = crop_sizes.global_2d
            l_2d_size = crop_sizes.local_2d

            if self.enable_3d:
                transforms["global_3d_resize"] = Resize(
                    output_size=(crop_sizes.global_3d,) * 3
                )
                transforms["slice_to_2d"] = Slice(channels=crop_sizes.channels)
            else:
                global_2d_augment = ImageTransforms()
                global_2d_augment += RandomSliceCrop(
                    size=g_2d_size, channels=crop_sizes.channels, scale=(0.3, 1.0)
                )
                global_2d_augment += Norm(**self.norm_cfg)
                transforms["global_crop_2d"] = global_2d_augment

            transforms["global_2d_augment"] = self._create_base_augmentations(
                skip_first=True
            )

            local_2d_augment = ImageTransforms()
            local_2d_augment += RandomCrop2D(
                size=l_2d_size, channels=crop_sizes.channels, scale=(0.1, 0.5)
            )
            local_2d_augment += self._create_base_augmentations(skip_first=True)
            local_2d_augment += GaussianBlur()
            transforms["local_2d"] = local_2d_augment

        return transforms

    def __call__(
        self, images: torch.Tensor, spacing: Tuple[float, float, float]
    ) -> Dict[str, List[torch.Tensor]]:
        """Applies the configured augmentations to an image."""
        output_crops: Dict[str, List[torch.Tensor]] = {}
        general_crop_3d: Optional[torch.Tensor] = None

        if self.enable_3d:
            general_crop_3d = self.transforms["general_crop_3d"](images, spacing)

            global_3d_input = (
                self.transforms["global_3d_resize"](general_crop_3d)
                if self.enable_2d
                else general_crop_3d
            )

            output_crops["global_3d"] = [
                self.transforms["global_3d"](global_3d_input)
                for _ in range(self.global_view_multiple)
            ]
            output_crops["local_3d"] = [
                self.transforms["local_3d"](global_3d_input)
                for _ in range(self.num_local_3d)
            ]

        if self.enable_2d:
            if general_crop_3d is not None:
                global_2d_source_crops = [
                    self.transforms["slice_to_2d"](general_crop_3d)
                    for _ in range(self.num_global_2d)
                ]
            else:
                global_2d_source_crops = [
                    self.transforms["global_crop_2d"](images, spacing)
                    for _ in range(self.num_global_2d)
                ]

            output_crops["local_2d"] = [
                self.transforms["local_2d"](view)
                for _ in range(self.num_local_2d)
                for view in global_2d_source_crops
            ]
            output_crops["global_2d"] = [
                self.transforms["global_2d_augment"](view)
                for view in global_2d_source_crops
                for _ in range(self.global_view_multiple)
            ]

        return output_crops

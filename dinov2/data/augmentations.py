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

        self.embed_layer = config.student.embed_layer.type

        self.num_crops_global = config.crops.num_crops_global
        self.num_crops_local = config.crops.num_crops_local

        self.size_global = config.crops.size_global
        self.size_local = config.crops.size_local

        self.scale_global = config.crops.scale_global
        self.scale_local = config.crops.scale_local

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

        if self.embed_layer == "patch_3d":

            general_crop_3d = ImageTransforms()
            general_crop_3d += RandomCrop3D(
                size=self.size_global, scale=self.scale_global
            )
            general_crop_3d += Norm(**self.norm_cfg)
            transforms["global"] = general_crop_3d

            transforms["global_augment"] = self._create_base_augmentations(
                skip_first=False
            )

            local_3d_augment = ImageTransforms()
            local_3d_augment += RandomCrop3D(
                size=self.size_local, scale=self.scale_local
            )
            local_3d_augment += self._create_base_augmentations(skip_first=False)
            local_3d_augment += GaussianBlur()
            transforms["local"] = local_3d_augment

        elif self.embed_layer == "patch_2d":

            global_2d_augment = ImageTransforms()
            global_2d_augment += RandomSliceCrop(
                size=self.size_global,
                channels=crop_sizes.channels,
                scale=self.scale_global,
            )
            global_2d_augment += Norm(**self.norm_cfg)
            transforms["global"] = global_2d_augment

            transforms["global_augment"] = self._create_base_augmentations(
                skip_first=True
            )

            local_2d_augment = ImageTransforms()
            local_2d_augment += RandomCrop2D(
                size=self.size_local, scale=self.scale_local
            )
            local_2d_augment += self._create_base_augmentations(skip_first=True)
            local_2d_augment += GaussianBlur()
            transforms["local"] = local_2d_augment

        return transforms

    def __call__(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Applies the configured augmentations to an image."""
        output: Dict[str, torch.Tensor] = {}

        global_crop = self.transforms["global"](images)

        output["global"] = torch.stack(
            [
                self.transforms["global_augment"](global_crop)
                for _ in range(self.num_crops_global)
            ]
        )
        output["local"] = torch.stack(
            [self.transforms["local"](global_crop) for _ in range(self.num_crops_local)]
        )

        return output

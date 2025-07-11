# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

import torch
from omegaconf import DictConfig
from typing import Dict, List
import copy

from .transforms import ImageTransforms


logger = logging.getLogger("dinov2")


class DataAugmentationDINO(object):
    def __init__(self, config: DictConfig, dataset_config: DictConfig) -> None:
        self.augmentations_group = dataset_config.augmentations
        self.transform_groups = config.augmentations[self.augmentations_group].copy()

        assert all(
            key in self.transform_groups for key in ["local_2d", "global_2d", "local_3d", "global_3d"]
        ), "Unrecognized augmentation group."

        self.dataset_config = dataset_config.copy()
        self.transform_groups_config = config.transform_groups.copy()
        self.embed_config = config.student.embed_layers.copy()

        self.transforms = self._load_transforms_from_cfg()

    def _load_transforms_from_cfg(self) -> Dict[str, ImageTransforms]:
        transform_groups = {}

        for transform_group, transform_key in self.transform_groups.items():

            transforms_obj = ImageTransforms(
                self.dataset_config.pixel_range.lower,
                self.dataset_config.pixel_range.upper,
            )
            transforms_list = copy.deepcopy(self.transform_groups_config[transform_key])
            for tc in transforms_list:
                name = tc.pop("name")
                transforms_obj.add_transform(name, tc)

            transforms_obj.add_normalize(
                self.dataset_config.norm.mean, self.dataset_config.norm.std
            )
            
            transform_groups[transform_group] = transforms_obj
        

        return transform_groups

    def __call__(self, image: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        Apply augmentations to the input image.

        Args:
            image: The input image to apply augmentations to.

        Returns:
            output: A dictionary containing the augmented image crops and offsets.
                - "global_crops": A list of global crops of the image.
                - "local_crops": A list of local crops of the image.

        """
        output = {}

        global_crop_1 = self.transforms["global_2d"](image)
        global_crop_2 = self.transforms["global_2d"](image)

        output["global_crops"] = [global_crop_1, global_crop_2]

        local_crops = [
            self.transforms["local_2d"](global_crop_1) for _ in range(8)
        ]

        output["local_crops"] = local_crops

        return output

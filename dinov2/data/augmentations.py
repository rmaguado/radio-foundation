# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

import torch
from omegaconf import DictConfig
import copy

from .transforms import ImageTransforms


logger = logging.getLogger("dinov2")


class DataAugmentationDINO(object):
    def __init__(
        self, config: DictConfig, dataset_config: DictConfig, use_full_image: bool
    ) -> None:
        """
        Initializes an instance of the Augmentations class.

        Args:
            config (DictConfig): The primary configuration object.
            dataset_config (DictConfig): The dataset configuration object.
            use_full_image (bool): Whether to use the full image
        """
        self.dataset_config = dataset_config
        self.local_crops_number = config.crops.local_crops_number
        self.local_crops_size = config.crops.local_crops_size
        self.local_crops_scale = config.crops.local_crops_scale

        self.global_crops_size = (
            config.student.full_image_size
            if use_full_image
            else config.crops.global_crops_size
        )
        self.global_crops_scale = config.crops.global_crops_scale

        self.global1, self.global2, self.local1 = self.load_transforms_from_cfg()

    def build_transform_group(self, transform_key):
        """
        Builds a transformation group based on the given transform key.

        Parameters:
            transform_key (str): The key to identify the desired transformation group.

        Returns:
            transforms.Compose: The composed transformation group.
        """
        image_transforms = ImageTransforms(
            self.dataset_config.pixel_range.lower,
            self.dataset_config.pixel_range.upper,
            self.dataset_config.channels,
        )
        augmentations_list = copy.deepcopy(
            self.dataset_config.augmentations[transform_key]
        )
        for tc in augmentations_list:
            name = tc.pop("name")
            if name == "localcrop":
                image_transforms.add_crop(self.local_crops_size, self.local_crops_scale)
            elif name == "globalcrop":
                image_transforms.add_crop(
                    self.global_crops_size, self.global_crops_scale
                )
            else:
                image_transforms.add_transform(name, tc)

        image_transforms.add_normalize(
            self.dataset_config.norm.mean, self.dataset_config.norm.std
        )
        return image_transforms

    def load_transforms_from_cfg(self):
        """
        Load transforms from configuration file for each group (global1, global2, local).

        Returns:
            tuple: A tuple of transform groups.
        """
        transform_groups = [
            self.build_transform_group(group)
            for group in ["global_1", "global_2", "local"]
        ]
        return tuple(transform_groups)

    def __call__(self, image: torch.tensor) -> dict[str, list[torch.tensor]]:
        """
        Apply augmentations to the input image.

        Args:
            image: The input image to apply augmentations to.

        Returns:
            output: A dictionary containing the augmented image crops and offsets.
                - "global_crops": A list of global crops of the image.
                - "global_crops_teacher": A list of global crops of the image.
                - "local_crops": A list of local crops of the image.
                - "offsets": An empty tuple.

        """
        output = {}

        global_crop_1 = self.global1(image)
        global_crop_2 = self.global2(image)

        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        local_crops = [self.local1(image) for _ in range(self.local_crops_number)]

        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output

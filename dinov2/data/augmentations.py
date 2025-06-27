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
        self.augmentations_config = config.augmentations[dataset_config.augmentation]

        # number of crops
        self.global_crops_3dld = config.crops.global_crops_3dld
        self.local_crops_3dld = config.crops.local_crops_3dld
        self.global_crops_2dld_peraxis = config.crops.global_crops_2dld_peraxis
        self.global_crops_2dhd_peraxis = config.crops.global_crops_2dhd_peraxis
        self.local_crops_2dhd_peraxis = config.crops.local_crops_2dhd_peraxis

        # scale of crops
        self.global_crops_scale = config.crops.global_crops_scale
        self.local_crops_scale = config.crops.local_crops_scale

        # size of crops
        if stage1:
            crop_size_config = config.crops.stage1
        else:
            crop_size_config = config.crops.stage2

        self.global_crop_size_ld = crop_size_config.global_crop_size_ld
        self.local_crop_size_ld = crop_size_config.local_crop_size_ld
        self.global_crop_size_hd = crop_size_config.global_crop_size_hd
        self.local_crop_size_hd = crop_size_config.local_crop_size_hd

        self.transforms = self.load_transforms_from_cfg()

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
        augmentations_list = copy.deepcopy(self.augmentations_config[transform_key])
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
            for group in [
                "global_3dld",
                "local_3dld",
                "global_2dld",
                "global_2dhd",
                "local_2dhd",
            ]
        ]
        return tuple(transform_groups)

    def __call__(self, image: torch.Tensor) -> dict[str, list[torch.Tensor]]:
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

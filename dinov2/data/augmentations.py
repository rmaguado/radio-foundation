# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

import torch
from omegaconf import DictConfig

from dinov2.data.transforms import *


logger = logging.getLogger("dinov2")


class DataAugmentationDINO(object):
    def __init__(self, config: DictConfig, dataset_config: DictConfig) -> None:
        """
        Initializes an instance of the Augmentations class.

        Args:
            config (DictConfig): The primary configuration object.
            dataset_config (DictConfig): The dataset configuration object.
        """
        self.dataset_config = dataset_config
        self.local_crops_number = config.crops.local_crops_number
        self.local_crops_size = config.crops.local_crops_size
        self.local_crops_scale = config.crops.local_crops_scale

        self.global_crops_size = config.crops.global_crops_size
        self.global_crops_scale = config.crops.global_crops_scale

        self.norm = Norm(mean=dataset_config.norm.mean, std=dataset_config.norm.std)

        self.global_base_crop = RandomCrop2D(
            size=self.global_crops_size, scale=self.global_crops_scale
        )

        self.global1 = ImageTransforms()
        self.global1 += RandomCrop2D(
            size=self.global_crops_size, scale=self.global_crops_scale
        )
        self.global1 += Permute2D()
        self.global1 += Flip2D()
        self.global1 += Window(p=0.5)
        self.global1 += self.norm

        self.global2 = ImageTransforms()
        self.global2 += RandomCrop2D(
            size=self.global_crops_size, scale=self.global_crops_scale
        )
        self.global2 += Permute2D()
        self.global2 += Flip2D()
        self.global2 += self.norm

        self.local1 = ImageTransforms()
        self.local1 += RandomCrop2D(
            size=self.local_crops_size, scale=self.local_crops_scale
        )
        self.local1 += self.norm

    def __call__(self, image: torch.Tensor) -> dict[str, list[torch.Tensor]]:
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

        global_crop_1 = self.global1(image)
        global_crop_2 = self.global2(image)

        output["global_crops"] = [global_crop_1, global_crop_2]

        local_crops = [self.local1(image) for _ in range(self.local_crops_number)]

        output["local_crops"] = local_crops

        return output

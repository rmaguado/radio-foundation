# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

import torch
from omegaconf import DictConfig
import copy
from typing import Tuple, Dict, List, Callable

from dinov2.data.transforms import ImageTransforms, get_transform


logger = logging.getLogger("dinov2")


class DataAugmentationDINO(object):
    def __init__(
        self, config: DictConfig, dataset_config: DictConfig
    ) -> None:
        """
        Initializes an instance of the Augmentations class.

        Args:
            config (DictConfig): The primary configuration object.
            dataset_config (DictConfig): The dataset configuration object.
        """
        self.dataset_config = dataset_config
        self.augmentations_config = config.augmentations[dataset_config.augmentation]

        self.transforms = self.load_transforms_from_cfg()

    def build_transform_group(self, transform_key) -> List[Callable]:
        """
        Builds a transformation group based on the given transform key.

        Parameters:
            transform_key (str): The key to identify the desired transformation group.

        Returns:
            transforms.Compose: The composed transformation group.
        """
        norm_cfg = {
            "mean": self.dataset_config.norm.mean,
            "std": self.dataset_config.norm.std
        }

        transforms_cfg = copy.deepcopy(self.augmentations_config[transform_key])
        image_transforms = ImageTransforms()
        
        for tc in transforms_cfg:
            name = tc.pop("name")
            if name == "norm":
                tc = norm_cfg
            image_transforms += get_transform(name, tc)

        return image_transforms

    def load_transforms_from_cfg(self) -> Dict[str, Callable]:
        """
        Load transforms from configuration file for each group (global1, global2, local).

        Returns:
            Dict[str, Callable]: A dict of transform groups.
        """
        
        return {
            group: self.build_transform_group(group)
            for group in ["global", "local"]
        }

    def __call__(self, image: torch.Tensor, spacing: Tuple[float,float,float]) -> Dict[str, List[torch.Tensor]]:
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

        global_crop_1 = self.transforms["global"](image, spacing)
        global_crop_2 = self.transforms["global"](image, spacing)

        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        local_crops = [self.transforms["local"](image) for _ in range(8)]

        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output

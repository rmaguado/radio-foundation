# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

import torch
from torchvision import transforms
from omegaconf import DictConfig

from .transforms import transformkeys


logger = logging.getLogger("dinov2")


err_not_recognized = "Transform '{s}' is not recognized. \
Please check the config file under augmentations."


class DataAugmentationDINO(object):
    def __init__(self, cfg: DictConfig, use_full_image: bool) -> None:
        """
        Initializes an instance of the Augmentations class.

        Args:
            cfg (DictConfig): configuration object
            use_full_image (bool): whether to use the full image
        """

        self.cfg = cfg
        self.local_crops_number = cfg.crops.local_crops_number
        self.local_crops_size = cfg.crops.local_crops_size
        self.local_crops_scale = cfg.crops.local_crops_scale

        self.global_crops_size = (
            cfg.student.full_image_size
            if use_full_image
            else cfg.crops.global_crops_size
        )
        self.global_crops_scale = cfg.crops.global_crops_scale

        self.localcrop = self.get_local_crop()
        self.globalcrop = self.get_global_crop()
        self.normalize = transforms.Normalize(mean=cfg.norm.mean, std=cfg.norm.std)

        self.global1, self.global2, self.local1 = self.load_transforms_from_cfg()

    def get_local_crop(self):
        """
        Returns a random resized crop transformation for local crops.

        Returns:
            transforms.RandomResizedCrop: A random resized crop transformation with the specified parameters.
        """
        return transforms.RandomResizedCrop(
            self.local_crops_size,
            scale=self.local_crops_scale,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )

    def get_global_crop(self):
        """
        Returns a random resized crop transformation for global crops.

        Returns:
            transforms.RandomResizedCrop: A random resized crop transformation with the specified parameters.
        """
        return transforms.RandomResizedCrop(
            self.global_crops_size,
            scale=self.global_crops_scale,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )

    def create_transform(self, transform_options: DictConfig):
        """
        Creates and returns a transform based on the given configuration.

        Args:
            transform_options (dict): A dictionary containing the configuration for the transform.

        Returns:
            transform: The created transform object.
        """
        name = transform_options["name"]
        if name == "localcrop":
            return self.localcrop
        elif name == "globalcrop":
            return self.globalcrop
        elif name in transformkeys:
            transform_cls = transformkeys[name]
            params = {k: float(v) for k, v in transform_options.items() if k != "name"}
            return transform_cls(**params)
        else:
            raise ValueError(err_not_recognized.format(s=name))

    def build_transform_group(self, transform_key):
        """
        Builds a transformation group based on the given transform key.

        Parameters:
            transform_key (str): The key to identify the desired transformation group.

        Returns:
            transforms.Compose: The composed transformation group.
        """
        transforms_list = [
            self.create_transform(tc) for tc in self.cfg.augmentations[transform_key]
        ]
        transforms_list.append(self.normalize)
        return transforms.Compose(transforms_list)

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

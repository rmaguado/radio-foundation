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
    def __init__(self, config: DictConfig, dataset_config: DictConfig) -> None:
        self.dataset_config = dataset_config
        self.augmentations_config = config.augmentations[dataset_config.augmentation]
        self.crops_config = config.crops

        self.global_crops_scale = self.crops_config.global_crops_scale
        self.local_crops_scale = self.crops_config.local_crops_scale

        self.crop_groups = {}
        self.target_groups = []
        self.nontarget_groups = []
        for group_dict in self.crops_config.crop_groups:
            group_name = group_dict.pop("name")
            self.crop_groups[group_name] = group_dict

            is_target = group_dict.get("is_target", False)

            if is_target:
                self.target_groups.append(group_name)
            else:
                self.nontarget_groups.append(group_name)

        self.transforms = self.load_transforms_from_cfg()

    def build_transform_group(self, transform_key):
        image_transforms = ImageTransforms(
            self.dataset_config.pixel_range.lower,
            self.dataset_config.pixel_range.upper,
            self.dataset_config.channels,
        )
        augmentations_list = copy.deepcopy(self.augmentations_config[transform_key])
        for tc in augmentations_list:
            name = tc.pop("name")
            if name == "globalcrop":
                crop_size = self.crop_groups[transform_key]["size"]
                image_transforms.add_crop(crop_size, self.global_crops_scale)
            elif name == "localcrop":
                crop_size = self.crop_groups[transform_key]["size"]
                image_transforms.add_crop(crop_size, self.local_crops_scale)
            else:
                image_transforms.add_transform(name, tc)

        image_transforms.add_normalize(
            self.dataset_config.norm.mean, self.dataset_config.norm.std
        )
        return image_transforms

    def load_transforms_from_cfg(self):
        transform_groups = {
            group: self.build_transform_group(group)
            for group in self.crop_groups.keys()
        }
        return transform_groups

    def __call__(self, image: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        output = {}

        for target_group in self.target_groups:
            transform_group = self.transforms[target_group]
            targets = self.crop_groups[target_group].get("targets", [target_group])
            encoder_type = self.crop_groups[target_group]["encoder_type"]
            group_outputs = [
                transform_group(image)
                for _ in range(self.crop_groups[target_group]["num_crops"])
            ]

            output[target_group] = {
                "images": group_outputs,
                "is_target": True,
                "targets": targets,
                "encoder_type": encoder_type,
            }

        for nontarget_group in self.nontarget_groups:
            transform_group = self.transforms[nontarget_group]
            targets = self.crop_groups[nontarget_group]["target_groups"]
            encoder_type = self.crop_groups[nontarget_group]["encoder_type"]
            main_target = targets[0]
            source_images = output[main_target]["images"]
            group_outputs = []

            for source_image in source_images:
                group_outputs.append(
                    [
                        transform_group(source_image)
                        for _ in range(self.crop_groups[nontarget_group]["num_crops"])
                    ]
                )

            output[nontarget_group] = {
                "images": group_outputs,
                "is_target": False,
                "targets": targets,
                "encoder_type": encoder_type,
            }

        return output

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
        self.embed_config = config.student.embed_layers

        self.global_crop_scale = self.crops_config.global_crop_scale
        self.local_crop_scale = self.crops_config.local_crop_scale

        self.crop_groups_config = {}
        self.target_groups = []
        self.nontarget_groups = []

        for group_dict in self.crops_config.crop_groups:
            group_dict_copy = group_dict.copy()
            group_name = group_dict_copy.pop("name")
            self.crop_groups_config[group_name] = group_dict_copy

            is_target = group_dict_copy.get("is_target", False)
            if is_target:
                self.target_groups.append(group_name)
            else:
                self.nontarget_groups.append(group_name)

        self.transforms = self.load_transforms_from_cfg()

        self.group_info = {}
        for group_name, group_config in self.crop_groups_config.items():
            is_target = group_config.get("is_target", False)
            embed_layer = group_config["embed_layer"]
            img_size = group_config["size"]

            patch_size_list = [
                x["patch_size"] for x in self.embed_config if x["type"] == embed_layer
            ]
            if not patch_size_list:
                logger.error(f"Embed layer '{embed_layer}' not found in embed_config.")
                raise ValueError(
                    f"Embed layer '{embed_layer}' not found in embed_config."
                )
            patch_size = patch_size_list[0]

            targets = group_config.get("targets", [group_name])

            self.group_info[group_name] = {
                "num_crops": group_config["num_crops"],
                "output_template": {
                    "is_target": is_target,
                    "targets": targets,
                    "embed_layer": embed_layer,
                    "patches_shape": img_size // patch_size,
                },
            }

    def build_transform_group(self, transform_key):
        image_transforms = ImageTransforms(
            self.dataset_config.pixel_range.lower,
            self.dataset_config.pixel_range.upper,
            self.dataset_config.channels,
        )
        group_config = self.crop_groups_config[transform_key]
        augmentations_list = copy.deepcopy(self.augmentations_config[transform_key])

        for tc in augmentations_list:
            name = tc.pop("name")
            if name == "globalcrop":
                crop_size = group_config["size"]
                image_transforms.add_crop(crop_size, self.global_crop_scale)
            elif name == "localcrop":
                crop_size = group_config["size"]
                image_transforms.add_crop(crop_size, self.local_crop_scale)
            else:
                image_transforms.add_transform(name, tc)

        image_transforms.add_normalize(
            self.dataset_config.norm.mean, self.dataset_config.norm.std
        )
        return image_transforms

    def load_transforms_from_cfg(self):
        transform_groups = {
            group: self.build_transform_group(group)
            for group in self.crop_groups_config.keys()
        }
        return transform_groups

    def __call__(self, image: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        output = {}

        for target_group in self.target_groups:
            info = self.group_info[target_group]
            transform_group = self.transforms[target_group]

            group_outputs = [transform_group(image) for _ in range(info["num_crops"])]

            output_dict = info["output_template"].copy()
            output_dict["images"] = group_outputs
            output[target_group] = output_dict

        for nontarget_group in self.nontarget_groups:
            info = self.group_info[nontarget_group]
            transform_group = self.transforms[nontarget_group]

            main_target = info["output_template"]["targets"][0]
            source_images = output[main_target]["images"]
            group_outputs = []
            for i, source_image in enumerate(source_images):
                transformed_crops = [
                    transform_group(source_image) for _ in range(info["num_crops"])
                ]
                group_outputs.append(transformed_crops)

            output_dict = info["output_template"].copy()
            output_dict["images"] = group_outputs
            output[nontarget_group] = output_dict

        return output

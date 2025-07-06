# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

import torch
from omegaconf import DictConfig
import copy
from einops import rearrange

from .transforms import ImageTransforms


logger = logging.getLogger("dinov2")


class DataAugmentationDINO(object):
    def __init__(self, config: DictConfig, dataset_config: DictConfig) -> None:
        self.dataset_config = dataset_config.copy()
        self.transform_groups = config.transform_groups.copy()
        self.crops_config = config.crops.copy()
        self.embed_config = config.student.embed_layers.copy()

        self.transforms_graph = config.augmentations[dataset_config.augmentation].copy()

        self.global_crop_scale = self.crops_config.global_crop_scale
        self.local_crop_scale = self.crops_config.local_crop_scale

        self.group_info = {}
        self.transforms = {}
        for group_config in self.transform_groups:
            group_name = group_config["name"]
            img_size = group_config["size"]
            num_crops = group_config["num_crops"]
            embed_layer = group_config["embed_layer"]
            transforms_list = group_config["transforms"]

            patch_size_list = [
                x["patch_size"] for x in self.embed_config if x["type"] == embed_layer
            ]
            if not patch_size_list:
                logger.error(f"Embed layer '{embed_layer}' not found in embed_config.")
                raise ValueError(
                    f"Embed layer '{embed_layer}' not found in embed_config."
                )
            patch_size = patch_size_list[0]
            patches_shape = img_size // patch_size
            if embed_layer == "patch2d":
                mask_shape = (patches_shape,) * 2
            elif embed_layer == "patch3d":
                mask_shape = (patches_shape,) * 3
            else:
                raise ValueError(f"Embed layer {embed_layer} not recognized.")

            self.group_info[group_name] = {
                "size": img_size,
                "num_crops": num_crops,
                "embed_layer": embed_layer,
                "mask_shape": mask_shape,
            }

            self.transforms[group_name] = self._build_transform_group(
                transforms_list, img_size
            )

    def _build_transform_group(self, transforms_list, img_size):
        image_transforms = ImageTransforms(
            self.dataset_config.pixel_range.lower,
            self.dataset_config.pixel_range.upper,
            self.dataset_config.channels,
        )
        transforms_list_copy = copy.deepcopy(transforms_list)

        for tc in transforms_list_copy:
            name = tc.pop("name")
            if name == "globalcrop":
                image_transforms.add_crop(img_size, self.global_crop_scale)
            elif name == "localcrop":
                image_transforms.add_crop(img_size, self.local_crop_scale)
            else:
                image_transforms.add_transform(name, tc)

        image_transforms.add_normalize(
            self.dataset_config.norm.mean, self.dataset_config.norm.std
        )
        return image_transforms

    def _recursive_graph_process(self, image_views, groups) -> dict:
        output = {}
        for group in groups:
            group_name = group["name"]
            group_info = self.group_info[group_name]
            num_crops = group_info["num_crops"]
            group_transforms = self.transforms[group_name]
            subgroups = group.get("subgroups", [])

            group_views = []
            for prev_view in image_views:
                group_views.extend([group_transforms(prev_view) for _ in range(num_crops)])

            stack = torch.stack(group_views)

            output[group_name] = {
                "images": stack,
                **group_info
            }

            subgroup_output = self._recursive_graph_process(group_views, subgroups)
            output.update(subgroup_output)

        return output

    def __call__(self, image: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        output = self._recursive_graph_process([image], self.transforms_graph)
        return output

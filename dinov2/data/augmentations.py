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
    """
    Data augmentation pipeline for DINO self-supervised learning.

    This class builds and applies a set of image transformation groups, each with its own
    cropping, normalization, and augmentation strategy, as specified by the configuration.
    It supports both 2D and 3D crops, and can recursively apply subgroups of transformations.

    Args:
        config (DictConfig): Main configuration object specifying augmentation and crop settings.
        dataset_config (DictConfig): Dataset-specific configuration (pixel range, normalization, etc).
    """

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
            targets = group_config.get("targets", [])
            is_target = group_config.get("is_target", False)
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
                "targets": targets,
                "is_target": is_target,
            }

            self.transforms[group_name] = self._build_transform_group(
                transforms_list, img_size
            )

    def _build_transform_group(self, transforms_list, img_size):
        """
        Constructs a transformation pipeline for a group.

        Args:
            transforms_list (list): List of transformation configs for this group.
            img_size (int): Target image size for cropping.

        Returns:
            ImageTransforms: Composed transformation pipeline.
        """
        image_transforms = ImageTransforms(
            self.dataset_config.pixel_range.lower,
            self.dataset_config.pixel_range.upper,
        )
        transforms_list_copy = copy.deepcopy(transforms_list)

        for tc in transforms_list_copy:
            name = tc.pop("name")
            if name == "globalcrop2d":
                image_transforms.add_crop(img_size, self.global_crop_scale, is_3d=False)
            elif name == "localcrop2d":
                image_transforms.add_crop(img_size, self.local_crop_scale, is_3d=False)
            elif name == "globalcrop3d":
                image_transforms.add_crop(img_size, self.global_crop_scale, is_3d=True)
            elif name == "localcrop3d":
                image_transforms.add_crop(img_size, self.local_crop_scale, is_3d=True)
            else:
                image_transforms.add_transform(name, tc)

        image_transforms.add_normalize(
            self.dataset_config.norm.mean, self.dataset_config.norm.std
        )
        return image_transforms

    def _recursive_graph_process(self, image_views, groups, crop_dims=None) -> dict:
        """
        Recursively applies transformation groups and their subgroups to the input image views.

        Args:
            image_views (list): List of input image tensors to transform.
            groups (list): List of group configs to apply.
            crop_dims (list, optional): Dimensions of crops at each recursion level.

        Returns:
            dict: Dictionary mapping group names to transformed images and metadata.
        """
        if crop_dims is None:
            crop_dims = []

        output = {}
        for group in groups:
            group_name = group["name"]
            group_info = self.group_info[group_name]
            num_crops = group_info["num_crops"]
            group_transforms = self.transforms[group_name]
            subgroups = group.get("subgroups", [])

            group_views = []
            for prev_view in image_views:
                transformed = [group_transforms(prev_view) for _ in range(num_crops)]
                group_views.extend(transformed)

            stacked = torch.stack(group_views)
            full_dims = crop_dims + [num_crops]
            view_shape = full_dims  # [n0, n1, n2, ...]

            sample_view = group_views[0]
            img_shape = sample_view.shape
            full_shape = view_shape + list(
                img_shape
            )  # [n0, n1, n2, ..., slices, height, width]

            stacked = stacked.view(*full_shape)

            output[group_name] = {
                "images": stacked,
                "view_shape": view_shape,
                **group_info,
            }
            subgroup_output = self._recursive_graph_process(
                group_views, subgroups, crop_dims + [num_crops]
            )
            output.update(subgroup_output)

        return output

    def __call__(self, image: torch.Tensor) -> dict:
        """
        Applies the full augmentation graph to an input image tensor.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            dict: Dictionary of transformed image views and associated metadata for each group.
        """
        output = self._recursive_graph_process([image], self.transforms_graph)
        # Check for NaNs/Infs in augmented images
        for group_name, group in output.items():
            if torch.isnan(group["images"]).any() or torch.isinf(group["images"]).any():
                logger.error(
                    f"NaN or Inf detected in augmented images for group {group_name}"
                )
        return output

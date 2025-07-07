# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random
import logging
import numpy as np
from einops import rearrange

from typing import List, Tuple, Dict, Callable, Any

logger = logging.getLogger("dinov2")


def collate_data_and_cast(
    samples_list: List[Any],
    mask_ratio_tuple: Tuple[float, float],
    mask_probability: float,
    dtype: torch.dtype,
    mask_generator: Callable,
) -> Dict[str, Any]:
    """
    Collates a list of augmented sample dictionaries into a batch suitable for model input.

    This function stacks image views, applies type casting, and generates random masks for target groups.
    It supports multi-view, multi-group data as used in DINO-style self-supervised learning.

    Args:
        samples_list (List[Any]): List of dictionaries, each containing augmented views for a sample.
        mask_ratio_tuple (Tuple[float, float]): Range (min, max) for mask ratios to sample from.
        mask_probability (float): Probability of applying a mask to a view.
        dtype (torch.dtype): Data type to cast images to (e.g., torch.float32).
        mask_generator (Callable): Function to generate a mask given a shape and ratio.

    Returns:
        Dict[str, Any]: Dictionary mapping group names to collated batch data, including images, masks, and metadata.
    """
    view_groups = samples_list[0].keys()
    collated_views = {}

    for group_name in view_groups:
        is_target = samples_list[0][group_name]["is_target"]

        group_uncollated_images = [
            samples_list[i][group_name]["images"] for i in range(len(samples_list))
        ]

        collated_views[group_name] = {
            "images": torch.stack(group_uncollated_images).to(dtype),
            "is_target": is_target,
            "targets": samples_list[0][group_name]["targets"],
            "embed_layer": samples_list[0][group_name]["embed_layer"],
            "mask_shape": samples_list[0][group_name]["mask_shape"],
            "view_shape": samples_list[0][group_name]["view_shape"],
        }

    target_group_names = [k for k, v in collated_views.items() if v["is_target"]]
    batch_size = collated_views[target_group_names[0]]["images"].shape[0]
    total_views = (
        sum(np.prod(collated_views[k]["view_shape"]) for k in target_group_names)
        * batch_size
    )
    n_samples_masked = int(total_views * mask_probability)

    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked).tolist()
    probs += [0] * (total_views - n_samples_masked)
    random.shuffle(probs)

    for group_name in target_group_names:

        batch_size = collated_views[group_name]["images"].shape[0]
        mask_shape = collated_views[group_name]["mask_shape"]
        view_shape = collated_views[group_name]["view_shape"]
        num_views = batch_size * np.prod(view_shape)

        masks = []

        for i in range(num_views):

            mask_ratio = probs.pop(0)
            masks.append(torch.from_numpy(mask_generator(mask_shape, mask_ratio)))

        stacked_masks = torch.stack(masks)
        target_mask_shape = (batch_size, *view_shape, np.prod(mask_shape))

        collated_views[group_name]["masks"] = stacked_masks.view(*target_mask_shape)

    return collated_views

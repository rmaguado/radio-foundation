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


MASKABLE_VIEW_NAMES = {"global_2d", "global_3d"}
VIEW_INFO = {
    "global_3d": {"is_target": True, "embed_layer": "patch_3d"},
    "local_3d": {"is_target": False, "embed_layer": "patch_3d"},
    "global_2d": {"is_target": True, "embed_layer": "patch_2d"},
    "local_2d": {"is_target": False, "embed_layer": "patch_2d"},
}


def collate_data_and_cast(
    samples: List[Dict[str, List[torch.Tensor]]],
    mask_ratio_range: Tuple[float, float],
    mask_probability: float,
    mask_shapes: Dict[str, Tuple[int, ...]],
    dtype: torch.dtype,
    mask_generator: Callable[[Tuple[int, ...], float], np.ndarray],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Collates, casts, and applies random masking to a batch of data.

    This function processes a list of samples, where each sample is a dictionary
    mapping view names to image tensors. It collates them into batches, casts them
    to a specified data type, and applies a random masking strategy to designated
    views (e.g., "global_2d", "global_3d").

    Args:
        samples: A list of dictionaries, where each dict represents a data sample.
        mask_ratio_range: A tuple (min_ratio, max_ratio) specifying the range for
                          the masking ratio.
        mask_probability: The probability that any given maskable view will be selected
                          for masking.
        mask_shapes: A dictionary mapping view names to their spatial dimensions
                     (e.g., (H, W)) for mask generation.
        dtype: The desired `torch.dtype` for the output tensors (e.g., `torch.float16`).
        mask_generator: A function that produces a mask array given a shape and a
                        mask ratio.

    Returns:
        A dictionary mapping each view name to its collated 'images' tensor
        and an optional 'masks' tensor if the view is maskable.
    """
    batch_size = len(samples)
    view_names = samples[0].keys()

    collated_data = {}
    total_maskable_views = 0
    for name in view_names:
        images = torch.stack([torch.stack(s[name]) for s in samples]).to(dtype)
        collated_data[name] = {"images": images, **VIEW_INFO[name]}
        if name in MASKABLE_VIEW_NAMES:
            total_maskable_views += int(np.prod(images.shape[:2]))

    num_to_mask = int(total_maskable_views * mask_probability)
    if num_to_mask == 0:
        return collated_data

    masked_ratios = torch.linspace(
        start=mask_ratio_range[0], end=mask_ratio_range[1], steps=num_to_mask
    )

    all_ratios = torch.zeros(total_maskable_views)
    all_ratios[:num_to_mask] = masked_ratios

    shuffled_ratios = all_ratios[torch.randperm(total_maskable_views)]

    mask_idx_counter = 0
    for name in MASKABLE_VIEW_NAMES:
        if name in collated_data:
            images = collated_data[name]["images"]
            num_views_in_batch = np.prod(images.shape[:2])
            mask_shape = mask_shapes[name]

            start, end = mask_idx_counter, mask_idx_counter + num_views_in_batch
            ratios_for_view = shuffled_ratios[start:end]
            mask_idx_counter = end

            masks = [
                torch.from_numpy(mask_generator(mask_shape, ratio.item()))
                for ratio in ratios_for_view
            ]

            stacked_masks = torch.stack(masks)
            collated_data[name]["masks"] = rearrange(
                stacked_masks, "(b v) ... -> b v (...)", b=batch_size
            )

    return collated_data

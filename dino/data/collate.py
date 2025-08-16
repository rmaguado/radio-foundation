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
    samples: List[Dict[str, torch.Tensor]],
    mask_ratio_range: Tuple[float, float],
    mask_probability: float,
    mask_shape: Tuple[int, ...],
    dtype: torch.dtype,
    mask_generator: Callable[[Tuple[int, ...], float], np.ndarray],
) -> Dict[str, torch.Tensor]:
    batch_size = len(samples)

    global_images = torch.stack([s["global"] for s in samples]).to(dtype)
    local_images = torch.stack([s["local"] for s in samples]).to(dtype)
    collated_data = {"global": global_images, "local": local_images}

    total_maskable_views = int(np.prod(global_images.shape[:2]))
    num_to_mask = int(total_maskable_views * mask_probability)

    if num_to_mask == 0:
        return collated_data

    masked_ratios = torch.linspace(
        start=mask_ratio_range[0], end=mask_ratio_range[1], steps=num_to_mask
    )

    all_ratios = torch.zeros(total_maskable_views)
    all_ratios[:num_to_mask] = masked_ratios
    shuffled_ratios = all_ratios[torch.randperm(total_maskable_views)]

    masks = [
        torch.from_numpy(ndarray=mask_generator(mask_shape, ratio.item()))
        for ratio in shuffled_ratios
    ]

    stacked_masks = torch.stack(masks)
    collated_data["masks"] = rearrange(
        stacked_masks, "(b v) ... -> b v (...)", b=batch_size
    )

    return collated_data

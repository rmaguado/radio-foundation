# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random

from typing import List, Tuple, Dict, Callable, Any


def collate_data_and_cast(
    samples_list: List[Any],
    mask_ratio_tuple: Tuple[float, float],
    mask_probability: float,
    dtype: torch.dtype,
    n_tokens: int,
    mask_generator: Callable,
) -> Dict[str, Any]:
    """
    Collates data and casts it to the specified data type.

    Args:
        samples_list (list): A list of samples.
        mask_ratio_tuple (tuple): A tuple containing the minimum and maximum mask ratios.
        mask_probability (float): The probability of masking.
        dtype (torch.dtype): The data type to cast the collated crops to.
        n_tokens (int): The number of tokens.
        mask_generator (function): A function that generates masks.

    Returns:
        dict: A dictionary containing the collated data.
            - collated_views (dict): A dictionary of collated views.
            - collated_masks (torch.Tensor): The collated masks.
            - mask_indices_list (torch.Tensor): The mask indices list.
            - masks_weight (torch.Tensor): The masks weight.
            - upperbound (int): The upperbound.
            - n_masked_patches (torch.Tensor): The number of masked patches.
    """

    view_groups = samples_list[0][0].keys()
    collated_views = {}

    for group_name in view_groups:
        is_target = samples_list[0][0][group_name]["is_target"]
        group_images = []

        for sample in samples_list:
            group_data = sample[0][group_name]["images"]
            if isinstance(group_data[0], list):
                group_images.extend([img for sublist in group_data for img in sublist])
            else:
                group_images.extend(group_data)

        collated_views[group_name] = {
            "images": torch.stack(group_images).to(dtype),
            "is_target": is_target,
            "targets": samples_list[0][0][group_name]["targets"],
        }

    target_group_names = [k for k, v in collated_views.items() if v["is_target"]]
    total_views = sum(collated_views[k]["images"].shape[0] for k in target_group_names)

    n_samples_masked = int(total_views * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []

    for i in range(n_samples_masked):
        mask_prob = probs[i].item()
        masks_list.append(torch.BoolTensor(mask_generator(input_shape, mask_prob)))
        upperbound += int(n_tokens * mask_prob)
    for _ in range(n_samples_masked, total_views):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (
        (1 / collated_masks.sum(-1).clamp(min=1.0))
        .unsqueeze(-1)
        .expand_as(collated_masks)[collated_masks]
    )

    return {
        "collated_views": {k: v["images"] for k, v in collated_views.items()},
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full(
            (1,), fill_value=mask_indices_list.shape[0], dtype=torch.long
        ),
    }

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random
import logging

from typing import List, Tuple, Dict, Callable, Any

logger = logging.getLogger("dinov2")


def collate_data_and_cast(
    samples_list: List[Any],
    mask_ratio_tuple: Tuple[float, float],
    mask_probability: float,
    dtype: torch.dtype,
    mask_generator: Callable,
) -> Dict[str, Any]:
    view_groups = samples_list[0].keys()
    collated_views = {}

    for group_name in view_groups:
        is_target = samples_list[0][group_name]["is_target"]
        group_images = []

        for sample_idx, sample in enumerate(samples_list):
            group_data = sample[group_name]["images"]

            if is_target:
                group_images.append(torch.stack(group_data))
            else:
                group_images.append(
                    torch.stack([torch.stack(crop) for crop in group_data])
                )

        collated_views[group_name] = {
            "images": torch.stack(group_images).to(dtype),
            "is_target": is_target,
            "targets": samples_list[0][group_name]["targets"],
            "embed_layer": samples_list[0][group_name]["embed_layer"],
            "mask_shape": samples_list[0][group_name]["mask_shape"],
        }

    target_group_names = [k for k, v in collated_views.items() if v["is_target"]]
    total_views = sum(
        sum(collated_views[k]["images"].shape[:2]) for k in target_group_names
    )
    n_samples_masked = int(total_views * mask_probability)

    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1).tolist()
    probs += [0] * (total_views - n_samples_masked)
    random.shuffle(probs)

    for group_name in target_group_names:

        group_masks = []
        batch_shape = collated_views[group_name]["images"].shape
        mask_shape = collated_views[group_name]["mask_shape"]

        for i in range(batch_shape[0]):

            crop_masks = []
            for j in range(batch_shape[1]):
                mask_ratio = probs.pop(0)
                crop_masks.append(
                    torch.from_numpy(mask_generator(mask_shape, mask_ratio))
                )
            group_masks.append(torch.stack(crop_masks))

        collated_views[group_name]["masks"] = torch.BoolTensor(torch.stack(group_masks))

    return collated_views

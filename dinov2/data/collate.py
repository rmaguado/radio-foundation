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
    mask_generator: Callable,
) -> Dict[str, Any]:

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
            "embed_layer": samples_list[0][0][group_name]["embed_layer"],
            "patches_shape": samples_list[0][0][group_name]["patches_shape"],
        }

    target_group_names = [k for k, v in collated_views.items() if v["is_target"]]
    total_views = sum(collated_views[k]["images"].shape[0] for k in target_group_names)

    n_samples_masked = int(total_views * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked).tolist()
    probs += [0] * (total_views - n_samples_masked)
    random.shuffle(probs)

    collated_masks = {}

    for group_name in target_group_names:
        n_images = collated_views[group_name]["images"].shape[0]
        group_masks = []

        for i in range(n_images):
            mask_ratio = probs.pop()
            image_shape = collated_views[group_name]["patches_shape"]
            mask = mask_generator(image_shape, mask_ratio)
            group_masks.append(mask)

        collated_masks[group_name] = torch.BoolTensor(group_masks)

    return {
        "collated_views": collated_views,
        "collated_masks": collated_masks,
    }

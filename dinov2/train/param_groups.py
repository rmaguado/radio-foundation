# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging


logger = logging.getLogger("dinov2")


def _get_decay_rate(
    name: str,
    lr_decay_rate: float,
    num_layers: int,
) -> float:
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        (float): lr decay rate for the given parameter.
    """
    if name in [
        "pos_embed",
        "cls_pos_embed",
        "mask_token",
        "cls_token",
        "register_tokens",
    ]:
        layer_id = 0
    elif "embed_layers" in name:
        layer_id = 0
    elif "blocks" in name:
        layer_id = int(name.split("blocks.")[1].split(".")[0]) + 1
    else:
        layer_id = num_layers + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_params_groups_with_decay(
    model,
    lr_decay_rate,
    patch_embed_lr_mult,
    num_layers,
):

    all_param_groups = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        decay_rate = _get_decay_rate(name, lr_decay_rate, num_layers)
        d = {
            "params": param,
            "is_last_layer": False,
            "lr_multiplier": decay_rate,
            "wd_multiplier": 1.0,
            "name": name,
        }

        if "last_layer" in name:
            d.update({"is_last_layer": True})

        if ".bias" in name or "norm" in name or "gamma" in name:
            d.update({"wd_multiplier": 0.0})

        if "embed_layers" in name:
            d.update({"lr_multiplier": d["lr_multiplier"] * patch_embed_lr_mult})

        all_param_groups.append(d)
        logger.debug(
            f"""{name}: lr_multiplier: {d["lr_multiplier"]}, wd_multiplier: {d["wd_multiplier"]}"""
        )

    return all_param_groups

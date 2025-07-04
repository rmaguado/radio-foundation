# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from collections import defaultdict
import logging


logger = logging.getLogger("dinov2")


def get_vit_lr_decay_rate(
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
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if (
            ".pos_embed" in name
            or ".patch_embed" in name
            or ".mask_token" in name
            or ".cls_token" in name
            or ".register_tokens" in name
        ):
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
        elif "blocks." in name and "residual." not in name:
            layer_id = int(name[name.find("blocks.") :].split(".")[1]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_params_groups_with_decay(model, lr_decay_rate=1.0, patch_embed_lr_mult=1.0):
    if hasattr(model, "blocks"):
        logger.debug("first code branch")
        n_blocks = len(model.blocks)
    elif hasattr(model, "backbone"):
        logger.debug("second code branch")
        n_blocks = len(model.backbone.blocks)
    else:
        logger.debug("else code branch")
        n_blocks = 0
    all_param_groups = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        decay_rate = get_vit_lr_decay_rate(
            name,
            lr_decay_rate,
            num_layers=n_blocks,
        )
        d = {
            "params": param,
            "is_last_layer": False,
            "lr_multiplier": decay_rate,
            "wd_multiplier": 1.0,
            "name": name,
        }

        if "last_layer" in name:
            d.update({"is_last_layer": True})

        if name.endswith(".bias") or "norm" in name or "gamma" in name:
            d.update({"wd_multiplier": 0.0})

        if "patch_embed" in name:
            d.update({"lr_multiplier": d["lr_multiplier"] * patch_embed_lr_mult})

        all_param_groups.append(d)
        logger.debug(
            f"""{name}: lr_multiplier: {d["lr_multiplier"]}, wd_multiplier: {d["wd_multiplier"]}"""
        )

    return all_param_groups

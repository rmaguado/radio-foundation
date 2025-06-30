# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from typing import Tuple, Optional
import torch.nn as nn

from .vision_transformer import DinoVisionTransformer


logger = logging.getLogger("dinov2")


def build_model(args, only_teacher: bool) -> Tuple[nn.Module | None, nn.Module]:
    vit_kwargs = dict(
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        qkv_bias=args.qkv_bias,
        ffn_bias=args.ffn_bias,
        proj_bias=args.proj_bias,
        ffn_layer=args.ffn_layer,
        num_register_tokens=args.num_register_tokens,
        embed_configs=args.embed_layers,
        init_values=args.layerscale,
    )
    teacher = DinoVisionTransformer(**vit_kwargs)
    if only_teacher:
        return None, teacher
    student = DinoVisionTransformer(
        **vit_kwargs,
        drop_path_rate=args.drop_path_rate,
        drop_path_uniform=args.drop_path_uniform,
    )
    return student, teacher


def build_model_from_cfg(
    cfg, only_teacher: bool = False
) -> Tuple[nn.Module | None, nn.Module]:
    return build_model(cfg.student, only_teacher=only_teacher)

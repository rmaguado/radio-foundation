# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from . import vision_transformer as vits
from .vision_transformer import DinoVisionTransformer

logger = logging.getLogger("dinov2")


def build_model(args, img_size):
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=args.patch_size,
        in_chans=args.channels,
        init_values=args.layerscale,
        ffn_layer=args.ffn_layer,
        block_chunks=args.block_chunks,
        qkv_bias=args.qkv_bias,
        proj_bias=args.proj_bias,
        ffn_bias=args.ffn_bias,
        embed_layer=args.embed_layer,
        conv_channels=args.conv_channels,
        num_register_tokens=args.num_register_tokens,
        interpolate_offset=args.interpolate_offset,
        interpolate_antialias=args.interpolate_antialias,
    )
    teacher = vits.__dict__[args.arch](**vit_kwargs)
    student = vits.__dict__[args.arch](
        **vit_kwargs,
        drop_path_rate=args.drop_path_rate,
        drop_path_uniform=args.drop_path_uniform,
    )
    embed_dim = student.embed_dim

    return student, teacher, embed_dim


def build_model_from_cfg(cfg):
    return build_model(cfg.student, img_size=cfg.student.full_image_size)

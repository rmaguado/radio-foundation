# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
from functools import partial
from typing import Sequence, List, Dict, Tuple, Callable, Optional

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from einops import repeat

from dinov2.layers import (
    Mlp,
    PatchEmbed2D,
    PatchEmbed3D,
    SwiGLUFFNFused,
    MemEffAttention,
    NestedTensorBlock as Block,
)


logger = logging.getLogger("dinov2")

EMBED_LAYER_REGISTRY = {
    "patch_2d": PatchEmbed2D,
    "patch_3d": PatchEmbed3D,
}

FFN_LAYER_REGISTRY = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFNFused,
    "swiglufused": SwiGLUFFNFused,
    "identity": nn.Identity,
}


def get_embedding_layer(embed_config: Dict, embed_dim: int):
    """
    Creates a ModuleDict of embedding layers from a list of configurations.

    Args:
        embed_configs (Dict): Configuration dictionaries for embedding layers (2d and 3d).
        embed_dim (int): Output embedding dimension for each layer.

    Returns:
        nn.Module: Dictionary of embedding layers keyed by type.
    """
    layer_config = embed_config.copy()
    layer_type = layer_config["type"]

    patch_kwargs = {
        "img_size": layer_config.get("img_size", 224),
        "patch_size": layer_config.get("patch_size", 14),
        "in_channels": layer_config.get("in_channels", 1),
        "embed_dim": embed_dim,
        "layer_norm": layer_config.get("layer_norm", False),
    }

    return EMBED_LAYER_REGISTRY[layer_type](**patch_kwargs)


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    """
    Recursively applies a function to all submodules of a given module.

    Args:
        fn (Callable): Function to apply. Should accept (module, name) as arguments.
        module (nn.Module): The root module to traverse.
        name (str, optional): Name prefix for submodules.
        depth_first (bool): Whether to apply function in depth-first order.
        include_root (bool): Whether to include the root module itself.

    Returns:
        nn.Module: The original module (for chaining).
    """
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """
    Initializes weights for Vision Transformer modules using the timm (PyTorch Image Models) scheme.

    Args:
        module (nn.Module): Module to initialize.
        name (str, optional): Name of the module (unused).
    """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DinoVisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) backbone for DINO self-supervised learning.

    Supports both 2D and 3D patch embedding, register tokens, masking, and flexible FFN layers.

    Args:
        embed_dim (int): Embedding dimension for transformer.
        depth (int): Number of transformer blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embed dim.
        qkv_bias (bool): If True, add bias to QKV projections.
        ffn_bias (bool): If True, add bias to FFN layers.
        proj_bias (bool): If True, add bias to projection layers.
        ffn_layer (str): Type of feed-forward network layer to use.
        num_register_tokens (int): Number of register tokens to use.
        embed_configs (List[Dict]): List of embedding layer configurations.
        drop_path_rate (float): Drop path rate for stochastic depth.
        drop_path_uniform (bool): If True, use uniform drop path rate.
        init_values (Optional[float]): Initial value for LayerScale.
        act_layer (Callable): Activation function constructor.
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        ffn_bias: bool,
        proj_bias: bool,
        ffn_layer: str,
        num_register_tokens: int,
        embed_config: Dict,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = True,
        init_values: Optional[float] = None,
        act_layer: Callable = nn.GELU,
    ):
        super().__init__()
        block_fn = partial(Block, attn_class=MemEffAttention)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_register_tokens = num_register_tokens

        self.embed_layer = get_embedding_layer(
            embed_config=embed_config,
            embed_dim=embed_dim,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens > 0
            else None
        )

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        dpr = (
            [drop_path_rate] * depth
            if drop_path_uniform
            else [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        )

        try:
            ffn_layer_class = FFN_LAYER_REGISTRY[ffn_layer]
        except KeyError:
            raise NotImplementedError(f"FFN layer '{ffn_layer}' is not implemented.")

        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop_path=dpr[i],
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    act_layer=act_layer,
                    ffn_layer=ffn_layer_class,
                    init_values=init_values,
                    attn_class=MemEffAttention,
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Identity()

        self.init_weights()

    def init_weights(self):
        """
        Initializes all learnable parameters in the transformer, including tokens and embeddings.
        """
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.cls_pos_embed, std=0.02)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)

        trunc_normal_(self.embed_layer.pos_embed, std=0.02)

        named_apply(init_weights_vit_timm, self)

    def _prepare_tokens(
        self, x: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Handles patch embedding, token masking, and concatenation of CLS and register tokens.

        Args:
            x (torch.Tensor): Input image tensor.
            masks (Optional[torch.Tensor]): Optional mask tensor for patch masking.

        Returns:
            torch.Tensor: Embedded and tokenized input ready for transformer blocks.
        """
        B = x.shape[0]

        x = self.embed_layer(x)

        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
            )

        cls_tokens = repeat(self.cls_token + self.cls_pos_embed, "1 1 e -> b 1 e", b=B)
        x = torch.cat([cls_tokens, x], dim=1)

        if self.register_tokens is not None:
            register_tokens = repeat(self.register_tokens, "1 n e -> b n e", b=B)
            x = torch.cat([x[:, :1, :], register_tokens, x[:, 1:, :]], dim=1)

        return x

    def forward(
        self,
        x: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = self._prepare_tokens(x, masks)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        return {
            "clstoken": x_norm[:, 0],
            "patchtokens": x_norm[:, self.num_register_tokens + 1 :],
        }

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        select_layers: Sequence[int] = (11,),
        norm: bool = True,
    ) -> Dict[str, torch.Tensor]:
        x = self._prepare_tokens(x)

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in select_layers:
                layer_output = self.norm(x) if norm else x
                outputs.append(layer_output)

        assert len(outputs) == len(
            select_layers
        ), f"Found {len(outputs)}/{len(select_layers)} layers."

        return {
            "clstoken": torch.stack([out[:, 0] for out in outputs]),
            "patchtokens": torch.stack(
                [out[:, 1 + self.num_register_tokens :] for out in outputs]
            ),
        }


def build_model(cfg, teacher_only=False):
    args = cfg.student
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
        embed_config=args.embed_layer,
        init_values=args.layerscale,
    )
    teacher = DinoVisionTransformer(**vit_kwargs)
    if teacher_only:
        return None, teacher
    student = DinoVisionTransformer(
        **vit_kwargs,
        drop_path_rate=args.drop_path_rate,
        drop_path_uniform=args.drop_path_uniform,
    )
    return student, teacher

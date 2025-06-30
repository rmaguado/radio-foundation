# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import logging
from typing import Sequence, Callable, Dict, List, Optional

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
    "patch2d": PatchEmbed2D,
    "patch3d": PatchEmbed3D,
}

FFN_LAYER_REGISTRY = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFNFused,
    "swiglufused": SwiGLUFFNFused,
    "identity": nn.Identity,
}


def get_embedding_layers(
    embed_configs: List[Dict], embed_dim: int, norm_layer: Callable
) -> nn.ModuleDict:
    """
    Creates a ModuleDict of embedding layers from a list of configurations.
    """
    embed_layers = nn.ModuleDict()
    for config in embed_configs:
        layer_type = config["type"]
        if layer_type not in EMBED_LAYER_REGISTRY:
            raise NotImplementedError(
                f"Embedding layer type '{layer_type}' is not implemented. Available: {list(EMBED_LAYER_REGISTRY.keys())}"
            )

        patch_kwargs = {
            "img_size": config.get("img_size", 224),
            "patch_size": config.get("patch_size", 16),
            "in_channels": config.get("in_channels", 1),
            "embed_dim": embed_dim,
            "norm_layer": norm_layer,
        }
        embed_layers[layer_type] = EMBED_LAYER_REGISTRY[layer_type](**patch_kwargs)

    return embed_layers


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
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
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        ffn_bias: bool,
        proj_bias: bool,
        drop_path_rate: float,
        drop_path_uniform: bool,
        ffn_layer: str,
        num_register_tokens: int,
        embed_configs: List[Dict],
        init_values: Optional[float] = None,
        act_layer: Callable = nn.GELU,
        block_fn: Callable = Block,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_register_tokens = num_register_tokens

        self.embed_layers = get_embedding_layers(
            embed_configs=embed_configs,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
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
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    ffn_layer=ffn_layer_class,
                    init_values=init_values,
                    attn_class=MemEffAttention,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.cls_pos_embed, std=0.02)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)

        for embed_layer in self.embed_layers.values():
            trunc_normal_(embed_layer.pos_embed, std=0.02)

        named_apply(init_weights_vit_timm, self)

    def _prepare_tokens(
        self, x: torch.Tensor, embed_layer: str, masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Handles patch embedding, token masking, and concatenation of CLS and register tokens.
        """
        B = x.shape[0]

        x = self.embed_layers[embed_layer](x)

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
        self, x: torch.Tensor, embed_layer: str = "patch2d", masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        x = self._prepare_tokens(x, embed_layer, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)

        return {
            "clstoken": x_norm[:, 0],
            "regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "patchtokens": x_norm[:, self.num_register_tokens + 1 :],
        }

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        embed_layer: str = "patch2d",
        select_layers: Sequence[int] = (11,),
        norm: bool = True,
    ) -> Dict[str, List[torch.Tensor]]:
        x = self._prepare_tokens(x, embed_layer)

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
            "clstoken": [out[:, 0] for out in outputs],
            "patchtokens": [out[:, 1 + self.num_register_tokens :] for out in outputs],
        }

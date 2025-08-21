# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
from functools import partial
from multiprocessing import Value
from typing import Sequence, List, Dict, Tuple, Callable, Optional

import torch
import torch.nn as nn

from einops import repeat

from dino.layers import (
    Mlp,
    SwiGLUFFN,
    LayerScale,
    PatchEmbed,
    RMSNorm,
    SelfAttentionBlock,
    LearnedPositionEmbedding,
    RopePositionEmbedding,
)


logger = logging.getLogger("dino")

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
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

    patch_kwargs = {
        "img_size": layer_config.get("img_size", 224),
        "ndims": layer_config.get("ndims", 2),
        "patch_size": layer_config.get("patch_size", 14),
        "in_channels": layer_config.get("in_channels", 1),
        "embed_dim": embed_dim,
        "layer_norm": layer_config.get("layer_norm", False),
    }

    return PatchEmbed(**patch_kwargs)


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


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        n_blocks: int,
        num_heads: int,
        ffn_ratio: int,
        img_size: int,
        patch_size: int,
        ndims: int,
        in_channels: int,
        pos_embed_type: str,
        rope_base: float,
        rope_shift_coords: Optional[float],
        rope_jitter_coords: Optional[float],
        rope_rescale_coords: Optional[float],
        drop_path_rate: float,
        layerscale_init: float,
        norm_layer: str,
        ffn_layer: str,
        qkv_bias: bool,
        proj_bias: bool,
        ffn_bias: bool,
        num_register_tokens: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.embed_dim = embed_dim
        self.n_blocks = n_blocks
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.num_register_tokens = num_register_tokens

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            ndims=ndims,
            patch_size=patch_size,
            embed_dim=embed_dim,
            in_channels=in_channels,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.register_tokens = (
            nn.Parameter(torch.empty(1, num_register_tokens, embed_dim, device=device))
            if num_register_tokens > 0
            else None
        )
        self.pos_embed_type = pos_embed_type
        if pos_embed_type == "rope":
            self.pos_embed = RopePositionEmbedding(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ndims=ndims,
                base=rope_base,
                shift_coords=rope_shift_coords,
                jitter_coords=rope_jitter_coords,
                rescale_coords=rope_rescale_coords,
                dtype=dtype,
                device=device,
            )
        elif pos_embed_type == "learned":
            num_patches = (img_size // patch_size) ** ndims
            self.pos_embed = LearnedPositionEmbedding(
                embed_dim=embed_dim, num_patches=num_patches, ndims=ndims
            )

        else:
            raise ValueError

        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer_cls,
                    act_layer=nn.GELU,
                    ffn_layer=ffn_layer_cls,
                    init_values=layerscale_init,
                    device=device,
                )
                for i in range(n_blocks)
            ]
        )
        self.norm = norm_layer_cls(embed_dim)

        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

    def init_weights(self):
        """
        Initializes all learnable parameters in the transformer, including tokens and embeddings.
        """

        self.pos_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def _prepare_tokens(
        self, x: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Size]:
        """
        Handles patch embedding, token masking, and concatenation of CLS and register tokens.

        Args:
            x (torch.Tensor): Input image tensor.
            masks (Optional[torch.Tensor]): Optional mask tensor for patch masking.

        Returns:
            torch.Tensor: Embedded and tokenized input ready for transformer blocks.
        """
        B = x.shape[0]

        x, patch_dims = self.patch_embed(x)

        if self.pos_embed_type == "learned":
            x += self.pos_embed(*patch_dims)

        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
            )

        cls_tokens = repeat(self.cls_token, "1 1 e -> b 1 e", b=B)

        if self.register_tokens is not None:
            register_tokens = repeat(self.register_tokens, "1 n e -> b n e", b=B)
            x = torch.cat([cls_tokens, register_tokens, x], dim=1)
        else:
            x = torch.cat(tensors=[cls_tokens, x], dim=1)

        return x, patch_dims

    def _get_rope(self, *patch_dims):
        if self.pos_embed_type == "rope":
            return self.pos_embed(*patch_dims)
        return None

    def forward(
        self,
        x: torch.Tensor,
        *,
        masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x, patch_dims = self._prepare_tokens(x, masks)

        for blk in self.blocks:
            rope_sincos = self._get_rope(*patch_dims)
            x = blk(x, rope_sincos)

        x_norm = self.norm(x)
        x_norm_cls = x_norm[:, 0]
        x_norm_patch = x_norm[:, self.num_register_tokens + 1 :]

        return {
            "clstoken": x_norm_cls,
            "patchtokens": x_norm_patch,
        }


def build_models(cfg):
    """
    Initialize two models (student and teacher) using config.
    Filters out `resume_from_teacher_chkpt` from config.
    """
    args = cfg.student.copy()
    drop_path_rate = args.pop("drop_path_rate", 0.0)
    _ = args.pop("resume_from_teacher_chkpt")

    teacher = DinoVisionTransformer(**args, drop_path_rate=0.0)
    student = DinoVisionTransformer(
        **args,
        drop_path_rate=drop_path_rate,
    )
    return student, teacher


def build_model_eval(cfg):
    args = cfg.student.copy()
    _ = args.pop("drop_path_rate", 0.0)

    teacher = DinoVisionTransformer(**args, drop_path_rate=0.0)
    return teacher

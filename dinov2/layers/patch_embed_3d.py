# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py
#
# Modified from original patch embedding to apply 3D convolutions

from typing import Callable, Optional, Tuple, Union

from torch import Tensor
import torch.nn as nn
from einops import rearrange


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed3D(nn.Module):
    """
    3D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 10,
        embed_dim: int = 768,
        img_depth: int = 300,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_DHW = (in_chans, image_HW[0], image_HW[1])
        patch_grid_size = (
            img_depth // in_chans,
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )
        assert (
            img_depth % in_chans == 0
        ), "img_depth must be divisible by in_chans when using patch embed_3d"

        self.img_size = image_HW
        self.patch_size = patch_DHW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1] * patch_grid_size[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv3d(1, embed_dim, kernel_size=patch_CHW, stride=patch_CHW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, D, H, W = x.shape
        patch_D, patch_H, patch_W = self.patch_size

        assert (
            D % patch_D == 0
        ), f"Input image depth {D} is not a multiple of patch channel: {patch_D}"
        assert (
            H % patch_H == 0
        ), f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert (
            W % patch_W == 0
        ), f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = x.unsqueeze(1)  # B 1 D H W
        x = self.proj(x)  # B C d h w
        d, h, w = x.size(2), x.size(3), x.size(4)
        x = rearrange(x, "b c d h w -> b (d h w) c")  # B dhw C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = rearrange(
                x, "b (d h w) c -> b d h w c", d=d, h=h, w=w, c=self.embed_dim
            )
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = (
            Ho
            * Wo
            * self.embed_dim
            * (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        )
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

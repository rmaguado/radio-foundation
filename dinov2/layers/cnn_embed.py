# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from typing import Callable, Optional, Tuple, Union

from torch import Tensor
import torch.nn as nn


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class CNNEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        conv_channels: Number of convolutional channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        conv_channels: int = 64,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.patch_size = make_2tuple(patch_size)
        self.proj_kernel = self.patch_size[0] // 4

        self.in_chans = in_chans
        self.conv_channels = conv_channels
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(in_chans, conv_channels, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(conv_channels, embed_dim, kernel_size=3, stride=2)
        self.proj = nn.Conv2d(
            conv_channels,
            embed_dim,
            kernel_size=self.proj_kernel,
            stride=self.proj_kernel,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        B, _, H, W = x.shape
        pH, pW = self.patch_size
        num_patches = (H // pH) * (W // pW)

        assert (
            H % pH == 0
        ), f"Input image height {H} is not a multiple of patch height {pH}"
        assert (
            W % pW == 0
        ), f"Input image width {W} is not a multiple of patch width: {pW}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)

        x = x.unfold(2, pH, pH).unfold(3, pW, pW)  # B C H W pH pW
        x = x.flatten(2, 3).transpose(1, 2)  # B HW C pH pW
        x = x.flatten(0, 1)  # B*HW C pH pW

        x = self.conv1(x)  # B*HW H pH/2 pW/2
        x = self.conv2(x)  # B*HW H pH/4 pW/4
        x = self.proj(x)
        x = x.flatten(2, 3)

        x = self.norm(x)

        x = x.view(B, num_patches, self.embed_dim)

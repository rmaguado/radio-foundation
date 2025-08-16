# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from einops import rearrange


class LearnedPositionEmbedding(nn.Module):
    def __init__(self, *, embed_dim, num_patches, ndims=3):
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.ndims = ndims

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        if ndims == 2:
            self.interpolate_fnc = self._get_pos_embed_2d
        elif ndims == 3:
            self.interpolate_fnc = self._get_pos_embed_3d
        else:
            raise ValueError(f"`num_dims` must be 2 or 3.")

    def _get_pos_embed_2d(self, *patch_dims: int) -> torch.Tensor:
        H, W = patch_dims
        num_patches = H * W
        if num_patches == self.num_patches:
            return self.pos_embed

        pos_embed = self.pos_embed[0]
        orig_size = int(self.num_patches**0.5)
        pos_embed_2d = rearrange(
            pos_embed, "(h w) d -> 1 d h w", h=orig_size, w=orig_size
        )
        pos_embed_2d = F.interpolate(
            pos_embed_2d, size=(H, W), mode="bicubic", align_corners=False
        )
        pos_embed_2d = rearrange(pos_embed_2d, "1 d h w -> 1 (h w) d")
        return pos_embed_2d

    def _get_pos_embed_3d(self, *patch_dims: int) -> torch.Tensor:
        D, H, W = patch_dims
        num_patches = D * H * W
        if num_patches == self.num_patches:
            return self.pos_embed
        pos_embed = self.pos_embed[0]
        orig_size = int(round(self.num_patches ** (1 / 3)))
        pos_embed_3d = rearrange(
            pos_embed, "(d h w) c -> 1 c d h w", d=orig_size, h=orig_size, w=orig_size
        )
        pos_embed_3d = F.interpolate(
            pos_embed_3d, size=(D, H, W), mode="trilinear", align_corners=False
        )
        pos_embed_3d = rearrange(pos_embed_3d, "1 c d h w -> 1 (d h w) c")
        return pos_embed_3d

    def forward(self, *patch_dims: int):
        return self.interpolate_fnc(*patch_dims)


class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        ndims: int = 2,
        base: float = 100.0,
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert (
            embed_dim % (2 * ndims * num_heads) == 0
        ), f"embed_dim must be divisible by 2*ndims*num_heads, got {embed_dim=}, {ndims=}, {num_heads=}"

        D_head = embed_dim // num_heads
        self.ndims = ndims
        self.base = base
        self.D_head = D_head
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // (2 * ndims), device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *patch_dims: int) -> tuple[Tensor, Tensor]:
        device, dtype = self.periods.device, self.dtype
        dd = {"device": device, "dtype": dtype}

        axes = [torch.arange(0.5, s, **dd) / s for s in patch_dims]
        coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        coords = coords.flatten(0, -2)
        coords = 2.0 * coords - 1.0

        if self.training and self.shift_coords is not None:
            shift = torch.empty(self.ndims, **dd).uniform_(
                -self.shift_coords, self.shift_coords
            )
            coords += shift[None, :]

        if self.training and self.jitter_coords is not None:
            jitter = (
                torch.empty(self.ndims, **dd)
                .uniform_(-np.log(self.jitter_coords), np.log(self.jitter_coords))
                .exp()
            )
            coords *= jitter[None, :]

        if self.training and self.rescale_coords is not None:
            rescale = (
                torch.empty(1, **dd)
                .uniform_(-np.log(self.rescale_coords), np.log(self.rescale_coords))
                .exp()
            )
            coords *= rescale

        angles = (
            2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # type: ignore
        )  # [N, ndims, D//(2*ndims)]
        angles = angles.flatten(1, 2)  # [N, D//2]
        angles = angles.tile(2)  # [N, D]
        cos, sin = torch.cos(angles), torch.sin(angles)
        return sin, cos

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        periods = self.base ** (
            2
            * torch.arange(self.D_head // (2 * self.ndims), device=device, dtype=dtype)  # type: ignore
            / (self.D_head // self.ndims)
        )
        self.periods.data = periods

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
    def __init__(self, *, embed_dim, num_patches, num_dims=3):
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.num_dims = num_dims

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        if num_dims == 2:
            self.interpolate_fnc = self._get_pos_embed_2d
        elif num_dims == 3:
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


# RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights
# Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`.
class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        ndim: int = 2,
        base: float = 100.0,
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert (
            embed_dim % (2 * ndim * num_heads) == 0
        ), f"embed_dim must be divisible by 2*ndim*num_heads, got {embed_dim=}, {ndim=}, {num_heads=}"

        D_head = embed_dim // num_heads
        self.base = base
        self.D_head = D_head
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        coords_h = torch.arange(0.5, H, **dd) / H  # [H]
        coords_w = torch.arange(0.5, W, **dd) / W  # [W]

        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1
        )  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(
                -self.shift_coords, self.shift_coords
            )
            coords += shift_hw[None, :]

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_hw[None, :]

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale_hw

        # Prepare angles and sin/cos
        angles = (
            2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # type: ignore
        )  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        periods = self.base ** (
            2
            * torch.arange(self.D_head // 4, device=device, dtype=dtype)  # type: ignore
            / (self.D_head // 2)
        )  # [D//4]

        self.periods.data = periods

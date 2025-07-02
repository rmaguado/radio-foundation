from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        in_channels: int = 1,
        norm_layer: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        self.patches_resolution = img_size // patch_size

        self.proj = self._get_projection_layer()
        self.num_patches = self._calculate_num_patches()
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def _get_projection_layer(self) -> nn.Module:
        """Return the appropriate projection layer (e.g., nn.Conv2d, nn.Conv3d)."""
        raise NotImplementedError

    def _calculate_num_patches(self) -> int:
        """Calculate the total number of patches based on dimensions."""
        raise NotImplementedError

    def _rearrange_projection(self, x: torch.Tensor) -> torch.Tensor:
        """Rearrange the projected tensor from (B, E, Dims...) to (B, N, E)."""
        raise NotImplementedError

    def get_pos_embed(self, *input_dims: int) -> torch.Tensor:
        """
        Interpolate positional embeddings to match the input's spatial resolution.
        `input_dims` should be (H, W) for 2D and (D, H, W) for 3D.
        """
        if self.pos_embed.shape[1] == self.num_patches:
            return self.pos_embed

        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, *spatial_dims = x.shape

        x = self.proj(x)
        x = self._rearrange_projection(x)
        x = self.norm(x)

        x = x + self.get_pos_embed(*spatial_dims)

        return x


class PatchEmbed2D(PatchEmbed):
    """2D Image to Patch Embedding."""

    def _get_projection_layer(self) -> nn.Module:
        return nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def _calculate_num_patches(self) -> int:
        return self.patches_resolution * self.patches_resolution

    def _rearrange_projection(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b e h w -> b (h w) e")

    def get_pos_embed(self, *input_dims: int) -> torch.Tensor:
        """
        Interpolate positional embeddings to match the input's spatial resolution.
        `input_dims` should be (H, W).
        """
        H, W = input_dims
        num_patches = H * W
        if num_patches == self.num_patches:
            return self.pos_embed
        # Interpolate
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


class PatchEmbed3D(PatchEmbed):
    """3D Volume to Patch Embedding."""

    def _get_projection_layer(self) -> nn.Module:
        return nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def _calculate_num_patches(self) -> int:
        return (
            self.patches_resolution * self.patches_resolution * self.patches_resolution
        )

    def _rearrange_projection(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b e d h w -> b (d h w) e")

    def get_pos_embed(self, *input_dims: int) -> torch.Tensor:
        """
        Interpolate positional embeddings to match the input's spatial resolution.
        `input_dims` should be (D, H, W).
        """
        D, H, W = input_dims
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

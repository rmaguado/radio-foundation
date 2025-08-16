import torch
import torch.nn as nn
import math
from typing import Tuple, Callable
from einops import rearrange


class PatchEmbed(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        ndims: int = 2,
        patch_size: int = 16,
        embed_dim: int = 768,
        in_channels: int = 1,
        norm_layer: Callable | None = None,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.ndims = ndims
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        if ndims == 3:
            assert (
                in_channels == 1
            ), f"For 3d patch embed, `in_channels` must be set to 1."

        self.patches_resolution = img_size // patch_size

        if ndims == 2:
            self.proj = nn.Conv2d(
                self.in_channels,
                self.embed_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
        elif ndims == 3:
            self.proj = nn.Conv3d(
                self.in_channels,
                self.embed_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
        else:
            raise ValueError(f"`ndims` must be 2 or 3, got {ndims}.")

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
        x = self.proj(x)

        patch_dims = x.shape[2:]

        x = rearrange(x, "b e ... -> b (...) e")
        x = self.norm(x)

        return x, patch_dims

    def reset_parameters(self):
        if self.ndims == 2:
            k = 1 / (self.in_channels * (self.patch_size**2))
        elif self.ndims == 3:
            k = 1 / (self.patch_size**3)
        else:
            raise ValueError(f"`ndims` must be 2 or 3, got {self.ndims}.")

        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))

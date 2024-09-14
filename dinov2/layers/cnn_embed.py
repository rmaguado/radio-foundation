from typing import Tuple, Union

from torch import Tensor
import torch.nn as nn


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class CnnEmbed(nn.Module):
    """
    CNN head for a hybrid ViT.

    Args:
        patch_size (int or tuple): patch size for the CNN head.
        in_chans (int): number of input channels.
        conv_chans (int): number of output channels for the first conv layer.
        embed_dim (int): number of output channels for the second
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: Union[int, Tuple[int, int]] = 14,
        in_chans: int = 3,
        embed_dim: int = 384,
        conv_channels: int = 96,
    ) -> None:
        super().__init__()
        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        embed_kernel_size = patch_size // 2

        self.feature_layer = nn.Conv2d(
            in_chans, conv_channels, kernel_size=5, stride=2, padding=2
        )

        self.embed_layer = nn.Conv2d(
            conv_channels,
            embed_dim,
            kernel_size=embed_kernel_size,
            stride=embed_kernel_size,
        )

        self.ln = nn.LayerNorm(embed_dim)
        self.gelu = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert (
            H % patch_H == 0
        ), f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert (
            W % patch_W == 0
        ), f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.feature_layer(x)
        x = self.embed_layer(x)

        x = x.transpose(1, 3)
        x = self.ln(x)
        x = x.transpose(1, 3)

        x = self.gelu(x)

        return x.flatten(2).transpose(1, 2)

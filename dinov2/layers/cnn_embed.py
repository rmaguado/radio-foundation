from typing import Tuple, Union

from torch import Tensor
import torch.nn as nn


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
        conv_chans: int = 96,
    ) -> None:
        super().__init__()
        embed_kernel_size = patch_size // 2

        self.feature_layer = nn.Conv2d(
            in_chans, conv_chans, kernel_size=5, stride=2, padding=2
        )

        self.embed_layer = nn.Conv2d(
            conv_chans,
            embed_dim,
            kernel_size=embed_kernel_size,
            stride=embed_kernel_size,
        )

        self.ln = nn.LayerNorm(conv_chans)
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

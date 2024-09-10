from typing import Tuple, Union

from torch import Tensor
import torch.nn as nn


class CNNHead(nn.Module):
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
    ) -> None:
        super().__init__()
        bottleneck_dim = embed_dim // 4
        embed_kernel_size = patch_size // 2

        self.stem_layer = nn.Conv2d(
            in_chans, bottleneck_dim, kernel_size=5, stride=2, padding=2
        )

        self.depthwise_conv = nn.Conv2d(
            bottleneck_dim,
            bottleneck_dim,
            kernel_size=7,
            padding="same",
            groups=bottleneck_dim,
        )

        self.ln = nn.LayerNorm(bottleneck_dim)
        self.gelu = nn.GELU()

        self.conv1x1_384 = nn.Conv2d(bottleneck_dim, embed_dim, kernel_size=1)
        self.conv1x1_96 = nn.Conv2d(embed_dim, bottleneck_dim, kernel_size=1)

        self.embed_layer = nn.Conv2d(
            bottleneck_dim,
            embed_dim,
            kernel_size=embed_kernel_size,
            stride=embed_kernel_size,
        )

    def forward(self, x: Tensor) -> Tensor:

        x = self.stem_layer(x)

        residual = x
        x = self.depthwise_conv(x)

        x = x.transpose(1, 3)
        x = self.ln(x)
        x = x.transpose(1, 3)

        x = self.conv1x1_384(x)
        x = self.gelu(x)

        x = self.conv1x1_96(x)
        x += residual

        x = self.embed_layer(x)

        return x.flatten(2).transpose(1, 2)

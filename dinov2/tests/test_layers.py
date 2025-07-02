import torch
import pytest
from dinov2.layers.block import Block
from dinov2.layers.attention import Attention
from dinov2.layers.dino_head import DINOHead
from dinov2.layers.patch_embed import PatchEmbed2D, PatchEmbed3D
from dinov2.layers.swiglu_ffn import SwiGLUFFN, SwiGLUFFNFused
from dinov2.layers.drop_path import DropPath
from dinov2.layers.layer_scale import LayerScale
from dinov2.layers.mlp import Mlp


def test_block_forward():
    block = Block(dim=8, num_heads=2)
    x = torch.randn(2, 4, 8)
    out = block(x)
    assert out.shape == x.shape


def test_attention_forward():
    attn = Attention(dim=8, num_heads=2)
    x = torch.randn(2, 4, 8)
    out = attn(x)
    assert out.shape == x.shape


def test_dino_head_forward():
    head = DINOHead(in_dim=8, out_dim=4, nlayers=2, hidden_dim=16, bottleneck_dim=8)
    x = torch.randn(2, 8)
    out = head(x)
    assert out.shape == (2, 4)


def test_patch_embed2d_forward():
    patch = PatchEmbed2D(img_size=32, patch_size=8, embed_dim=8, in_channels=1)
    x = torch.randn(2, 1, 32, 32)
    out = patch(x)
    assert out.shape[0] == 2
    assert out.shape[2] == 8


def test_patch_embed3d_forward():
    patch = PatchEmbed3D(img_size=16, patch_size=8, embed_dim=8, in_channels=1)
    x = torch.randn(2, 1, 16, 16, 16)
    out = patch(x)
    assert out.shape[0] == 2
    assert out.shape[2] == 8


def test_swigluffn_forward():
    ffn = SwiGLUFFN(in_features=8, hidden_features=16, out_features=8)
    x = torch.randn(2, 4, 8)
    out = ffn(x)
    assert out.shape == (2, 4, 8)


def test_swigluffnfused_forward():
    ffn = SwiGLUFFNFused(in_features=8, hidden_features=16, out_features=8)
    x = torch.randn(2, 4, 8)
    out = ffn(x)
    assert out.shape == (2, 4, 8)


def test_drop_path_forward():
    dp = DropPath(drop_prob=0.5)
    x = torch.randn(2, 4, 8)
    dp.train()
    out = dp(x)
    assert out.shape == x.shape


def test_layer_scale_forward():
    ls = LayerScale(dim=8)
    x = torch.randn(2, 4, 8)
    out = ls(x)
    assert out.shape == x.shape


def test_mlp_forward():
    mlp = Mlp(in_features=8, hidden_features=16, out_features=8)
    x = torch.randn(2, 4, 8)
    out = mlp(x)
    assert out.shape == (2, 4, 8)

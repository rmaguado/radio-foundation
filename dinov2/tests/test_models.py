import torch
import pytest
from dinov2.models.vision_transformer import DinoVisionTransformer
from dinov2.models import build_model, build_model_from_cfg
from omegaconf import OmegaConf
import os


@pytest.fixture
def minimal_cfg():
    from dinov2.configs import dinov2_default_config

    minimal_model_path = os.path.join(
        os.path.dirname(__file__), "configs/minimal_model.yaml"
    )
    minimal_model_cfg = OmegaConf.load(minimal_model_path)
    merged_cfg = OmegaConf.merge(dinov2_default_config, minimal_model_cfg)
    return merged_cfg


def test_dinovisiontransformer_forward(minimal_cfg):
    model = DinoVisionTransformer(
        embed_dim=minimal_cfg.embed_dim,
        depth=minimal_cfg.depth,
        num_heads=minimal_cfg.num_heads,
        mlp_ratio=minimal_cfg.mlp_ratio,
        qkv_bias=minimal_cfg.qkv_bias,
        ffn_bias=minimal_cfg.ffn_bias,
        proj_bias=minimal_cfg.proj_bias,
        ffn_layer=minimal_cfg.ffn_layer,
        num_register_tokens=minimal_cfg.num_register_tokens,
        embed_configs=minimal_cfg.embed_layers,
        drop_path_rate=minimal_cfg.drop_path_rate,
        drop_path_uniform=minimal_cfg.drop_path_uniform,
        init_values=minimal_cfg.layerscale,
    )
    x = torch.randn(2, 1, 16, 16)
    out = model(x, embed_layer="patch2d")
    assert "clstoken" in out
    assert out["clstoken"].shape[0] == 2


def test_build_model(minimal_cfg):
    student, teacher = build_model(minimal_cfg, only_teacher=False)
    assert isinstance(student, DinoVisionTransformer)
    assert isinstance(teacher, DinoVisionTransformer)
    # Only teacher
    _, teacher_only = build_model(minimal_cfg, only_teacher=True)
    assert isinstance(teacher_only, DinoVisionTransformer)


def test_build_model_from_cfg(minimal_cfg):
    student, teacher = build_model_from_cfg(minimal_cfg, only_teacher=False)
    assert isinstance(student, DinoVisionTransformer)
    assert isinstance(teacher, DinoVisionTransformer)

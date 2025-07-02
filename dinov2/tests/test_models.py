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
    student, teacher = build_model_from_cfg(minimal_cfg, only_teacher=False)
    x = torch.randn(2, 1, 16, 16)
    out = teacher(x, embed_layer="patch2d")
    assert "clstoken" in out
    assert out["clstoken"].shape[0] == 2

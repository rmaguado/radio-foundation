import pytest
import torch
from types import SimpleNamespace
from omegaconf import OmegaConf
import os
from dinov2.train.ssl_meta_arch import SSLMetaArch


class DummyBackbone(torch.nn.Module):
    def __init__(self, embed_dim=8):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, embed_layer=None, masks=None):
        B = x.shape[0]
        return {
            "clstoken": torch.zeros(B, self.embed_dim),
            "patchtokens": torch.zeros(B, 2, self.embed_dim),
        }


class DummyHead(torch.nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x):
        return torch.zeros(x.shape[0], self.out_dim)


class DummyLoss:
    def __init__(self, out_dim=None):
        pass

    def __call__(self, *args, **kwargs):
        return torch.tensor(0.0)

    def softmax_center_teacher(self, x, temp):
        return x


@pytest.fixture
def minimal_cfg(monkeypatch):
    # Patch build_model_from_cfg and DINOHead, DINOLoss, KoLeoLoss, iBOTPatchLoss
    from dinov2 import models, layers, loss
    from dinov2.configs import dinov2_default_config

    monkeypatch.setattr(
        models,
        "build_model_from_cfg",
        lambda cfg, only_teacher=False: (DummyBackbone(), DummyBackbone()),
    )
    monkeypatch.setattr(layers, "DINOHead", DummyHead)
    monkeypatch.setattr(loss, "DINOLoss", DummyLoss)
    monkeypatch.setattr(loss, "KoLeoLoss", DummyLoss)
    monkeypatch.setattr(loss, "iBOTPatchLoss", DummyLoss)
    # Load and merge configs
    minimal_model_path = os.path.join(
        os.path.dirname(__file__), "configs/minimal_model.yaml"
    )
    minimal_model_cfg = OmegaConf.load(minimal_model_path)
    merged_cfg = OmegaConf.merge(dinov2_default_config, minimal_model_cfg)

    # Convert to SimpleNamespace recursively
    def to_ns(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: to_ns(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            return [to_ns(i) for i in obj]
        else:
            return obj

    cfg_ns = to_ns(OmegaConf.to_container(merged_cfg, resolve=True))
    return cfg_ns


def test_ssl_meta_arch_init(minimal_cfg):
    arch = SSLMetaArch(minimal_cfg)
    assert isinstance(arch, SSLMetaArch)
    assert hasattr(arch, "student")
    assert hasattr(arch, "teacher")


def test_ssl_meta_arch_forward(minimal_cfg):
    arch = SSLMetaArch(minimal_cfg)
    # Dummy collated_views
    collated_views = {
        "g1": {
            "images": torch.zeros(2, 1, 8, 8),
            "masks": torch.zeros(2, 1, 8, 8, dtype=torch.bool),
            "embed_layer": "patch2d",
            "is_target": True,
            "targets": [],
        }
    }
    teacher_temp = 1.0
    loss, loss_dict = arch.forward(collated_views, teacher_temp)
    assert isinstance(loss, torch.Tensor)
    assert "dino_loss" in loss_dict
    assert "total_loss" in loss_dict


def test_ssl_meta_arch_init_error(monkeypatch):
    # Patch build_model_from_cfg to return None for student
    from dinov2 import models

    monkeypatch.setattr(
        models,
        "build_model_from_cfg",
        lambda cfg, only_teacher=False: (None, DummyBackbone()),
    )
    cfg = SimpleNamespace(
        student=SimpleNamespace(pretrained_weights=None, embed_dim=8),
        dino=SimpleNamespace(
            head_n_prototypes=4,
            head_hidden_dim=8,
            head_bottleneck_dim=4,
            head_nlayers=1,
            koleo_loss_weight=0.0,
            loss_weight=1.0,
        ),
        ibot=SimpleNamespace(loss_weight=0.0, separate_head=False),
        crops=SimpleNamespace(
            crop_groups=[{"name": "g1", "is_target": True, "embed_layer": "patch2d"}]
        ),
        optim=SimpleNamespace(layerwise_decay=1.0, patch_embed_lr_mult=1.0),
    )
    with pytest.raises(ValueError):
        SSLMetaArch(cfg)


def test_train_placeholder():
    # TODO: Implement tests for train/train.py
    assert True


def test_setup_placeholder():
    # TODO: Implement tests for train/setup.py
    assert True


def test_utils_placeholder():
    # TODO: Implement tests for train/utils.py
    assert True


def test_parser_placeholder():
    # TODO: Implement tests for train/parser.py
    assert True

import pytest
import torch
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
    minimal_model_path = os.path.join(
        os.path.dirname(__file__), "configs/minimal_model.yaml"
    )
    minimal_model_cfg = OmegaConf.load(minimal_model_path)
    merged_cfg = OmegaConf.merge(dinov2_default_config, minimal_model_cfg)
    return merged_cfg


def test_ssl_meta_arch_init(minimal_cfg):
    arch = SSLMetaArch(minimal_cfg)
    assert isinstance(arch, SSLMetaArch)
    assert hasattr(arch, "student")
    assert hasattr(arch, "teacher")


def test_ssl_meta_arch_forward(minimal_cfg):
    arch = SSLMetaArch(minimal_cfg)
    collated_views = {
        "2d_global_hd": {
            "images": torch.zeros(8, 3, 1, 224, 224),
            "masks": torch.zeros(8, 3, 16, 16, dtype=torch.bool),
            "embed_layer": "patch2d",
            "is_target": True,
            "targets": ["2d_global_hd"],
        },
        "2d_local_hd": {
            "images": torch.zeros(8, 3, 2, 1, 112, 112),
            "masks": torch.zeros(8, 3, 8, 8, dtype=torch.bool),
            "embed_layer": "patch2d",
            "is_target": False,
            "targets": ["2d_global_hd"],
        },
    }
    teacher_temp = 1.0
    loss, loss_dict = arch.forward(collated_views, teacher_temp)
    assert isinstance(loss, torch.Tensor)
    assert "dino_loss" in loss_dict
    assert "total_loss" in loss_dict


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


def test_ssl_meta_arch_intermediate_steps(minimal_cfg):
    arch = SSLMetaArch(minimal_cfg)
    collated_views = {
        "2d_global_hd": {
            "images": torch.zeros(2, 1, 224, 224),
            "masks": torch.zeros(2, 1, 16, 16, dtype=torch.bool),
            "embed_layer": "patch2d",
            "is_target": True,
            "targets": ["2d_global_hd"],
        },
        "2d_local_hd": {
            "images": torch.zeros(2, 1, 112, 112),
            "masks": torch.zeros(2, 1, 8, 8, dtype=torch.bool),
            "embed_layer": "patch2d",
            "is_target": False,
            "targets": ["2d_global_hd"],
        },
    }
    teacher_temp = 1.0

    # _prepare_inputs
    images, masks = arch._prepare_inputs(collated_views)
    assert set(images.keys()) == set(collated_views.keys())
    assert set(masks.keys()) == set(collated_views.keys())
    for k in images:
        assert isinstance(images[k], torch.Tensor)
        assert isinstance(masks[k], torch.Tensor)

    # _run_teacher_pass
    teacher_outputs = arch._run_teacher_pass(
        images, masks, collated_views, teacher_temp
    )
    assert "dino" in teacher_outputs and "ibot" in teacher_outputs
    for group in arch.target_group_names:
        assert group in teacher_outputs["dino"]
        assert isinstance(teacher_outputs["dino"][group], torch.Tensor)

    # _run_student_pass
    student_outputs = arch._run_student_pass(images, masks, collated_views)
    assert "dino" in student_outputs and "ibot" in student_outputs
    for group in images:
        assert group in student_outputs["dino"]
        assert isinstance(student_outputs["dino"][group], torch.Tensor)

    # _calculate_dino_loss
    dino_loss = arch._calculate_dino_loss(
        student_outputs["dino"], teacher_outputs["dino"], collated_views
    )
    assert isinstance(dino_loss, torch.Tensor)

    # _calculate_koleo_loss
    koleo_loss = arch._calculate_koleo_loss(student_outputs["dino"])
    assert isinstance(koleo_loss, torch.Tensor)

    # _calculate_ibot_loss
    ibot_loss = arch._calculate_ibot_loss(
        student_outputs["ibot"], teacher_outputs["ibot"]
    )
    assert isinstance(ibot_loss, torch.Tensor)

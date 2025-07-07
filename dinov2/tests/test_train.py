import pytest
import torch
from dinov2.configs import get_cfg_from_path
from dinov2.train.ssl_meta_arch import SSLMetaArch


@pytest.fixture
def arch():
    cfg = get_cfg_from_path("dinov2/tests/configs/minimal_model.yaml")
    return SSLMetaArch(cfg, device=torch.device("cpu"))


@pytest.fixture
def collated_views():
    return {
        "3d_global_ld": {
            "images": torch.rand(8, 1, 112, 112, 112),
            "is_target": True,
            "targets": ["3d_global_ld"],
            "embed_layer": "patch3d",
            "mask_shape": (8, 8, 8),
            "view_shape": (1,),
            "masks": torch.rand(8, 1, 512) > 0.5,
        },
        "3d_local_ld": {
            "images": torch.rand(8, 1, 3, 56, 56, 56),
            "is_target": False,
            "targets": ["3d_global_ld"],
            "embed_layer": "patch3d",
            "mask_shape": (4, 4, 4),
            "view_shape": (1, 3),
        },
        "2d_global_hd": {
            "images": torch.rand(8, 1, 3, 1, 224, 224),
            "is_target": True,
            "targets": ["3d_global_ld", "2d_global_hd"],
            "embed_layer": "patch2d",
            "mask_shape": (16, 16),
            "view_shape": (1, 3),
            "masks": torch.rand(8, 1, 3, 256) > 0.5,
        },
        "2d_local_hd": {
            "images": torch.rand(8, 1, 3, 2, 1, 112, 112),
            "is_target": False,
            "targets": ["3d_global_ld", "2d_global_hd"],
            "embed_layer": "patch2d",
            "mask_shape": (8, 8),
            "view_shape": (1, 3, 2),
        },
    }


def test_ssl_meta_arch_intermediate_steps(arch, collated_views):
    teacher_temp = 1.0

    # _prepare_inputs
    arch._prepare_inputs(collated_views)

    # _run_teacher_pass
    teacher_outputs = arch._run_teacher_pass(collated_views, teacher_temp)

    # _run_student_pass
    student_outputs = arch._run_student_pass(collated_views)

    # _calculate_dino_loss
    dino_loss = arch._calculate_dino_loss(
        student_outputs["dino"], teacher_outputs["dino"], collated_views
    )
    assert isinstance(dino_loss, torch.Tensor)

    # _calculate_koleo_loss
    student_target_dino_tokens = {
        group_name: tokens
        for group_name, tokens in student_outputs["dino"].items()
        if collated_views[group_name]["is_target"]
    }
    koleo_loss = arch._calculate_koleo_loss(student_target_dino_tokens)
    assert isinstance(koleo_loss, torch.Tensor)

    # _calculate_ibot_loss
    ibot_loss = arch._calculate_ibot_loss(
        student_outputs["ibot"],
        teacher_outputs["ibot"],
        student_outputs["mask_weights"],
    )
    assert isinstance(ibot_loss, torch.Tensor)

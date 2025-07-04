import pytest
import torch
from dinov2.configs import get_cfg_from_path
from dinov2.train.ssl_meta_arch import SSLMetaArch


@pytest.fixture
def arch():
    cfg = get_cfg_from_path("tests/configs/minimal_model.yaml")
    return SSLMetaArch(cfg, device=torch.device("cpu"))


@pytest.fixture
def collated_views():
    return {
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


def test_ssl_meta_arch_intermediate_steps(arch, collated_views):
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

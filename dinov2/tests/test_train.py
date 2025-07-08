import pytest
import torch
from functools import partial

from dinov2.configs import get_cfg_from_path
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.data import collate_data_and_cast, MaskingGenerator, DataAugmentationDINO
from dinov2.logging.helpers import custom_repr_nested


@pytest.fixture
def cfg():
    return get_cfg_from_path("dinov2/tests/configs/minimal_model.yaml")


@pytest.fixture
def device():
    return torch.device(0)


@pytest.fixture
def input_dtype(cfg):
    dtype_str = cfg.compute_precision
    if dtype_str == "fp16":
        return torch.half
    elif dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


@pytest.fixture
def arch(cfg, device, input_dtype):
    arch = SSLMetaArch(cfg, device=device, dtype=input_dtype)
    arch.to(device)
    return arch


@pytest.fixture
def collated_views(cfg, input_dtype):

    dataset_config = cfg.datasets[0]

    augmentation_module = DataAugmentationDINO(
        config=cfg,
        dataset_config=dataset_config,
    )

    mask_generator = MaskingGenerator()

    collator = partial(
        collate_data_and_cast,
        mask_ratio_tuple=(0.1, 0.5),
        mask_probability=0.5,
        dtype=input_dtype,
        mask_generator=mask_generator,
    )

    batch_size = 8
    samples = [
        augmentation_module(torch.rand(512, 512, 512)) for _ in range(batch_size)
    ]

    return collator(samples)


def test_forward(arch, collated_views):
    teacher_temp = 1.0

    arch._prepare_inputs(collated_views)

    with arch.autocast_ctx():
        teacher_outputs = arch._run_teacher_pass(collated_views, teacher_temp)
        student_outputs = arch._run_student_pass(collated_views)

    dino_loss = arch._calculate_dino_loss(
        student_outputs["dino"], teacher_outputs["dino"], collated_views
    )
    ibot_loss = arch._calculate_ibot_loss(
        student_outputs["ibot"],
        teacher_outputs["ibot"],
        student_outputs["mask_weights"],
    )
    student_target_dino_tokens = {
        group_name: tokens
        for group_name, tokens in student_outputs["dino"].items()
        if collated_views[group_name]["is_target"]
    }
    koleo_loss = arch._calculate_koleo_loss(student_target_dino_tokens)

    loss_accumulator = (
        (arch.dino_loss_weight * dino_loss)
        + (arch.ibot_loss_weight * ibot_loss)
        + (arch.koleo_loss_weight * koleo_loss)
    )

    loss_accumulator.backward()

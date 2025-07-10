import pytest
import torch
import logging
import time
from functools import partial

from dinov2.configs import get_cfg_from_path
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.data import collate_data_and_cast, MaskingGenerator, DataAugmentationDINO
from dinov2.logging.helpers import custom_repr_nested

logger = logging.getLogger("test")
logger.setLevel(logging.DEBUG)


@pytest.fixture
def cfg():
    return get_cfg_from_path("dinov2/tests/configs/minimal_model.yaml")


@pytest.fixture
def device():
    return torch.device(0)


@pytest.fixture
def arch(cfg, device):
    arch = SSLMetaArch(cfg, device=device)
    arch.to(device)
    return arch


@pytest.fixture
def collated_views(cfg):

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
        dtype=torch.half,
        mask_generator=mask_generator,
    )

    batch_size = cfg.train.batch_size_per_gpu
    samples = [
        augmentation_module(torch.rand(512, 512, 512)) for _ in range(batch_size)
    ]

    return collator(samples)


def test_forward(arch, collated_views):
    teacher_temp = 1.0

    start_forward = time.time()

    start_prepare = time.time()
    arch._prepare_inputs(collated_views)
    time_prepare = time.time() - start_prepare
    logger.debug(f"Prepare time: {time_prepare:.06f}")

    with arch.autocast_ctx():
        start_teacher = time.time()
        teacher_outputs = arch._run_teacher_pass(collated_views, teacher_temp)
        time_teacher = time.time() - start_teacher
        logger.debug(f"Teacher time: {time_teacher:.06f}")

        start_student = time.time()
        student_outputs = arch._run_student_pass(collated_views)
        time_student = time.time() - start_student
        logger.debug(f"Student time: {time_student:.06f}")

    start_loss = time.time()
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
    time_loss = time.time() - start_loss
    logger.debug(f"Backward Loss: {time_loss:.06f}")

    start_backward = time.time()
    loss_accumulator.backward()
    time_backward = time.time() - start_backward
    logger.debug(f"Backward time: {time_backward:.06f}")

    time_forward = time.time() - start_forward
    logger.debug(f"Total foward time: {time_forward:.06f}")

    start_forward = time.time()

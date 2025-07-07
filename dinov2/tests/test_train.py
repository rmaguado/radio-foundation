import pytest
import torch
from functools import partial

from dinov2.configs import get_cfg_from_path
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.data import collate_data_and_cast, MaskingGenerator, DataAugmentationDINO


@pytest.fixture
def cfg():
    return get_cfg_from_path("dinov2/tests/configs/minimal_model.yaml")


@pytest.fixture
def arch(cfg):
    return SSLMetaArch(cfg, device=torch.device("cpu"), dtype=torch.float32)


@pytest.fixture
def collated_views(cfg):

    dataset_config = cfg.datasets[0]

    augmentation_module = DataAugmentationDINO(
        config=cfg.augmentations.default_ct,
        dataset_config=dataset_config,
    )

    mask_generator = MaskingGenerator()

    collator = partial(
        collate_data_and_cast,
        mask_ratio_tuple=(0.1, 0.5),
        mask_probability=0.5,
        dtype=torch.float32,
        mask_generator=mask_generator,
    )

    batch_size = 8
    samples = [
        augmentation_module(torch.rand(512, 512, 512)) for _ in range(batch_size)
    ]

    return collator(samples)


def test_e2e(arch, collated_views):

    total_loss, loss_dict = arch.train_model(collated_views, teacher_temp=1.0)


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

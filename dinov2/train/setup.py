import logging
import torch
from functools import partial
import numpy as np
import random

from dinov2.train.checkpointer import DDPCheckpointer, DDPPeriodicCheckpointer

from dinov2.data import collate_data_and_cast, MaskingGenerator
from dinov2.data import SamplerType, make_data_loader, make_train_dataset


class CosineScheduler(object):
    def __init__(
        self,
        base_value,
        final_value,
        total_iters,
        warmup_iters=0,
        start_warmup_value=0,
        freeze_iters=0,
    ):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / len(iters))
        )
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(
        params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2)
    )


def build_schedulers(cfg):
    epoch_len = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * epoch_len,
        warmup_iters=cfg.optim["warmup_epochs"] * epoch_len,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * epoch_len,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * epoch_len,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * epoch_len,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * epoch_len,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * epoch_len
    ] = 0

    return {
        "lr": lr_schedule,
        "wd": wd_schedule,
        "momentum": momentum_schedule,
        "teacher_temp": teacher_temp_schedule,
        "last_layer_lr": last_layer_lr_schedule,
    }


def setup_dataloader(cfg, inputs_dtype):

    image_size = cfg.crops.global_crops_size

    patch_size = cfg.student.patch_size
    n_tokens = (image_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(image_size // patch_size, image_size // patch_size),
        max_num_patches=0.5 * image_size // patch_size * image_size // patch_size,
    )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    dataset, weights = make_train_dataset(cfg)

    batch_size_per_gpu = cfg.train.batch_size_per_gpu
    if weights is not None:
        sampler_type = SamplerType.WEIGHTED_SHARDED_INFINITE
    else:
        sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        seed=cfg.train.seed,
        weights=weights,
        sampler_type=sampler_type,
        drop_last=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    return data_loader


def get_max_iter(cfg):
    num_epochs = cfg.optim.epochs
    epoch_len = cfg.train.OFFICIAL_EPOCH_LENGTH
    return num_epochs * epoch_len


def setup_training_components(cfg, model):
    logger = logging.getLogger("dinov2")

    optimizer = build_optimizer(cfg, model.get_params_groups())
    logger.info("Optimizer ready.")
    schedulers = build_schedulers(cfg)
    logger.info("Schedulers ready.")

    checkpointer = DDPCheckpointer(
        model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True
    )

    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS)
    max_iter = get_max_iter(cfg)

    checkpointer = DDPPeriodicCheckpointer(
        checkpointer,
        period=cfg.train.saveckp_iterations,
        max_iter=max_iter,
        max_to_keep=3,
    )

    return (
        optimizer,
        schedulers,
        checkpointer,
        start_iter,
        max_iter,
    )

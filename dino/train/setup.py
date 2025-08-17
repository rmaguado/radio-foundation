import logging
import torch
from functools import partial
from typing import Tuple
import numpy as np
import random

from dino.train.checkpointer import DDPCheckpointer, DDPPeriodicCheckpointer

from dino.data import collate_data_and_cast, MaskingGenerator
from dino.data import SamplerType, make_data_loader, make_train_dataset
from dino.data.augmentations import DataAugmentationDINO


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


def linear_warmup_cosine_decay(
    start: float,
    peak: float,
    end: float,
    warmup_iterations: int,
    total_iterations: int,
    cosine_iterations: int | None = None,
) -> np.ndarray:
    """
    Create a learning rate schedule with linear warmup, a cosine, and an optional constant part in the end.

    Args:
        start (float): Initial learning rate.
        peak (float): Learning rate after linear warmup.
        end (float): Final learning rate after cosine.
        warmup_iterations (int): Number of iterations for linear warmup.
        total_iterations (int): Total number of iterations for the schedule.
        cosine_iterations (int | None): Number of iterations for cosine.
            If None, cosine part will be over remaining iterations after warmup.
    Returns:
        np.ndarray: Learning rate schedule as a numpy array.
    """
    linear = np.linspace(start, peak, warmup_iterations, endpoint=False)
    if cosine_iterations is None:
        cosine_iterations = total_iterations - warmup_iterations
    cosine = np.cos(np.linspace(0, np.pi, cosine_iterations))
    cosine = (cosine + 1) / 2
    cosine = (peak - end) * cosine + end
    remaining_iterations = total_iterations - cosine_iterations - warmup_iterations
    assert remaining_iterations >= 0
    constant = np.full((remaining_iterations,), fill_value=end)
    return np.concatenate([linear, cosine, constant])


def build_schedulers(cfg):
    epoch_len = cfg.train.iterations_per_epoch

    lr_peak = cfg.schedules.lr.peak
    lr_end = cfg.schedules.lr.end
    lr_schedule = linear_warmup_cosine_decay(
        start=cfg.schedules.lr.start,
        peak=lr_peak,
        end=lr_end,
        warmup_iterations=epoch_len * cfg.schedules.lr.warmup_epochs,
        total_iterations=epoch_len,
        cosine_iterations=(
            epoch_len * cfg.schedules.lr.cosine_epochs
            if cfg.schedules.lr.cosine_epochs is not None
            else None
        ),
    )
    last_layer_lr_schedule = lr_schedule.copy()
    last_layer_lr_schedule[: epoch_len * cfg.schedules.lr.freeze_last_layer_epochs] = 0

    wd_schedule = linear_warmup_cosine_decay(
        start=cfg.schedules.weight_decay.start,
        peak=cfg.schedules.weight_decay.peak,
        end=cfg.schedules.weight_decay.end,
        warmup_iterations=epoch_len * cfg.schedules.weight_decay.warmup_epochs,
        total_iterations=epoch_len,
        cosine_iterations=(
            epoch_len * cfg.schedules.weight_decay.cosine_epochs
            if cfg.schedules.weight_decay.cosine_epochs is not None
            else None
        ),
    )

    momentum_schedule = linear_warmup_cosine_decay(
        start=cfg.schedules.momentum.start,
        peak=cfg.schedules.momentum.peak,
        end=cfg.schedules.momentum.end,
        warmup_iterations=epoch_len * cfg.schedules.momentum.warmup_epochs,
        total_iterations=epoch_len,
        cosine_iterations=(
            epoch_len * cfg.schedules.momentum.cosine_epochs
            if cfg.schedules.momentum.cosine_epochs is not None
            else None
        ),
    )
    teacher_temp_schedule = linear_warmup_cosine_decay(
        start=cfg.schedules.teacher_temp.start,
        peak=cfg.schedules.teacher_temp.peak,
        end=cfg.schedules.teacher_temp.end,
        warmup_iterations=epoch_len * cfg.schedules.teacher_temp.warmup_epochs,
        total_iterations=epoch_len,
        cosine_iterations=(
            epoch_len * cfg.schedules.teacher_temp.cosine_epochs
            if cfg.schedules.teacher_temp.cosine_epochs is not None
            else None
        ),
    )

    return {
        "lr": lr_schedule,
        "wd": wd_schedule,
        "momentum": momentum_schedule,
        "teacher_temp": teacher_temp_schedule,
        "last_layer_lr": last_layer_lr_schedule,
    }


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(
        params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2)
    )


def setup_collate_fn(cfg, inputs_dtype):
    mask_shape = (cfg.crops.size_global // cfg.student.embed_layer.patch_size,)
    if cfg.student.embed_layer.type == "patch_2d":
        mask_shape *= 2
    elif cfg.student.embed_layer.type == "patch_3d":
        mask_shape *= 3

    mask_generator = MaskingGenerator()

    return partial(
        collate_data_and_cast,
        mask_ratio_range=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        mask_shape=mask_shape,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )


def setup_dataloader(cfg, inputs_dtype, iteration: int = 0):
    collate_fn = setup_collate_fn(cfg, inputs_dtype)

    dataset, weights = make_train_dataset(cfg)

    if weights is not None:
        sampler_type = SamplerType.WEIGHTED_INFINITE
    else:
        sampler_type = SamplerType.INFINITE

    data_loader = make_data_loader(
        dataset=dataset,
        batch_size_per_gpu=cfg.train.batch_size_per_gpu,
        batch_size_total=cfg.train.batch_size_total,
        num_workers=cfg.train.num_workers,
        iteration=iteration,
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
    epoch_len = cfg.train.iterations_per_epoch
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

    start_iter = checkpointer.resume_or_load()
    max_iter = get_max_iter(cfg)

    checkpointer = DDPPeriodicCheckpointer(
        checkpointer,
        period=cfg.checkpoints.save_checkpoint_iterations,
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

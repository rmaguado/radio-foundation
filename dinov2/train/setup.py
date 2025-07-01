import torch
from functools import partial
from typing import Dict, Tuple

from dinov2.utils.checkpointer import get_checkpointer, DDPPeriodicCheckpointer
from dinov2.utils.utils import CosineScheduler
from dinov2.data import collate_data_and_cast, MaskingGenerator
from dinov2.data import SamplerType, make_data_loader, make_train_dataset


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(
        params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2)
    )


def build_schedulers(cfg):
    epoch_len = cfg.train.iterations_per_epoch
    total_epochs = cfg.train.stage1.epochs
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=total_epochs * epoch_len,
        warmup_iters=cfg.optim["warmup_epochs"] * epoch_len,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=total_epochs * epoch_len,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=total_epochs * epoch_len,
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
    mask_generator = MaskingGenerator()

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    dataset, weights = make_train_dataset(cfg)

    batch_size = cfg.train.stage1.batch_size_per_gpu
    if weights is not None:
        sampler_type = SamplerType.WEIGHTED_SHARDED_INFINITE
    else:
        sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=cfg.train.num_workers,
        seed=cfg.train.seed,
        weights=weights,
        sampler_type=sampler_type,
        drop_last=True,
        collate_fn=collate_fn,
    )

    return data_loader


def setup_training_components(cfg, model, resume) -> Tuple[
    torch.optim.Optimizer,
    Dict[str, CosineScheduler],
    DDPPeriodicCheckpointer,
    int,
    int,
]:
    optimizer = build_optimizer(cfg, model.module.get_params_groups())
    schedulers = build_schedulers(cfg)

    total_epochs = cfg.train.stage1.epochs
    epoch_len = cfg.train.iterations_per_epoch

    # start_iter = checkpointer.resume_or_load(cfg.train.output_dir, resume=resume).get("iteration", -1) + 1
    start_iter = 0
    max_iter = total_epochs * epoch_len

    checkpointer = get_checkpointer(cfg, model, optimizer, max_iter=max_iter)

    return (
        optimizer,
        schedulers,
        checkpointer,
        start_iter,
        max_iter,
    )

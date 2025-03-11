import logging
import torch
from functools import partial

from fvcore.common.checkpoint import PeriodicCheckpointer

from dinov2.fsdp import FSDPCheckpointer, FlexibleFSDPCheckpointer
from dinov2.utils.utils import CosineScheduler
from dinov2.data import collate_data_and_cast, MaskingGenerator
from dinov2.data import SamplerType, make_data_loader, make_train_dataset


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


def setup_dataloader(cfg, inputs_dtype, use_full_image: bool):

    image_size = (
        cfg.student.full_image_size if use_full_image else cfg.crops.global_crops_size
    )

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

    dataset, weights = make_train_dataset(cfg, use_full_image)

    batch_size = (
        cfg.train.full_image.batch_size_per_gpu
        if use_full_image
        else cfg.train.batch_size_per_gpu
    )
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


def get_full_size_iter(cfg):
    full_img_epochs = cfg.train.full_image.epochs
    epoch_len = cfg.train.OFFICIAL_EPOCH_LENGTH
    return full_img_epochs * epoch_len


def get_cropped_iter(cfg):
    total_epochs = cfg.optim.epochs
    full_img_epochs = cfg.train.full_image.epochs
    cropped_epochs = total_epochs - full_img_epochs
    epoch_len = cfg.train.OFFICIAL_EPOCH_LENGTH
    return cropped_epochs * epoch_len


def setup_training_components(cfg, model, resume):
    logger = logging.getLogger("dinov2")

    optimizer = build_optimizer(cfg, model.get_params_groups())
    logger.info("Optimizer ready.")
    schedulers = build_schedulers(cfg)
    logger.info("Schedulers ready.")

    sharding_strategy = cfg.compute_precision.teacher.backbone.sharding_strategy
    if sharding_strategy in [
        "NO_SHARD",
        "SHARD_GRAD_OP",
    ]:
        checkpointer_wrapper = FlexibleFSDPCheckpointer
    else:
        checkpointer_wrapper = FSDPCheckpointer

    checkpointer = checkpointer_wrapper(
        model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True
    )

    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get(
            "iteration", -1
        )
        + 1
    )

    full_size_iter = get_full_size_iter(cfg)
    cropped_iter = get_cropped_iter(cfg)
    max_iter = cropped_iter + full_size_iter

    checkpointer = PeriodicCheckpointer(
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
        full_size_iter,
    )

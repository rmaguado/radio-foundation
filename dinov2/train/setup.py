import logging
import torch
from functools import partial

from fvcore.common.checkpoint import PeriodicCheckpointer

from dinov2.fsdp import FSDPCheckpointer
from dinov2.utils.utils import CosineScheduler
from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
from dinov2.data import SamplerType, make_data_loader, make_dataset


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))

def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0

    return {
        'lr': lr_schedule,
        'wd': wd_schedule,
        'momentum': momentum_schedule,
        'teacher_temp': teacher_temp_schedule,
        'last_layer_lr': last_layer_lr_schedule
    }
    

def setup_dataloader(cfg, image_mode="crop", inputs_dtype=torch.half, ):
    if image_mode == "crop":
        image_size = cfg.augmentations.crops.global_crops_size
        batch_size = cfg.train.batch_size_per_gpu
    elif image_mode == "full":
        image_size = cfg.student.full_image_size
        batch_size = cfg.train.batch_size_reduced
        
    patch_size = cfg.student.patch_size
    n_tokens = (image_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(image_size // patch_size, image_size // patch_size),
        max_num_patches=0.5 * image_size // patch_size * image_size // patch_size,
    )

    data_transform = DataAugmentationDINO(cfg.augmentations)

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        transform=data_transform,
        target_transform=lambda _: (),
    )

    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=cfg.train.seed,
        sampler_type=sampler_type,
        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )
    
    return data_loader

def setup_training_components(cfg, model, resume):
    logger = logging.getLogger("dinov2")
    
    optimizer = build_optimizer(cfg, model.get_params_groups())
    logger.info("Optimizer ready.")
    schedulers = build_schedulers(cfg)
    logger.info("Schedulers ready.")

    fsdp_checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)
    
    start_iter = fsdp_checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    max_iter = cfg.optim.epochs * cfg.train.OFFICIAL_EPOCH_LENGTH * cfg.train.grad_accum_steps
    
    checkpointer = PeriodicCheckpointer(
        fsdp_checkpointer,
        period=cfg.train.saveckp_iterations,
        max_iter=max_iter,
        max_to_keep=3,
    )
    
    return optimizer, schedulers, checkpointer, start_iter, max_iter

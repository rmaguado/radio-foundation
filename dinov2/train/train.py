# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
from functools import partial

from fvcore.common.checkpoint import PeriodicCheckpointer
import torch

from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler

from dinov2.train.ssl_meta_arch import SSLMetaArch


torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )
    return parser


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
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return {
        'lr': lr_schedule,
        'wd': wd_schedule,
        'momentum': momentum_schedule,
        'teacher_temp': teacher_temp_schedule,
        'last_layer_lr': last_layer_lr_schedule
    }
    

def setup_dataloader(cfg, image_mode="crop", inputs_dtype=torch.half):
    if image_mode == "crop":
        image_size = cfg.augmentations.crops.global_crops_size
        batch_size = cfg.train.batch_size_per_gpu
    elif image_mode == "full":
        image_size = cfg.student.image_size_full
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
    # sampler_type = SamplerType.INFINITE
    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )
    
    return data_loader

def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def do_test(cfg, model, iteration):
    torch.cuda.synchronize()
    
    new_state_dict = {k: v.cpu() for k, v in model.teacher.state_dict().items()}

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        
        torch.cuda.empty_cache()
        
        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)

def setup_training_components(cfg, model, resume):
    optimizer = build_optimizer(cfg, model.get_params_groups())
    schedulers = build_schedulers(cfg)

    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    max_iter = cfg.optim.epochs * cfg.train.OFFICIAL_EPOCH_LENGTH * cfg.train.grad_accum_steps
    
    fsdp_checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)
    checkpointer = PeriodicCheckpointer(
        fsdp_checkpointer,
        period=cfg.train.saveckp_iterations,
        max_iter=max_iter,
        max_to_keep=3,
    )
    
    return optimizer, schedulers, checkpointer, start_iter, max_iter

def update_schedules(optimizer, schedulers):
    lr = schedulers['lr'][train_step]
    wd = schedulers['wd'][train_step]
    momentum = schedulers['momentum'][train_step]
    teacher_temp = schedulers['teacher_temp'][train_step]
    last_layer_lr = schedulers['last_layer_lr'][train_step]
    apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

    return momentum, teacher_temp

def apply_gradient_operations(cfg, model, optimizer, fp16_scaler):
    if fp16_scaler is not None:
        fp16_scaler.unscale_(optimizer)

    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.div_(GRAD_ACCUM_STEPS)

    if cfg.optim.clip_grad:
        for v in model.student.values():
            v.clip_grad_norm_(cfg.optim.clip_grad)

    if fp16_scaler is not None:
        fp16_scaler.step(optimizer)
        fp16_scaler.update()
    else:
        optimizer.step()

        
def log_training_step(metric_logger, loss_dict, schedulers, train_step, current_batch_size):
    if distributed.get_global_size() > 1:
        for v in loss_dict.values():
            torch.distributed.all_reduce(v)
    loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

    if math.isnan(sum(loss_dict_reduced.values())):
        logger.error(f"NaN detected in reduced loss at iteration {iteration}")
        logger.info(f"Reduced loss dict: {loss_dict_reduced}")
        raise AssertionError

    losses_reduced = sum(loss for loss in loss_dict_reduced.values())

    metric_logger.update(lr=schedulers['lr'][train_step])
    metric_logger.update(wd=schedulers['wd'][train_step])
    metric_logger.update(mom=schedulers['momentum'][train_step])
    metric_logger.update(last_layer_lr=schedulers['last_layer_lr'][train_step])
    metric_logger.update(current_batch_size=current_batch_size)
    metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

    
def should_reset_grad(cfg, grad_accum_counter):
    return grad_accum_counter % cfg.train.grad_accum_steps == 0

def should_apply_training_step(cfg, grad_accum_counter):
    return (grad_accum_counter + 1) % cfg.train.grad_accum_steps == 0

def should_eval_model(cfg, iteration):
    return cfg.evaluation.eval_period_iterations > 0 and \
    (iteration + 1) % cfg.evaluation.eval_period_iterations == 0

def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler

    optimizer, schedulers, checkpointer, start_iter, max_iter = setup_training_components(cfg, model, resume)
    
    full_size_start_iter = max_iter - cfg.train.full_size_steps * cfg.train.grad_accum_steps
    
    data_loader = setup_dataloader(cfg, "crop", inputs_dtype)

    iteration = start_iter
    train_step = start_iter // cfg.train.grad_accum_steps
    grad_accum_counter = 0

    logger.info("Starting training from iteration {}".format(start_iter))
    metric_logger = MetricLogger(delimiter="  ", output_file=os.path.join(cfg.train.output_dir, "training_metrics.json"))
    
    for data in metric_logger.log_every(
        data_loader,
        cfg.train.print_freq,
        "Training",
        max_iter,
        start_iter,
    ):
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return
        
        if should_reset_grad(cfg, grad_accum_counter):
            mom, teacher_temp = update_schedules()
            optimizer.zero_grad(set_to_none=True)
        
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)
        
        if should_apply_training_step(cfg, grad_accum_counter):
            apply_gradient_operations(cfg, model, optimizer, fp16_scaler)
            model.update_teacher(mom)
            log_training_step(metric_logger, loss_dict, schedulers, train_step, current_batch_size)
            train_step += 1
        
        if should_eval_model(cfg, iteration):
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        
        checkpointer.step(iteration)
        grad_accum_counter += 1
        iteration += 1
    
    metric_logger.synchronize_between_processes()
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")

    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)

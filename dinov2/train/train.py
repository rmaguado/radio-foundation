# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import torch
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from dinov2.logging import MetricLogger
from dinov2.utils.config import setup

from dinov2.train.utils import (
    update_schedules,
    apply_gradient_operations,
    log_training_step,
    do_test,
)
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.train.parser import get_args_parser
from dinov2.train.setup import setup_training_components, setup_dataloader

torch.backends.cuda.matmul.allow_tf32 = True
logger = logging.getLogger("dinov2")


def should_reset_grad(cfg, grad_accum_counter, accum_steps):
    return grad_accum_counter % accum_steps == 0


def should_apply_training_step(cfg, grad_accum_counter, accum_steps):
    return (grad_accum_counter + 1) % accum_steps == 0


def should_eval_model(cfg, iteration):
    return (
        cfg.evaluation.eval_period_iterations > 0
        and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0
    )


def train(
    cfg,
    metric_logger,
    model,
    optimizer,
    schedulers,
    checkpointer,
    img_mode: str,
    start_iter: int,
    max_iter: int,
    train_step: int,
):
    fp16_scaler = model.fp16_scaler
    grad_accum_counter = 0
    iteration = start_iter

    if img_mode == "crop":
        accum_steps = cfg.train.grad_accum_steps
    elif img_mode == "full":
        accum_steps = cfg.train.full_image.grad_accum_steps
    else:
        raise ValueError

    for data in metric_logger.log_every(
        cfg.train.print_freq,
        "Training",
        max_iter,
        start_iter,
    ):
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return

        if should_reset_grad(cfg, grad_accum_counter, accum_steps):
            mom, teacher_temp = update_schedules(optimizer, schedulers, train_step)
            optimizer.zero_grad(set_to_none=True)

        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

        if should_apply_training_step(cfg, grad_accum_counter, accum_steps):
            apply_gradient_operations(cfg, model, optimizer, fp16_scaler, accum_steps)
            model.update_teacher(mom)
            log_training_step(
                metric_logger, loss_dict, schedulers, train_step, current_batch_size
            )
            train_step += 1

        if should_eval_model(cfg, iteration):
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()

        checkpointer.step(iteration)
        grad_accum_counter += 1
        iteration += 1
    return iteration, train_step


def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half

    (
        optimizer,
        schedulers,
        checkpointer,
        start_iter,
        train_step,
        max_iter,
        full_size_iter,
    ) = setup_training_components(cfg, model, resume)

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metric_logger = MetricLogger(
        output_file=os.path.join(cfg.train.output_dir, "training_metrics.json"),
    )

    train_components = [cfg, metric_logger, model, optimizer, schedulers, checkpointer]

    if iteration < max_iter - full_size_iter:
        data_loader = setup_dataloader(cfg, inputs_dtype, use_full_image=False)
        metric_logger.set_dataloader(data_loader)

        iteration, train_step = train(
            *train_components,
            img_mode="crop",
            start_iter=start_iter,
            max_iter=max_iter - full_size_iter,
            train_step=train_step,
        )

        logger.info(f"Finished training on resize-crop images.")
    logger.info(f"Resuming with full-size images for {max_iter - iteration} steps")

    data_loader = setup_dataloader(cfg, inputs_dtype, use_full_image=True)
    metric_logger.set_dataloader(data_loader)
    train(
        *train_components,
        img_mode="full",
        start_iter=iteration,
        max_iter=max_iter,
        train_step=train_step,
    )
    logger.info("Finished training on full-size images")

    metric_logger.synchronize_between_processes()

    return {k: meter.avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    logger.info("Model:\n{}".format(model))

    do_train(cfg, model, resume=not args.no_resume)

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import torch

from torch.profiler import profile, ProfilerActivity, record_function

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
import dinov2.distributed as dist

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


def get_dtype(dtype_str):
    if dtype_str == "fp16":
        return torch.half
    elif dtype_str == "bf16":
        return torch.bfloat16
    return torch.float


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
):
    dtype = get_dtype(cfg.compute_precision)
    grad_accum_counter = 0
    iteration = start_iter

    if img_mode == "crop":
        accum_steps = cfg.train.grad_accum_steps
    elif img_mode == "full":
        accum_steps = cfg.train.grad_accum_steps
    else:
        raise ValueError

    for data in metric_logger.log_every(
        cfg.train.print_freq,
        "Training",
        max_iter,
        start_iter,
        accum_steps,
    ):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
        ) as prof:
            if iteration > max_iter:
                return

            if should_reset_grad(cfg, grad_accum_counter, accum_steps):
                mom, teacher_temp = update_schedules(optimizer, schedulers, iteration)
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", enabled=True, dtype=dtype):
                with record_function("mode-forward"):
                    loss_dict, loss_accumulator = model.forward(
                        data, teacher_temp=teacher_temp
                    )

            with record_function("model-backward"):
                loss_accumulator.backward()

                torch.cuda.synchronize()

            if should_apply_training_step(cfg, grad_accum_counter, accum_steps):
                with record_function("grad-operations"):
                    apply_gradient_operations(cfg, model, optimizer, accum_steps)
                model.update_teacher(mom)

                with record_function("logging"):
                    log_training_step(metric_logger, loss_dict, schedulers, iteration)

                    checkpointer.step(iteration)

                iteration += 1

                if should_eval_model(cfg, iteration):
                    do_test(cfg, model, f"training_{iteration}")
                    torch.cuda.synchronize()

            grad_accum_counter += 1

        if dist.is_main_process():
            prof.export_chrome_trace(f"runs/_traces/trace_{iteration:05}.json")

    return iteration


def do_train(cfg, model):
    model.train()
    inputs_dtype = get_dtype(cfg.compute_precision)

    (
        optimizer,
        schedulers,
        checkpointer,
        start_iter,
        max_iter,
    ) = setup_training_components(cfg, model)

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metric_logger = MetricLogger(
        output_file=os.path.join(cfg.train.output_dir, "training_metrics.json"),
    )

    train_components = [cfg, metric_logger, model, optimizer, schedulers, checkpointer]

    if iteration < max_iter:
        data_loader = setup_dataloader(cfg, inputs_dtype)
        metric_logger.set_dataloader(data_loader)

        iteration = train(
            *train_components,
            img_mode="crop",
            start_iter=start_iter,
            max_iter=max_iter,
        )

        logger.info("Finished training on resize-crop images.")

    do_test(cfg, model, f"training_{iteration}")


def main():

    dist.setup_distributed_slurm()
    rank = dist.get_rank()

    args = get_args_parser(add_help=True).parse_args()
    cfg = setup(args)

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training(rank)

    try:
        do_train(cfg, model)
    finally:
        dist.cleanup_distributed()


if __name__ == "__main__":
    if os.environ.get("PYTHONPATH") is not None and not os.path.exists("dinov2"):
        os.chdir(os.environ["PYTHONPATH"])

    main()

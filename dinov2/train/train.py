# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from dinov2.logging import MetricLogger

from dinov2.configs import get_cfg_from_path, write_config, validate_config
from dinov2.train.utils import (
    update_schedules,
    apply_gradient_operations,
    log_training_step,
    do_test,
)
from dinov2.utils import utils
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.train.parser import get_args_parser
from dinov2.train.setup import setup_training_components, setup_dataloader

torch.backends.cuda.matmul.allow_tf32 = True
logger = logging.getLogger("dinov2")


def should_reset_grad(grad_accum_counter, accum_steps):
    return grad_accum_counter % accum_steps == 0


def should_apply_training_step(grad_accum_counter, accum_steps):
    return (grad_accum_counter + 1) % accum_steps == 0


def should_eval_model(iteration, save_teacher_iterations):
    return (
        save_teacher_iterations > 0 and (iteration + 1) % save_teacher_iterations == 0
    )


def train(
    cfg,
    metric_logger,
    model,
    optimizer,
    schedulers,
    checkpointer,
    start_iter: int,
    max_iter: int,
):
    grad_accum_counter = 0
    iteration = start_iter
    accum_steps = cfg.train.stage1.grad_accum_steps

    for data in metric_logger.log_every(
        cfg.checkpoints.print_iterations,
        "Training",
        max_iter,
        start_iter,
        accum_steps,
    ):
        if iteration > max_iter:
            return

        if should_reset_grad(grad_accum_counter, accum_steps):
            mom, teacher_temp = update_schedules(optimizer, schedulers, iteration)
            optimizer.zero_grad(set_to_none=True)

        loss_accumulator, loss_dict = model.forward(data, teacher_temp=teacher_temp)
        loss_accumulator.backward()

        if should_apply_training_step(grad_accum_counter, accum_steps):
            apply_gradient_operations(cfg, model, optimizer, accum_steps)
            model.update_teacher(mom)

            log_training_step(metric_logger, loss_dict, schedulers, iteration)

            checkpointer.step(iteration)

            iteration += 1

            if should_eval_model(iteration, cfg.checkpoints.save_teacher_iterations):
                do_test(cfg, model, f"training_{iteration}")
                torch.cuda.synchronize()

        grad_accum_counter += 1

    return iteration


def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half

    (
        optimizer,
        schedulers,
        checkpointer,
        start_iter,
        max_iter,
    ) = setup_training_components(cfg, model, resume)

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metric_logger = MetricLogger(
        output_file=os.path.join(cfg.train.output_dir, "training_metrics.json"),
    )

    train_components = [cfg, metric_logger, model, optimizer, schedulers, checkpointer]

    data_loader = setup_dataloader(cfg, inputs_dtype)
    metric_logger.set_dataloader(data_loader)

    iteration = train(
        *train_components,
        start_iter=start_iter,
        max_iter=max_iter,
    )

    logger.info(f"Finished training stage 1.")
    do_test(cfg, model, f"training_{iteration}")
    logger.info("Finished training on full-size images")


def main(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    args = get_args_parser(add_help=True).parse_args()
    cfg = get_cfg_from_path(args.config_path)
    os.makedirs(args.output_path, exist_ok=True)

    seed = getattr(args, "seed", 0)
    rank = dist.get_rank()

    if getattr(args, "debug", False):
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO

    global logger
    setup_logging(output=args.output_path, level=logging_level)
    logger = logging.getLogger("dinov2")

    utils.fix_random_seeds(seed + rank)

    write_config(cfg, args.output_path)
    validate_config(cfg)

    model = SSLMetaArch(cfg).to(rank)
    model = DDP(model, device_ids=[rank])

    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    if os.environ.get("PYTHONPATH") is not None and not os.path.exists("dinov2"):
        os.chdir(os.environ["PYTHONPATH"])

    world_size = torch.cuda.device_count()
    try:
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    finally:
        dist.destroy_process_group()

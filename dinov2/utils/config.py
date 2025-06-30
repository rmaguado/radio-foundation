# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import torch.distributed as dist

from omegaconf import OmegaConf, DictConfig, ListConfig

from dinov2.logging import setup_logging
from dinov2.utils import utils
from dinov2.configs import dinov2_default_config
from dinov2.configs.validation import validate_config


logger = logging.getLogger("dinov2")


def write_config(cfg, output_dir, name="config.yaml"):
    logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_cfg_from_args(args):
    args.output_dir = os.path.abspath(args.output_dir)
    args.opts += [f"train.output_dir={args.output_dir}"]
    default_cfg = OmegaConf.create(dinov2_default_config)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli(args.opts))
    return cfg


def setup(args) -> DictConfig:
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg_from_args(args)
    assert isinstance(cfg, DictConfig), "Config should be a DictConfig instance"
    os.makedirs(args.output_dir, exist_ok=True)

    seed = getattr(args, "seed", 0)
    rank = dist.get_rank()

    if getattr(args, "debug", False):
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO

    global logger
    setup_logging(output=args.output_dir, level=logging_level)
    logger = logging.getLogger("dinov2")

    utils.fix_random_seeds(seed + rank)
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )

    cfg.optim.lr = cfg.optim.base_lr
    write_config(cfg, args.output_dir)
    validate_config(cfg)

    return cfg

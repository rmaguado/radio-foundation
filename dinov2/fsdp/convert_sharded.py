from dinov2.fsdp import AntiFSDPConverter
from dinov2.train.parser import get_args_parser
from dinov2.utils.config import setup
from dinov2.train.ssl_meta_arch import SSLMetaArch

import os
import torch
import logging

logger = logging.getLogger("dinov2")


def load_sharded_save_unsharded(cfg, model):

    optimizer = torch.optim.AdamW(
        model.get_params_groups(), betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2)
    )

    logger.info("Loading checkpoint from %s", cfg.MODEL.WEIGHTS)

    checkpointer = AntiFSDPConverter(
        model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True
    )

    iteration = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=True).get(
        "iteration", -1
    )

    logger.info(
        "Saving unsharded model to %s",
        cfg.MODEL.WEIGHTS.replace(".pth", "_unsharded.pth"),
    )

    checkpointer.save("model_{:07d}".format(iteration), iteration=iteration)

    logger.info("Model saved.")


def main(args):
    cfg = setup(args)

    logger.warning("Loaded config without running validation.")

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    logger.info("Model ready.")

    load_sharded_save_unsharded(cfg, model)


if __name__ == "__main__":
    if os.environ.get("PYTHONPATH") is not None and not os.path.exists("dinov2"):
        os.chdir(os.environ["PYTHONPATH"])

    args = get_args_parser(add_help=True).parse_args()
    try:
        main(args)
    finally:
        torch.distributed.destroy_process_group()

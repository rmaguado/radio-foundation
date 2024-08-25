import os
from omegaconf import DictConfig
import logging


logger = logging.getLogger("dinov2")


class Errors:
    OUTPUT_DIR_MISSING = "Output directory missing or invalid."


def test_output_dir_given(config: DictConfig) -> None:
    if not hasattr(config.train, "output_dir"):
        logger.error(Errors.OUTPUT_DIR_MISSING)
        return False
    output_dir = config.train.output_dir


def validate_train(config: DictConfig) -> None:
    return True

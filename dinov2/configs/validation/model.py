import os
from omegaconf import DictConfig
import logging


logger = logging.getLogger("dinov2")


class Errors:
    PATH_NOT_FOUND = "Path not found: {}"


def validate_model(config: DictConfig) -> None:
    return True

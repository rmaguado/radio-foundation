import os
from omegaconf import DictConfig
import logging


logger = logging.getLogger("dinov2")


class Errors:
    NO_CONFIG = "Config has no augmentations group."
    PATH_NOT_FOUND = "Path not found: {}"


def validate_augmentations(config: DictConfig) -> None:
    return True

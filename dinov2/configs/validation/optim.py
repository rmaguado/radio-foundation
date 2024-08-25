import os
from omegaconf import DictConfig
import logging


logger = logging.getLogger("dinov2")


class Errors:
    NO_CONFIG = "Config has no optim group."
    EPOCHS_INVALID = "Invalid number of epochs: {}"
    BASE_LR_INVALID = "Invalid base learning rate: {}"
    WEIGHT_DECAY_INVALID = "Invalid weight decay value: {}"


def test_epochs(config: DictConfig) -> None:
    if not hasattr(config.optim, "epochs"):
        return False
    epochs = config.optim.epochs
    if not epochs > 0:
        logger.error(Errors.EPOCHS_INVALID.format(epochs))
        return False
    return True


def validate_optim(config: DictConfig) -> None:
    return True

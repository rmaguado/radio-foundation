from omegaconf import DictConfig

from .data import validate_data
from .augmentations import validate_augmentations
from .model import validate_model
from .optim import validate_optim
from .train import validate_train


def validate_config(config: DictConfig):
    return all(
        [
            validate_data(config),
            validate_augmentations(config),
            validate_model(config),
            validate_optim(config),
            validate_train(config),
        ]
    )

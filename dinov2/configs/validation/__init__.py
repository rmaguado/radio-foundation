from omegaconf import DictConfig

from .data import validate_data
from .dino import validate_dino
from .ibot import validate_ibot
from .student import validate_student
from .teacher import validate_teacher
from .optim import validate_optim
from .train import validate_train
from .augmentations import validate_augmentations


def validate_config(config: DictConfig) -> bool:
    return all(
        [
            validate_data(config),
            validate_dino(config),
            validate_ibot(config),
            validate_student(config),
            validate_teacher(config),
            validate_optim(config),
            validate_train(config),
            validate_augmentations(config),
        ]
    )

from omegaconf import DictConfig
import copy

from .data import validate_data
from .dino import validate_dino
from .ibot import validate_ibot
from .student import validate_student
from .teacher import validate_teacher
from .optim import validate_optim
from .train import validate_train


def validate_config(config: DictConfig) -> bool:
    config = copy.deepcopy(config)
    return all(
        [
            validate_data(config),
            validate_dino(config),
            validate_ibot(config),
            validate_student(config),
            validate_teacher(config),
            validate_optim(config),
            validate_train(config),
        ]
    )

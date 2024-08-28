from omegaconf import DictConfig
import logging

from .utils import (
    test_has_section,
    test_attributes_dtypes,
    test_attributes_range,
    ValueRange,
)

logger = logging.getLogger("dinov2")


def validate_teacher(config: DictConfig) -> bool:
    if not test_has_section(config, "teacher"):
        return False
    teacher_config = config.teacher
    required_attributes = [
        ("momentum_teacher", float),
        ("final_momentum_teacher", float),
        ("warmup_teacher_temp", float),
        ("teacher_temp", float),
        ("warmup_teacher_temp_epochs", int),
    ]
    if not test_attributes_dtypes(teacher_config, required_attributes, "teacher"):
        return False

    attributes_ranges = [
        ("momentum_teacher", ValueRange(0.0, 1.0)),
        ("final_momentum_teacher", ValueRange(0.0, 1.0)),
        ("warmup_teacher_temp", ValueRange(0.0, 1.0)),
        ("teacher_temp", ValueRange(0.0, 1.0)),
        ("warmup_teacher_temp_epochs", ValueRange(0)),
    ]
    if not test_attributes_range(teacher_config, attributes_ranges, "teacher"):
        return False

    logger.debug("'teacher' config is valid.")
    return True

import os
from omegaconf import DictConfig
import logging
from typing import List, Tuple

logger = logging.getLogger("dinov2")


class Errors:
    MISSING_ATTR = "Missing attribute in {}: {}."
    INVALID_TYPE = "Invalid type in {}: {} should be of type {}. Found {}"
    INVALID_VALUE = "Invalid value in {}: {} should be {}."
    INVALID_LENGTH = "Invalid length in {}: {} should have length {}."
    PATH_NOT_FOUND = "Path not found: {}."
    NO_DATASETS_FOUND = "No datasets found in data config."
    VIT_SMALL_DROP_PATH = "Drop path rate should be 0.0 for vit_small."
    INVALID_TRAIN_SETUP = "Train setup is invalid. Check epoch num and warmup configs."
    INVALID_VALUE_PAIR = "Invalid pair of values for {}: {} and {}."
    INVALID_VALUE = "Invalid value {}"
    INVALID_DATASET_TYPE = "Invalid dataset type: {}."
    UNRECOGNIZED_TRANSFORM = "Transform '{}' is not recognized. See dinov2/data/transforms.py for recognized transforms for implemented transforms."
    TRANSFORM_INIT_ERROR = "Error initializing transform '{}': {}"
    NO_CROP = "No crop found in augmentations list."
    NO_TRANSFORM_NAME = "Transform name not found in augmentations list."
    NO_AUGMENTATION_PRESET = "Augmentation preset not found: {}."


class ValueRange:
    def __init__(
        self,
        min_val=-float("inf"),
        max_val=float("inf"),
        left_inclusive=True,
        right_inclusive=True,
    ):
        self.min_val = min_val
        self.max_val = max_val
        self.left_inclusive = left_inclusive
        self.right_inclusive = right_inclusive

    def __contains__(self, value):
        if self.left_inclusive:
            if value < self.min_val:
                return False
        else:
            if value <= self.min_val:
                return False
        if self.right_inclusive:
            if value > self.max_val:
                return False
        else:
            if value >= self.max_val:
                return False
        return True

    def __repr__(self):
        left_bracket = "[" if self.left_inclusive else "("
        right_bracket = "]" if self.right_inclusive else ")"
        return f"{left_bracket}{self.min_val}, {self.max_val}{right_bracket}"


def test_has_section(config: DictConfig, section: str) -> bool:
    if not hasattr(config, section):
        logger.error(Errors.MISSING_ATTR.format("config", section))
        return False
    return True


def test_attributes_dtypes(
    config: DictConfig, required_attrs: List[Tuple[str, type]], config_section: str
) -> bool:
    missing_attributes = []
    incorrect_dtype = []
    for attr, dtype in required_attrs:
        if not hasattr(config, attr):
            missing_attributes.append(attr)
        elif not isinstance(getattr(config, attr), dtype):
            incorrect_dtype.append(attr)
    if len(missing_attributes) > 0:
        for attr in missing_attributes:
            logger.error(Errors.MISSING_ATTR.format(config_section, attr))
        return False
    if len(incorrect_dtype) > 0:
        for attr in incorrect_dtype:
            logger.error(
                Errors.INVALID_TYPE.format(
                    config_section, attr, dtype, type(getattr(config, attr))
                )
            )
        return False
    return True


def test_attributes_range(
    config: DictConfig,
    value_ranges: List[ValueRange],
    config_section: str,
) -> bool:
    invalid_values = []
    for attr, value_range in value_ranges:
        if not getattr(config, attr) in value_range:
            invalid_values.append((attr, value_range))
    if len(invalid_values) > 0:
        for attr, value_range in invalid_values:
            logger.error(
                Errors.INVALID_VALUE.format(
                    config_section, attr, value_range.__repr__()
                )
            )
        return False
    return True


def test_path_exists(path) -> bool:
    if not os.path.exists(path):
        logger.error(Errors.PATH_NOT_FOUND.format(path))
        return False
    return True

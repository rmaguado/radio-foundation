from omegaconf import DictConfig, ListConfig
import logging

from .utils import (
    test_has_section,
    test_attributes_dtypes,
    test_attributes_range,
    ValueRange,
    Errors,
)

logger = logging.getLogger("dinov2")


def test_mask_ratio_min_max(ibot_config: DictConfig) -> bool:
    if not len(ibot_config.mask_ratio_min_max) == 2:
        logger.error(Errors.INVALID_LENGTH.format("ibot", "mask_ratio_min_max", 2))
        return False
    if not all(isinstance(val, float) for val in ibot_config.mask_ratio_min_max):
        logger.error(
            Errors.INVALID_TYPE.format(
                "ibot",
                "mask_ratio_min_max",
                "float",
                type(ibot_config.mask_ratio_min_max),
            )
        )
        return False
    mask_ratio_range = ValueRange(0.0, 1.0, right_inclusive=False)
    if not all(val in mask_ratio_range for val in ibot_config.mask_ratio_min_max):
        logger.error(
            Errors.INVALID_VALUE.format(
                "ibot", "mask_ratio_min_max", mask_ratio_range.__repr__()
            )
        )
        return False
    return True


def validate_ibot(config: DictConfig) -> bool:
    if not test_has_section(config, "ibot"):
        return False
    ibot_config = config.ibot
    required_attributes = [
        ("loss_weight", float),
        ("mask_sample_probability", float),
        ("mask_ratio_min_max", ListConfig),
        ("separate_head", bool),
        ("head_n_prototypes", int),
        ("head_bottleneck_dim", int),
        ("head_nlayers", int),
        ("head_hidden_dim", int),
    ]
    if not test_attributes_dtypes(ibot_config, required_attributes, "ibot"):
        return False

    attributes_ranges = [
        ("loss_weight", ValueRange(0.0)),
        ("mask_sample_probability", ValueRange(0.0, 1.0, right_inclusive=False)),
        ("head_n_prototypes", ValueRange(1)),
        ("head_bottleneck_dim", ValueRange(1)),
        ("head_nlayers", ValueRange(1)),
        ("head_hidden_dim", ValueRange(1)),
    ]
    if not test_attributes_range(ibot_config, attributes_ranges, "ibot"):
        return False
    if not test_mask_ratio_min_max(ibot_config):
        return False
    return True

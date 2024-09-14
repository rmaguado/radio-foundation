from omegaconf import DictConfig
import logging

from .utils import (
    test_has_section,
    test_attributes_dtypes,
    test_path_exists,
    Errors,
)
from .augmentations import validate_augmentations

logger = logging.getLogger("dinov2")


def test_dataset_type_is_valid(dataset_config: DictConfig) -> bool:
    dataset_type = dataset_config.type
    valid_dataset_types = ["ct"]
    if dataset_type not in valid_dataset_types:
        logger.error(Errors.INVALID_DATASET_TYPE.format(dataset_type))
        return False
    return True


def test_channels_is_valid(options_config: DictConfig, dataset_name: str) -> bool:
    channels = options_config.channels
    if not all(
        [
            isinstance(channels, int),
            channels > 0,
        ]
    ):
        logger.error(
            Errors.INVALID_VALUE.format(f"{dataset_name} channels: {channels}")
        )
        return False
    return True


def test_pixel_range_is_valid(pixel_range: DictConfig) -> bool:
    pixel_range_attributes = [
        ("lower", float),
        ("upper", float),
    ]
    if not test_attributes_dtypes(pixel_range, pixel_range_attributes, "pixel_range"):
        return False
    if not pixel_range.lower < pixel_range.upper:
        logger.error(Errors.INVALID_VALUE.format("pixel_range lower < upper"))
        return False
    return True


def test_norm_is_valid(norm: DictConfig) -> bool:
    norm_attributes = [
        ("mean", float),
        ("std", float),
    ]
    if not test_attributes_dtypes(norm, norm_attributes, "norm"):
        return False
    return True


def validate_dataset(config: DictConfig, dataset_config: DictConfig) -> bool:
    dataset_required_attributes = [
        ("name", str),
        ("index_path", str),
        ("type", str),
        ("channels", int),
        ("pixel_range", DictConfig),
        ("norm", DictConfig),
        ("augmentation", str),
    ]
    if not test_attributes_dtypes(
        dataset_config,
        dataset_required_attributes,
        (
            "unnamed dataset"
            if not hasattr(dataset_config, "name")
            else dataset_config.name
        ),
    ):
        return False

    if not all(
        [
            test_path_exists(dataset_config.index_path),
            test_dataset_type_is_valid(dataset_config),
            test_channels_is_valid(dataset_config, dataset_config.name),
            test_pixel_range_is_valid(dataset_config.pixel_range),
            test_norm_is_valid(dataset_config.norm),
        ]
    ):
        return False

    if not validate_augmentations(config, dataset_config):
        return False

    return True

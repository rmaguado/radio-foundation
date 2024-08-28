import os
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


def test_root_path_exists(data_config: DictConfig) -> None:
    datasets_root = data_config.root_path
    if not os.path.exists(datasets_root):
        logger.error(Errors.PATH_NOT_FOUND.format(datasets_root))
        return False
    return True


def test_has_dataset(data_config: DictConfig) -> None:
    if len(data_config.datasets) == 0:
        logger.error(Errors.NO_DATASETS_FOUND)
        return False
    return True


def validate_data(config: DictConfig) -> None:
    if not test_has_section(config, "data"):
        return False
    data_config = config.data
    required_attributes = [
        ("root_path", str),
        ("datasets", ListConfig),
    ]
    if not test_attributes_dtypes(data_config, required_attributes, "data"):
        return False
    if not test_root_path_exists(data_config):
        return False
    if not test_has_dataset(data_config):
        return False
    for dataset_config in data_config.datasets:
        dataset_required_attributes = [
            ("name", str),
            ("type", str),
            ("split", str),
            ("channels", int),
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
        attributes_ranges = [
            ("channels", ValueRange(1)),
        ]
        if not test_attributes_range(dataset_config, attributes_ranges, "dataset"):
            return False

    logger.info("'data' config is valid.")

    return True

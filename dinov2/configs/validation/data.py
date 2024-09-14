from omegaconf import DictConfig, ListConfig
import logging

from .utils import (
    test_has_section,
    test_attributes_dtypes,
    test_path_exists,
    Errors,
)
from .dataset import validate_dataset


logger = logging.getLogger("dinov2")


def test_has_dataset(data_config: DictConfig) -> bool:
    if len(data_config.datasets) == 0:
        logger.error(Errors.NO_DATASETS_FOUND)
        return False
    return True


def test_root_path_exists(data_config: DictConfig) -> bool:
    if not test_path_exists(data_config.root_path):
        return False
    return True


def validate_data(config: DictConfig) -> bool:
    if not test_has_section(config, "data"):
        return False

    data_config = config.data
    required_attributes = [
        ("root_path", str),
        ("datasets", ListConfig),
    ]
    if not test_attributes_dtypes(data_config, required_attributes, "data"):
        return False

    if not all([test_root_path_exists(data_config), test_has_dataset(data_config)]):
        return False
    if not all(
        [
            validate_dataset(config, dataset_config)
            for dataset_config in data_config.datasets
        ]
    ):
        return False

    logger.debug("'data' config is valid.")

    return True

from omegaconf import DictConfig, ListConfig
import logging

from .utils import (
    test_has_section,
    Errors,
)
from .dataset import validate_dataset_instance


logger = logging.getLogger("dinov2")


def test_has_dataset(data_config: ListConfig) -> bool:
    if len(data_config) == 0:
        logger.error(Errors.NO_DATASETS_FOUND)
        return False
    return True


def validate_data(config: DictConfig) -> bool:
    if not test_has_section(config, "datasets"):
        return False

    data_configs = config.datasets

    if not test_has_dataset(data_configs):
        return False
    if not all(
        [
            validate_dataset_instance(config, dataset_config)
            for dataset_config in data_configs
        ]
    ):
        return False

    logger.debug("'datasets' config is valid.")

    return True

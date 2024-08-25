import os
from omegaconf import DictConfig
import logging


logger = logging.getLogger("dinov2")


class Errors:
    NO_CONFIG = "Config has no data group."
    PATH_NOT_FOUND = "Path not found: {}"
    NO_DATASETS_FOUND = "No datasets specified in config."
    DATASET_MISSING_ATTR = "Missing attribute(s) in dataset {}: {}"
    SPLIT_FILE_NOT_FOUND = "Missing data index file: {}"


def test_has_data_config(config: DictConfig) -> None:
    if not hasattr(config, "data"):
        logger.error(Errors.NO_CONFIG)
        return False
    return True


def test_root_path_exists(config: DictConfig) -> None:
    if not hasattr(config.data, "root_path"):
        logger.error(Errors.MISSING_ATTR.format("root_path"))
        return False
    datasets_root = config.data.root_path
    if not os.path.exists(datasets_root):
        logger.error(Errors.PATH_NOT_FOUND.format(datasets_root))
        return False
    return True


def test_has_dataset(config: DictConfig) -> None:
    if hasattr(config.data, "datasets"):
        if len(config.data.datasets) > 0:
            return True
    logger.error(Errors.NO_DATASETS_FOUND)
    return False


def test_dataset_attributes_exist(datasetConfig: DictConfig) -> None:
    missing_attributes = []
    for attr in ["name", "split", "type"]:
        if not hasattr(datasetConfig, "name"):
            missing_attributes.append(attr)
    if hasattr(datasetConfig, "type") and not hasattr(datasetConfig, "num_slices"):
        missing_attributes.append("num_slices")
    if len(missing_attributes) > 0:
        logger.error(
            Errors.DATASET_MISSING_ATTR.format(
                datasetConfig.name, ", ".join(missing_attributes)
            )
        )
        return False
    return True


def test_dataset_data_exist(datasetConfig: DictConfig, root_path: str) -> None:
    if not hasattr(datasetConfig, "name"):
        return False
    data_path = os.path.join(root_path, datasetConfig.name, "data")
    if not os.path.exists(data_path):
        logger.error(Errors.PATH_NOT_FOUND.format(data_path))
        return False
    return True


def test_dataset_extra_exist(datasetConfig: DictConfig, root_path: str) -> None:
    if not hasattr(datasetConfig, "name"):
        return False
    extra_path = os.path.join(root_path, datasetConfig.name, "extra")
    if not os.path.exists(extra_path):
        logger.error(Errors.PATH_NOT_FOUND.format(extra_path))
        return False
    if hasattr(datasetConfig, "split"):
        split_file = os.path.join(extra_path, f"{datasetConfig.split}.json")
        if not os.path.exists(split_file):
            logger.error(Errors.SPLIT_FILE_NOT_FOUND.format(datasetConfig.name))
            return False
        return True
    return False


def validate_data(config: DictConfig) -> None:
    if not test_has_data_config(config):
        return False
    if not test_root_path_exists(config):
        return False
    if not test_has_dataset(config):
        return False
    root_path = config.data.root_path
    for datasetConfig in config.data.datasets:
        return all(
            [
                test_dataset_attributes_exist(datasetConfig, root_path),
                test_dataset_data_exist(datasetConfig, root_path),
                test_dataset_extra_exist(datasetConfig, root_path),
            ]
        )
    return True

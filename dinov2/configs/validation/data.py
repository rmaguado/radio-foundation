from omegaconf import DictConfig, ListConfig
import logging

from .utils import (
    test_has_section,
    test_attributes_dtypes,
    test_attributes_range,
    test_path_exists,
    ValueRange,
    Errors,
)


logger = logging.getLogger("dinov2")


def test_has_dataset(data_config: DictConfig) -> None:
    if len(data_config.datasets) == 0:
        logger.error(Errors.NO_DATASETS_FOUND)
        return False
    return True


def test_dataset_options_is_valid(dataset_config: DictConfig) -> None:
    dataset_name = dataset_config.name
    options_config = dataset_config.options
    dataset_type = dataset_config.type

    if dataset_type == "ct":
        if hasattr(options_config, "channels"):
            channels = options_config.channels
            if not isinstance(channels, int):
                logger.error(
                    Errors.INVALID_CHANNELS.format(
                        dataset_config.name, channels
                    )
                )
                return False
            if options_config.channels < 1:
                logger.error(
                    Errors.INVALID_CHANNELS.format(
                        dataset_config.name, channels
                    )
                )
                return False
        if hasattr(options_config, "lower_window"):
            if not test_attributes_dtypes(options_config, [("lower_window", float)], dataset_name):
                return False
        if hasattr(options_config, "upper_window"):
            if not test_attributes_dtypes(options_config, [("upper_window", float)], dataset_name):
                return False
        if hasattr(options_config, "lower_window") and hasattr(
            options_config, "upper_window"
        ):
            if options_config.lower_window >= options_config.upper_window:
                logger.error(
                    Errors.INVALID_VALUE_PAIR.format(
                        options_config.lower_window, options_config.upper_window
                    )
                )
                return False
    return True


def test_dataset_is_valid(dataset_config: DictConfig) -> None:
    dataset_required_attributes = [
        ("name", str),
        ("index_path", str),
        ("type", str),
    ]
    acceptable_dataset_types = ["ct"]
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
    if dataset_config.type not in acceptable_dataset_types:
        logger.error(Errors.INVALID_DATASET_TYPE.format(dataset_config.type))
        return False
    if hasattr(dataset_config, "options"):
        if not test_dataset_options_is_valid(dataset_config):
            return False
    if not test_path_exists(dataset_config.index_path):
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
    if not test_has_dataset(data_config):
        return False
    if not all(
        [
            test_dataset_is_valid(dataset_config) for dataset_config in data_config.datasets
        ]
    ):
        return False

    logger.debug("'data' config is valid.")

    return True

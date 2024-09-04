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


def test_has_dataset(data_config: DictConfig) -> bool:
    if len(data_config.datasets) == 0:
        logger.error(Errors.NO_DATASETS_FOUND)
        return False
    return True


def test_ct_channels_is_valid(options_config: DictConfig, dataset_name: str) -> bool:
    channels = options_config.channels
    error = Errors.INVALID_CHANNELS.format(dataset_name, channels)
    if not isinstance(channels, int):
        logger.error(error)
        return False
    if channels < 1:
        logger.error(error)
        return False
    return True


def test_ct_windows_is_valid(
    lower_window: float, upper_window: float, dataset_name: str
) -> bool:
    if lower_window >= upper_window:
        logger.error(
            Errors.INVALID_VALUE_PAIR.format(dataset_name, lower_window, upper_window)
        )
        return False
    return True


def test_ct_options_is_valid(options_config: DictConfig, dataset_name: str) -> bool:
    if hasattr(options_config, "channels"):
        if not test_ct_channels_is_valid(options_config, dataset_name):
            return False
    lower_window = options_config.get("lower_window")
    upper_window = options_config.get("upper_window")
    if lower_window or upper_window:
        if not test_attributes_dtypes(
            options_config,
            [
                ("lower_window", float),
                ("upper_window", float),
            ],
            dataset_name + ".options",
        ):
            return False

    if lower_window and upper_window:
        if not test_ct_windows_is_valid(lower_window, upper_window, dataset_name):
            return False
    return True


def test_dataset_options_is_valid(dataset_config: DictConfig) -> bool:
    dataset_name = dataset_config.name
    options_config = dataset_config.options
    dataset_type = dataset_config.type

    if dataset_type == "ct":
        if not test_ct_options_is_valid(options_config, dataset_name):
            return False
    return True


def test_dataset_is_valid(dataset_config: DictConfig) -> bool:
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
    if not test_has_dataset(data_config):
        return False
    if not all(
        [
            test_dataset_is_valid(dataset_config)
            for dataset_config in data_config.datasets
        ]
    ):
        return False

    logger.debug("'data' config is valid.")

    return True

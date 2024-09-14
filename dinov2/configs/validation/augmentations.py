from omegaconf import DictConfig, ListConfig
from typing import Dict
import logging
import copy

from dinov2.data import ImageTransforms

from .utils import (
    test_has_section,
    test_attributes_dtypes,
    Errors,
)

logger = logging.getLogger("dinov2")


def test_has_crops(augmentations_list: DictConfig, required_crop: str) -> bool:
    if not any([transform.name == required_crop for transform in augmentations_list]):
        logger.error(Errors.NO_CROP)
        return False
    return True


def test_transform_is_valid(
    transform_obj: ImageTransforms, transform_kwargs: Dict
) -> bool:
    transform_kwargs = copy.deepcopy(transform_kwargs)
    if not hasattr(transform_kwargs, "name"):
        logger.error(Errors.NO_TRANSFORM_NAME)
        return False
    transform_name = transform_kwargs.pop("name")
    if transform_name in ["localcrop", "globalcrop"]:
        return True
    if transform_name not in transform_obj.keys():
        logger.error(Errors.UNRECOGNIZED_TRANSFORM.format(transform_name))
        return False
    try:
        transform_obj._get_random_transform(transform_name, transform_kwargs)
    except Exception as e:
        logger.error(Errors.TRANSFORM_INIT_ERROR.format(transform_name, e))
        return False
    return True


def test_transforms_list(
    transform_obj: ImageTransforms,
    augmentations_list: DictConfig,
    augmentation_group: str,
) -> bool:
    if not all(
        [
            test_transform_is_valid(transform_obj, kwargs)
            for kwargs in augmentations_list
        ]
    ):
        return False

    required_crop = (
        "globalcrop" if augmentation_group in ["global_1", "global_2"] else "localcrop"
    )
    if not test_has_crops(augmentations_list, required_crop):
        return False

    return True


def dataset_has_augmentation(
    augmentations_config: DictConfig, augmentation_preset: str
) -> bool:
    if augmentation_preset not in augmentations_config:
        logger.error(Errors.NO_AUGMENTATION_PRESET.format(augmentation_preset))
        return False
    return True


def validate_augmentations(config: DictConfig, dataset_config) -> bool:
    if not test_has_section(config, "augmentations"):
        return False

    augmentations_config = config.augmentations
    augmentations_preset = dataset_config.augmentation
    if not dataset_has_augmentation(augmentations_config, augmentations_preset):
        return False

    selected_augmentations = augmentations_config[augmentations_preset]

    required_attributes = [
        ("global_1", ListConfig),
        ("global_2", ListConfig),
        ("local", ListConfig),
    ]
    if not test_attributes_dtypes(
        selected_augmentations,
        required_attributes,
        f"augmentations: {augmentations_preset}",
    ):
        return False
    lower_bound = dataset_config.pixel_range.lower
    upper_bound = dataset_config.pixel_range.upper
    channels = dataset_config.channels
    transforms_obj = ImageTransforms(lower_bound, upper_bound, channels)
    for group in ["global_1", "global_2", "local"]:
        if not test_transforms_list(
            transforms_obj, selected_augmentations[group], group
        ):
            return False

    logger.debug("'augmentations' config is valid.")
    return True

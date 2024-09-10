from omegaconf import DictConfig, ListConfig
from typing import Dict
import logging

from dinov2.data import ImageTransforms

from .utils import (
    test_has_section,
    test_attributes_dtypes,
    test_attributes_range,
    ValueRange,
    Errors,
)

logger = logging.getLogger("dinov2")


def test_has_crops(augmentations_list: DictConfig) -> bool:
    if not any(
        [
            "localcrop" in transform.name or "globalcrop" in transform.name
            for transform in augmentations_list
        ]
    ):
        logger.error(Errors.NO_CROP)
        return False
    return True


def test_transform_is_valid(
    transform_obj: ImageTransforms, transform_kwargs: Dict
) -> bool:
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
    if not all([test_transform_is_valid(transform_obj, augmentations_list)]):
        return False

    required_crop = (
        "global" if augmentation_group in ["global_1", "global_2"] else "local"
    )
    if not test_has_crops(augmentations_list, required_crop):
        return False

    return True


def validate_augmentations(dataset_config: DictConfig) -> bool:
    if not test_has_section(dataset_config, "augmentations"):
        return False
    augmentations_config = dataset_config.augmentations
    required_attributes = [
        ("global_1", ListConfig),
        ("global_2", ListConfig),
        ("local", ListConfig),
    ]
    if not test_attributes_dtypes(
        augmentations_config, required_attributes, "augmentations"
    ):
        return False
    lower_bound = dataset_config.pixel_range.lower
    upper_bound = dataset_config.pixel_range.upper
    channels = dataset_config.channels
    transforms_obj = ImageTransforms(lower_bound, upper_bound, channels)
    for group in ["global_1", "global_2", "local"]:
        if not test_transforms_list(transforms_obj, augmentations_config[group], group):
            return False

    logger.debug("'augmentations' config is valid.")
    return True

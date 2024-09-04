from omegaconf import DictConfig, ListConfig
from typing import Dict
import logging

from dinov2.data.transforms import transformkeys

from .utils import (
    test_has_section,
    test_attributes_dtypes,
    test_attributes_range,
    ValueRange,
    Errors,
)

logger = logging.getLogger("dinov2")


def test_transform_is_valid(transform_name: str, transform_kwargs: Dict) -> bool:
    if transform_name in ["localcrop", "globalcrop"]:
        return True
    if transform_name not in transformkeys.keys():
        logger.error(Errors.UNRECOGNIZED_TRANSFORM.format(transform_name))
        return False
    try:
        transformkeys[transform_name](**transform_kwargs)
    except Exception as e:
        logger.error(Errors.TRANSFORM_INIT_ERROR.format(transform_name, e))
        return False
    return True


def test_transforms_list(
    augmentations_list: DictConfig, augmentations_group: str
) -> bool:
    problem_transforms = []
    for transform_dict in augmentations_list:
        transform_name = (
            transform_dict.name
            if hasattr(transform_dict, "name")
            else "unnamed transform"
        )
        transform_kwargs = {k: v for k, v in transform_dict.items() if k != "name"}
        if not test_transform_is_valid(transform_name, transform_kwargs):
            problem_transforms.append(transform_name)

    if len(problem_transforms) > 0:
        return False
    return True


def test_has_crops(augmentations_config: DictConfig) -> bool:
    required_transforms = {
        "global1": "globalcrop",
        "global2": "globalcrop",
        "local": "localcrop",
    }
    for augmentations_group, transforms_list in augmentations_config.items():
        transform_names = [t.name for t in transforms_list if hasattr(t, "name")]
        if required_transforms[augmentations_group] not in transform_names:
            logger.error(
                Errors.MISSING_ATTR.format(
                    "augmentations." + augmentations_group,
                    required_transforms[augmentations_group],
                )
            )
            return False


def validate_augmentations(config: DictConfig) -> bool:
    if not test_has_section(config, "augmentations"):
        return False
    augmentations_config = config.augmentations
    required_attributes = [
        ("global_1", ListConfig),
        ("global_2", ListConfig),
        ("local", ListConfig),
    ]
    if not test_attributes_dtypes(
        augmentations_config, required_attributes, "augmentations"
    ):
        return False
    for group in ["global_1", "global_2", "local"]:
        if not test_transforms_list(augmentations_config[group], group):
            return False
    if not test_has_crops(augmentations_config):
        return False

    logger.debug("'augmentations' config is valid.")
    return True

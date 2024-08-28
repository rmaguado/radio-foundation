from omegaconf import DictConfig
import logging

from .utils import (
    test_has_section,
    test_attributes_dtypes,
    test_attributes_range,
    ValueRange,
    Errors,
)

logger = logging.getLogger("dinov2")


def test_weight_decay(optim_config: DictConfig) -> bool:
    if optim_config.weight_decay > optim_config.weight_decay_end:
        logger.error(
            Errors.INVALID_VALUE_PAIR.format(
                "optim", "weight_decay", "weight_decay_end"
            )
        )
        return False
    return True


def validate_optim(config: DictConfig) -> bool:
    if not test_has_section(config, "optim"):
        return False
    optim_config = config.optim
    required_attributes = [
        ("epochs", int),
        ("weight_decay", float),
        ("weight_decay_end", float),
        ("base_lr", float),
        ("warmup_epochs", int),
        ("min_lr", float),
        ("clip_grad", float),
        ("freeze_last_layer_epochs", int),
        ("patch_embed_lr_mult", float),
        ("layerwise_decay", float),
        ("adamw_beta1", float),
        ("adamw_beta2", float),
    ]
    if not test_attributes_dtypes(optim_config, required_attributes, "optim"):
        return False

    attributes_ranges = [
        ("epochs", ValueRange(1)),
        ("weight_decay", ValueRange(0, 1)),
        ("weight_decay_end", ValueRange(0, 1)),
        ("base_lr", ValueRange(0, 1, left_inclusive=False)),
        ("warmup_epochs", ValueRange(0)),
        ("min_lr", ValueRange(0, 1, left_inclusive=False)),
        ("clip_grad", ValueRange(0)),
        ("freeze_last_layer_epochs", ValueRange(0)),
        ("patch_embed_lr_mult", ValueRange(0)),
        ("layerwise_decay", ValueRange(0, left_inclusive=False)),
        ("adamw_beta1", ValueRange(0, 1, left_inclusive=False, right_inclusive=False)),
        ("adamw_beta2", ValueRange(0, 1, left_inclusive=False, right_inclusive=False)),
    ]
    if not all(
        [
            test_attributes_range(optim_config, attributes_ranges, "optim"),
            test_weight_decay(optim_config),
        ]
    ):
        return False
    logger.debug("'optim' config is valid.")
    return True

from omegaconf import DictConfig

from .utils import (
    test_has_section,
    test_attributes_dtypes,
    test_attirbutes_range,
    ValueRange,
)


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
        ("scaling_rule", str),
        ("patch_embed_lr_mult", float),
        ("layerwise_decay", float),
        ("adamw_beta1", float),
        ("adamw_beta2", float),
    ]
    if not test_attributes_dtypes(optim_config, required_attributes, "optim"):
        return False

    attributes_ranges = [
        ("epochs", ValueRange(1)),
        ("weight_decay", ValueRange(0)),
        ("weight_decay_end", ValueRange(0)),
        ("base_lr", ValueRange(0, left_inclusive=False)),
        ("warmup_epochs", ValueRange(0)),
        ("min_lr", ValueRange(0, left_inclusive=False)),
        ("clip_grad", ValueRange(0, left_inclusive=False)),
        ("freeze_last_layer_epochs", ValueRange(0)),
        ("patch_embed_lr_mult", ValueRange(0)),
        ("layerwise_decay", ValueRange(0, left_inclusive=False)),
        ("adamw_beta1", ValueRange(0, 1, left_inclusive=False, right_inclusive=False)),
        ("adamw_beta2", ValueRange(0, 1, left_inclusive=False, right_inclusive=False)),
    ]
    if not test_attirbutes_range(optim_config, attributes_ranges, "optim"):
        return False
    return True

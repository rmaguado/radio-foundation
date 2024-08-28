from omegaconf import DictConfig

from .utils import (
    test_has_section,
    test_attributes_dtypes,
    test_attributes_range,
    ValueRange,
)


def validate_dino(config: DictConfig) -> bool:
    if not test_has_section(config, "dino"):
        return False
    dino_config = config.dino

    required_attributes = [
        ("loss_weight", float),
        ("head_n_prototypes", int),
        ("head_bottleneck_dim", int),
        ("head_nlayers", int),
        ("head_hidden_dim", int),
        ("koleo_loss_weight", float),
    ]
    if not test_attributes_dtypes(dino_config, required_attributes, "dino"):
        return False

    attributes_ranges = [
        ("loss_weight", ValueRange(0.0)),
        ("head_n_prototypes", ValueRange(1)),
        ("head_bottleneck_dim", ValueRange(1)),
        ("head_nlayers", ValueRange(1)),
        ("head_hidden_dim", ValueRange(1)),
        ("koleo_loss_weight", ValueRange(0.0)),
    ]
    if not test_attributes_range(dino_config, attributes_ranges, "dino"):
        return False

    logger.info("'dino' config is valid.")

    return True

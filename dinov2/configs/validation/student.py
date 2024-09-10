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


def test_architecture(student_config: DictConfig) -> bool:
    valid_architectures = ["vit_small", "vit_base", "vit_large", "vit_giant2"]
    if student_config.arch not in valid_architectures:
        logger.error(
            Errors.INVALID_VALUE.format(
                "student", "arch", ",".join(valid_architectures)
            )
        )
        return False
    return True


def test_vit_small_drop_path(student_config: DictConfig) -> bool:
    if student_config.arch != "vit_small":
        return True
    if student_config.drop_path_rate > 0:
        logger.error(
            Errors.VIT_SMALL_DROP_PATH,
        )
        return False
    return True


def test_ffn_layer(student_config: DictConfig) -> bool:
    valid_ffn_layers = ["swiglu", "mlp"]
    if student_config.ffn_layer not in valid_ffn_layers:
        logger.error(
            Errors.INVALID_VALUE.format(
                "student", "ffn_layer", ",".join(valid_ffn_layers)
            )
        )
        return False
    return True


def validate_student(config: DictConfig) -> bool:
    if not test_has_section(config, "student"):
        return False
    student_config = config.student
    required_attributes = [
        ("arch", str),
        ("patch_size", int),
        ("full_image_size", int),
        ("channels", int),
        ("drop_path_rate", float),
        ("layerscale", float),
        ("drop_path_uniform", bool),
        ("pretrained_weights", str),
        ("ffn_layer", str),
        ("block_chunks", int),
        ("qkv_bias", bool),
        ("proj_bias", bool),
        ("ffn_bias", bool),
        ("embed_layer", bool),
        ("num_register_tokens", int),
        ("interpolate_antialias", bool),
        ("interpolate_offset", float),
    ]
    if not test_attributes_dtypes(student_config, required_attributes, "student"):
        return False

    attributes_ranges = [
        ("patch_size", ValueRange(1)),
        ("full_image_size", ValueRange(1)),
        ("channels", ValueRange(1)),
        ("drop_path_rate", ValueRange(0.0, 1.0, right_inclusive=False)),
        ("layerscale", ValueRange(0.0)),
        ("block_chunks", ValueRange(0)),
        ("num_register_tokens", ValueRange(1)),
        ("interpolate_offset", ValueRange(0.0, 1.0)),
    ]
    if not test_attributes_range(student_config, attributes_ranges, "student"):
        return False

    if not all(
        [
            test_architecture(student_config),
            test_vit_small_drop_path(student_config),
            test_ffn_layer(student_config),
        ]
    ):
        return False
    logger.debug("'student' config is valid.")
    return True

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


def test_epoch_nums_config(config: DictConfig) -> bool:
    if not test_has_section(config, "optim"):
        return False
    if not test_has_section(config, "teacher"):
        return False
    if not test_has_section(config.train, "full_image"):
        return False

    if not test_attributes_dtypes(
        config.optim,
        [
            ("epochs", int),
            ("freeze_last_layer_epochs", int),
            ("warmup_epochs", int),
        ],
        "optim",
    ):
        return False
    if not test_attributes_dtypes(
        config.teacher, [("warmup_teacher_temp_epochs", int)], "teacher"
    ):
        return False
    if not test_attributes_dtypes(
        config.train.full_image, [("epochs", int)], "full_image"
    ):
        return False

    epochs = config.optim.epochs
    freeze_last_layer_epochs = config.optim.freeze_last_layer_epochs
    warmup_epochs = config.optim.warmup_epochs
    warmup_teacher_temp_epochs = config.teacher.warmup_teacher_temp_epochs
    full_image_epochs = config.train.full_image.epochs

    is_valid = True
    if freeze_last_layer_epochs > epochs:
        logger.error(
            Errors.INVALID_VALUE.format(
                "optim", "freeze_last_layer_epochs", "less than epochs"
            )
        )
        is_valid = False
    if warmup_epochs > epochs:
        logger.error(
            Errors.INVALID_VALUE.format("optim", "warmup_epochs", "less than epochs")
        )
        is_valid = False
    if warmup_teacher_temp_epochs > epochs:
        logger.error(
            Errors.INVALID_VALUE.format(
                "teacher", "warmup_teacher_temp_epochs", "less than epochs"
            )
        )
        is_valid = False
    if full_image_epochs > epochs:
        logger.error(
            Errors.INVALID_VALUE.format(
                "train.full_image", "epochs", "less than epochs"
            )
        )
        is_valid = False

    return is_valid


def test_full_image_config(train_config: DictConfig) -> bool:
    full_image_config = train_config.full_image

    required_attributes = [
        ("epochs", int),
        ("batch_size_per_gpu", int),
        ("grad_accum_steps", int),
    ]
    if not test_attributes_dtypes(
        full_image_config,
        required_attributes,
        "full_image",
    ):
        return False

    attributes_ranges = [
        ("epochs", ValueRange(1)),
        ("batch_size_per_gpu", ValueRange(1)),
        ("grad_accum_steps", ValueRange(1)),
    ]
    if not test_attributes_range(full_image_config, attributes_ranges, "full_image"):
        return False
    return True


def test_centering_config(train_config: DictConfig) -> bool:
    centering_modes = ["centering", "sinkhorn_knopp"]
    if train_config.centering not in centering_modes:
        logger.error(
            Errors.INVALID_VALUE.format("train", "centering", train_config.centering)
        )
        return False
    return True


def validate_train(config: DictConfig) -> bool:
    if not test_has_section(config, "train"):
        return False
    train_config = config.train
    required_attributes = [
        ("grad_accum_steps", int),
        ("batch_size_per_gpu", int),
        ("output_dir", str),
        ("saveckp_iterations", int),
        ("print_freq", int),
        ("seed", int),
        ("num_workers", int),
        ("OFFICIAL_EPOCH_LENGTH", int),
        ("cache_dataset", bool),
        ("centering", str),
        ("full_image", DictConfig),
    ]
    if not test_attributes_dtypes(train_config, required_attributes, "train"):
        return False

    attributes_ranges = [
        ("grad_accum_steps", ValueRange(1)),
        ("batch_size_per_gpu", ValueRange(1)),
        ("saveckp_iterations", ValueRange(1)),
        ("print_freq", ValueRange(1)),
        ("num_workers", ValueRange(0)),
        ("OFFICIAL_EPOCH_LENGTH", ValueRange(1)),
    ]
    if not test_attributes_range(train_config, attributes_ranges, "train"):
        return False

    if not all(
        [
            test_epoch_nums_config(config),
            test_full_image_config(train_config),
            test_centering_config(train_config),
        ]
    ):
        return False
    logger.debug("'train' config is valid.")
    return True

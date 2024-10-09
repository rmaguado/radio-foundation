# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
from typing import Any, Callable, Tuple, List, Optional, TypeVar
from omegaconf import DictConfig

import torch
from torch.utils.data import Sampler

from .datasets import MultiDataset, DicomCtDataset, NiftiCtDataset, MedicalImageDataset
from .samplers import (
    InfiniteSampler,
    WeightedInfiniteSampler,
    ShardedInfiniteSampler,
    WeightedShardedInfiniteSampler,
)
from .augmentations import DataAugmentationDINO


logger = logging.getLogger("dinov2")


class SamplerType(Enum):
    INFINITE = 0
    SHARDED_INFINITE = 1
    WEIGHTED_INFINITE = 2
    WEIGHTED_SHARDED_INFINITE = 3


def make_train_dataset(
    config: DictConfig,
    use_full_image: bool,
) -> Tuple[MedicalImageDataset, Optional[List[float]]]:
    """
    Parse the dataset from the given OmegaConf configuration.

    Args:
        config (DictConfig): The OmegaConf dictionary configuration for the dataset.
        use_full_image (bool): Whether to set the global crop size to the full size.

    Returns:
        MedicalImageDataset: The corresponding dataset object(s).
    """

    dataset_objects = []
    weights = []

    for dataset_config in config.datasets:
        dataset_object, weight = build_dataset_from_cfg(
            config, use_full_image, dataset_config
        )
        dataset_objects.append(dataset_object)
        weights.append(weight)
    if any(weight is None for weight in weights):
        weights = None

    if len(dataset_objects) > 1:
        logger.info("Multiple datasets detected. Using MultiDataset.")
        return MultiDataset(datasets=dataset_objects), weights
    return dataset_objects[0], [1.0]


def build_dataset_from_cfg(
    config, use_full_image, dataset_config
) -> Tuple[MedicalImageDataset, Optional[float]]:

    def get_ct_kwargs(dataset_config):
        return {
            "channels": dataset_config.channels,
            "lower_window": dataset_config.pixel_range.lower,
            "upper_window": dataset_config.pixel_range.upper,
        }

    dataset_type = dataset_config.type
    dataset_storage = dataset_config.storage
    transform = DataAugmentationDINO(config, dataset_config, use_full_image)

    weight = dataset_config.weight if hasattr(dataset_config, "weight") else None

    dataset_kwargs = {
        "dataset_name": dataset_config.name,
        "root_path": dataset_config.root_path,
        "output_path": config.train.output_dir,
        "transform": transform,
    }

    if dataset_type == "ct":
        dataset_kwargs.update(get_ct_kwargs(dataset_config))
        if dataset_storage == "dicom":
            dataset_object = DicomCtDataset(**dataset_kwargs)
        elif dataset_storage == "nifti":
            dataset_object = NiftiCtDataset(**dataset_kwargs)
        else:
            raise ValueError(f"Unsupported dataset storage: {dataset_storage}")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    return dataset_object, weight


def _make_sampler(
    *,
    dataset,
    weights: Optional[List[float]] = None,
    sampler_type: Optional[SamplerType] = None,
    seed: int = 0,
) -> Optional[Sampler]:
    """
    Creates a sampler with the specified parameters.
    A sampler is a strategy for sampling data from a dataset.
    Supported sampler types includes:
        - INFINITE, SHARDED_INFINITE, WEIGHTED_INFINITE, WEIGHTED_SHARDED_INFINITE.

    Args:
        dataset: The dataset to create the sampler for.
        weights (Optional[List[float]]): The weights for each dataset if using multiple groups of data. Defaults to None.
        sampler_type (Optional[SamplerType]): The type of sampler to create. Defaults to None.
        seed (int): The random seed for shuffling. Defaults to 0.

    Returns:
        Optional[Sampler]: The created sampler or None if no sampler is created.
    """
    if isinstance(dataset, MultiDataset):
        dataset_sizes = dataset.get_dataset_sizes()
    else:
        dataset_sizes = [len(dataset)]
    sample_count = len(dataset)

    if sampler_type in [
        SamplerType.WEIGHTED_INFINITE,
        SamplerType.WEIGHTED_SHARDED_INFINITE,
    ]:
        assert weights is not None, "Weights must be provided for weighted sampling"

    if sampler_type == SamplerType.INFINITE:
        logger.info("sampler: infinite")
        return InfiniteSampler(sample_count=sample_count, seed=seed)
    elif sampler_type == SamplerType.SHARDED_INFINITE:
        logger.info("sampler: sharded infinite")
        return ShardedInfiniteSampler(sample_count=sample_count, seed=seed)
    elif sampler_type == SamplerType.WEIGHTED_INFINITE:
        logger.info("sampler: weighted infinite")
        return WeightedInfiniteSampler(
            dataset_sizes=dataset_sizes, weights=weights, seed=seed
        )
    elif sampler_type == SamplerType.WEIGHTED_SHARDED_INFINITE:
        logger.info("sampler: weighted sharded infinite")
        return WeightedShardedInfiniteSampler(
            dataset_sizes=dataset_sizes, weights=weights, seed=seed
        )

    logger.info("sampler: none")
    return None


T = TypeVar("T")


def make_data_loader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    seed: int = 0,
    weights: Optional[List[float]] = None,
    sampler_type: Optional[SamplerType] = SamplerType.INFINITE,
    drop_last: bool = True,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable[[List[T]], Any]] = None,
):
    """
    Creates a data loader with the specified parameters.

    Args:
        dataset: A dataset (third party, LaViDa or WebDataset).
        batch_size: The size of batches to generate.
        num_workers: The number of workers to use.
        seed: The random seed to use.
        weights: The weights for each dataset if using multiple groups of data.
        sampler_type: Which sampler to use: EPOCH, INFINITE, SHARDED_INFINITE, DISTRIBUTED or None.
        sampler_size: The number of images per epoch (when applicable) or -1 for the entire dataset.
        drop_last: Whether the last non-full batch of data should be dropped.
        persistent_workers: maintain the workers Dataset instances alive after a dataset has been consumed once.
        collate_fn: Function that performs batch collation
    """

    sampler = _make_sampler(
        dataset=dataset,
        weights=weights,
        sampler_type=sampler_type,
        seed=seed,
    )

    logger.info("using PyTorch data loader")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    try:
        logger.info(f"# of batches: {len(data_loader):,d}")
    except TypeError:
        logger.info("infinite data loader")
    return data_loader

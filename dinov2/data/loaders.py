# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
from typing import Any, Callable, List, Optional, TypeVar
from omegaconf import DictConfig
from tabulate import tabulate

from torch.utils.data import Sampler, DataLoader

from .datasets import (
    MultiDataset,
    DicomVolumeDataset,
    NiftiVolumeDataset,
    TorchVolumeDataset,
)
from .samplers import (
    InfiniteSampler,
    WeightedInfiniteSampler,
)
from .augmentations import DataAugmentationDINO

# from .dataloader import ParallelSampleDataLoader


logger = logging.getLogger("dinov2")


class SamplerType(Enum):
    INFINITE = 0
    WEIGHTED_INFINITE = 2


def make_train_dataset(
    config: DictConfig,
):
    """
    Parse the dataset from the given OmegaConf configuration.

    Args:
        config (DictConfig): The OmegaConf dictionary configuration for the dataset.

    Returns:
        MedicalImageDataset: The corresponding dataset object(s).
    """

    dataset_objects = []
    weights = []

    for dataset_config in config.datasets:
        dataset_object, weight = build_dataset_from_cfg(config, dataset_config)
        dataset_objects.append(dataset_object)
        weights.append(weight)
    if any(weight is None for weight in weights):
        weights = None

    if len(dataset_objects) > 1:
        logger.info("Multiple datasets detected. Using MultiDataset.")
        return MultiDataset(datasets=dataset_objects), weights
    return dataset_objects[0], [1.0]


def build_dataset_from_cfg(cfg, dataset_config):
    dataset_storage = dataset_config.storage

    weight = dataset_config.weight if hasattr(dataset_config, "weight") else None
    norm = dataset_config.norm
    transforms = DataAugmentationDINO(cfg, norm.mean, norm.std)

    dataset_kwargs = {
        "config": cfg,
        "dataset_name": dataset_config.name,
        "index_path": dataset_config.index_path,
        "modality": dataset_config.type,
        "transforms": transforms,
        "bounds": dataset_config.bounds,
    }

    if dataset_storage == "dicom":
        dataset_object = DicomVolumeDataset(**dataset_kwargs)
    elif dataset_storage == "nifti":
        dataset_object = NiftiVolumeDataset(**dataset_kwargs)
    elif dataset_storage == "torch":
        dataset_object = TorchVolumeDataset(**dataset_kwargs)
    else:
        raise ValueError(f"Unsupported dataset storage: {dataset_storage}")

    return dataset_object, weight


def _make_sampler(
    dataset,
    weights: Optional[List[float]] = None,
    sampler_type: Optional[SamplerType] = None,
    seed: int = 0,
    advance: int = 0,
) -> Optional[Sampler]:
    """
    Creates a sampler with the specified parameters.
    A sampler is a strategy for sampling data from a dataset.
    Supported sampler types includes:
        - INFINITE
        - WEIGHTED_INFINITE

    Args:
        dataset: The dataset to create the sampler for.
        weights (Optional[List[float]]): The weights for each dataset if using multiple groups of data. Defaults to None.
        sampler_type (Optional[SamplerType]): The type of sampler to create. Defaults to None.
        seed (int): The random seed for shuffling. Defaults to 0.
        advance (int): number of sampler steps to skip.

    Returns:
        Optional[Sampler]: The created sampler or None if no sampler is created.
    """
    if isinstance(dataset, MultiDataset):
        dataset_sizes = dataset.get_dataset_sizes()
        dataset_names = dataset.get_dataset_names()
    else:
        dataset_sizes = [len(dataset)]
        dataset_names = [dataset.dataset_name]
    sample_count = len(dataset)

    if sampler_type == SamplerType.WEIGHTED_INFINITE:
        assert weights is not None, "Weights must be provided for weighted sampling"

    if weights is not None:
        logger.info("Using weighted sampling.")
        table = [
            ["Dataset", "Size", "Weight"],
            *[
                [dataset_name, f"{size:,d}", weight]
                for dataset_name, size, weight in zip(
                    dataset_names, dataset_sizes, weights
                )
            ],
        ]
    else:
        logger.info(
            "Not using weighted sampling. Provide a weight for each dataset to enable."
        )
        table = [
            ["Dataset", "Size"],
            *[
                [dataset_name, f"{size:,d}"]
                for dataset_name, size in zip(dataset_names, dataset_sizes)
            ],
        ]

    logger.info("\n" + tabulate(table, headers="firstrow", tablefmt="orgtbl"))

    if sampler_type == SamplerType.INFINITE:
        logger.info("sampler: infinite")
        return InfiniteSampler(sample_count=sample_count, seed=seed, advance=advance)
    elif sampler_type == SamplerType.WEIGHTED_INFINITE:
        logger.info("sampler: weighted infinite")
        assert weights is not None
        return WeightedInfiniteSampler(
            dataset_names=dataset_names,
            sizes=dataset_sizes,
            weights=weights,
            seed=seed,
            advance=advance,
        )

    logger.info("sampler: none")
    return None


def make_data_loader(
    dataset,
    batch_size_per_gpu: int,
    batch_size_total: int,
    num_workers: int,
    iteration: int = 0,
    seed: int = 0,
    weights: Optional[List[float]] = None,
    sampler_type: Optional[SamplerType] = SamplerType.INFINITE,
    drop_last: bool = True,
    persistent_workers: bool = True,
    collate_fn: Optional[Callable] = None,
):
    """
    Creates a data loader with the specified parameters.

    Args:
        dataset: A dataset (third party, LaViDa or WebDataset).
        batch_size: The size of batches to generate.
        num_workers: The number of workers to use.
        seed: The random seed to use.
        weights: The weights for each dataset if using multiple groups of data.
        sampler_type: Which sampler to use: EPOCH, INFINITE, DISTRIBUTED or None.
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
        advance=iteration * batch_size_total,
    )

    logger.info("using PyTorch data loader")
    # data_loader = DataLoader(
    #    dataset,
    #    sampler=sampler,
    #    batch_size=batch_size_per_gpu,
    #    num_workers=num_workers,
    #    pin_memory=True,
    #    drop_last=drop_last,
    #    persistent_workers=persistent_workers,
    #    collate_fn=collate_fn,
    # )
    data_loader = ParallelSampleDataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        collate_fn=collate_fn,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=False,
        shared_memory=False,
    )

    try:
        logger.info(f"# of batches: {len(data_loader):,d}")
    except TypeError:
        logger.info("infinite data loader")
    return data_loader

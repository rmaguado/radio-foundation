# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
from typing import Any, Callable, List, Optional, TypeVar
from omegaconf import DictConfig

import torch
from torch.utils.data import Sampler

from .datasets import MultiDataset, DicomCtDataset, NiftiCtDataset, MedicalImageDataset
from .samplers import EpochSampler, InfiniteSampler, ShardedInfiniteSampler
from .augmentations import DataAugmentationDINO


logger = logging.getLogger("dinov2")


class SamplerType(Enum):
    DISTRIBUTED = 0
    EPOCH = 1
    INFINITE = 2
    SHARDED_INFINITE = 3


def make_train_dataset(
    config: DictConfig,
    use_full_image: bool,
) -> MedicalImageDataset:
    """
    Parse the dataset from the given OmegaConf configuration.

    Args:
        config (DictConfig): The OmegaConf dictionary configuration for the dataset.
        use_full_image (bool): Whether to set the global crop size to the full size.

    Returns:
        MedicalImageDataset: The corresponding dataset object(s).
    """

    dataset_objects = []

    for dataset_config in config.datasets:
        dataset_object = build_dataset_from_cfg(config, use_full_image, dataset_config)
        dataset_objects.append(dataset_object)

    return (
        dataset_objects[0]
        if len(dataset_objects) == 1
        else MultiDataset(dataset_objects)
    )


def build_dataset_from_cfg(config, use_full_image, dataset_config):
    def get_ct_kwargs(dataset_config):
        return {
            "channels": dataset_config.channels,
            "lower_window": dataset_config.pixel_range.lower,
            "upper_window": dataset_config.pixel_range.upper,
        }

    dataset_type = dataset_config.type
    dataset_storage = dataset_config.storage
    transform = DataAugmentationDINO(config, dataset_config, use_full_image)

    dataset_kwargs = {
        "dataset_name": dataset_config.name,
        "index_path": dataset_config.index_path,
        "root_path": dataset_config.data.root_path,
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
    return dataset_object


def _make_sampler(
    *,
    dataset,
    sampler_type: Optional[SamplerType] = None,
    shuffle: bool = False,
    seed: int = 0,
    size: int = -1,
) -> Optional[Sampler]:
    """
    Creates a sampler with the specified parameters.
    A sampler is a strategy for sampling data from a dataset.
    Supported sampler types includes:
        - EPOCH, INFINITE, SHARDED_INFINITE, DISTRIBUTED.

    Args:
        dataset: The dataset to create the sampler for.
        type (Optional[SamplerType]): The type of sampler to create. Defaults to None.
        shuffle (bool): Whether to shuffle the samples. Defaults to False.
        seed (int): The random seed for shuffling. Defaults to 0.
        size (int): The size of the sampler. Defaults to -1.

    Returns:
        Optional[Sampler]: The created sampler or None if no sampler is created.
    """
    sample_count = len(dataset)

    if sampler_type == SamplerType.INFINITE:
        logger.info("sampler: infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        return InfiniteSampler(sample_count=sample_count, shuffle=shuffle, seed=seed)
    elif sampler_type == SamplerType.SHARDED_INFINITE:
        logger.info("sampler: sharded infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        return ShardedInfiniteSampler(
            sample_count=sample_count, shuffle=shuffle, seed=seed
        )
    elif sampler_type == SamplerType.EPOCH:
        logger.info("sampler: epoch")
        size = size if size > 0 else sample_count
        logger.info(f"# of samples / epoch: {size:,d}")
        return EpochSampler(
            size=size,
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
        )
    elif sampler_type == SamplerType.DISTRIBUTED:
        logger.info("sampler: distributed")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        return torch.utils.data.DistributedSampler(
            dataset=dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )

    logger.info("sampler: none")
    return None


T = TypeVar("T")


def make_data_loader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 0,
    sampler_type: Optional[SamplerType] = SamplerType.INFINITE,
    sampler_size: int = -1,
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
        shuffle: Whether to shuffle samples.
        seed: The random seed to use.
        sampler_type: Which sampler to use: EPOCH, INFINITE, SHARDED_INFINITE, DISTRIBUTED or None.
        sampler_size: The number of images per epoch (when applicable) or -1 for the entire dataset.
        drop_last: Whether the last non-full batch of data should be dropped.
        persistent_workers: maintain the workers Dataset instances alive after a dataset has been consumed once.
        collate_fn: Function that performs batch collation
    """

    sampler = _make_sampler(
        dataset=dataset,
        sampler_type=sampler_type,
        shuffle=shuffle,
        seed=seed,
        size=sampler_size,
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
    except TypeError:  # data loader has no length
        logger.info("infinite data loader")
    return data_loader

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import itertools
from typing import List, Any, Optional
import logging

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

import dinov2.distributed as dist


logger = logging.getLogger("dinov2")


def _get_torch_dtype(size: int) -> Any:
    return torch.int32 if size <= 2**31 else torch.int64


def _generate_randperm_indices(*, size: int, generator: torch.Generator):
    """Generate the indices of a random permutation."""
    dtype = _get_torch_dtype(size)
    # This is actually matching PyTorch's CPU implementation, see: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorFactories.cpp#L900-L921
    perm = torch.arange(size, dtype=dtype)
    for i in range(size):
        j = torch.randint(i, size, size=(1,), generator=generator).item()

        # Always swap even if no-op
        value = perm[j].item()
        perm[j] = perm[i].item()
        perm[i] = value
        yield value


def _weighted_generate_randperm_indices(
    *, sizes: List[int], weights: List[float], generator: torch.Generator
):
    """Generate indices of a weighted random permutation across multiple datasets."""
    cumulative_sizes = np.cumsum([0] + sizes)
    normalized_weights = [w / sum(weights) for w in weights]

    min_size = min(sizes)

    while True:
        indices = []
        for dataset_idx, (size, weight) in enumerate(zip(sizes, normalized_weights)):
            dataset_indices = (
                torch.randperm(size, generator=generator)
                + cumulative_sizes[dataset_idx]
            )

            sample_size = int(weight * min_size)
            indices.extend(dataset_indices[:sample_size].tolist())

        indices = torch.tensor(indices)
        perm = torch.randperm(len(indices), generator=generator)
        for idx in perm:
            yield indices[idx].item()


def _new_shuffle_tensor_slice(
    *, tensor: torch.Tensor, start: int = 0, step: int = 1, generator: torch.Generator
) -> np.ndarray:
    stop = len(tensor)
    count = stop // step
    drop_count = stop - step * count
    if drop_count:
        logger.warning(f"# of dropped samples: {drop_count}")
    indices = torch.randperm(count, dtype=torch.int64, generator=generator)
    return tensor[start::step][indices].numpy()


def _make_seed(seed: int, start: int, iter_count: int) -> int:
    # NOTE: Tried a few variants (including iter_count << 32), this one worked best.
    return seed + start + (iter_count << 24)


def check_weighted_sampler_params(
    datasets: List[str], sizes: List[int], weights: List[float]
):
    if len(sizes) != len(weights):
        raise ValueError(
            f"Dataset sizes and weights must have the same length, got {len(sizes)} and {len(weights)}"
        )

    if not all(isinstance(weight, float) for weight in weights):
        raise ValueError("Weights must be floats")

    total_size = sum(sizes)
    size_threshold = total_size * 0.01
    for i, dataset_name in enumerate(datasets):
        dataset_size = sizes[i]
        if dataset_size < size_threshold:
            logger.warning(
                f"Dataset {dataset_name} with size {dataset_size} is less than 1% of the total data size {total_size}. Weighted sampling may be unstable."
            )


class InfiniteSampler(Sampler):
    def __init__(
        self,
        *,
        sample_count: int,
        seed: int = 0,
    ):
        self._sample_count = sample_count
        self._seed = seed
        self._start = dist.get_rank()
        self._step = dist.get_world_size()

    def __iter__(self):
        iterator = self._iterator()

        yield from itertools.islice(iterator, 0, None)

    def _iterator(self):
        # Instantiate a generator here (rather than in the ctor) to keep the class
        # picklable (requirement of mp.spawn)
        generator = torch.Generator()
        generator.manual_seed(self._seed)

        while True:
            iterable = _generate_randperm_indices(
                size=self._sample_count, generator=generator
            )
            yield from itertools.islice(iterable, self._start, None, self._step)


class WeightedInfiniteSampler(Sampler):
    def __init__(
        self,
        *,
        dataset_names: List[str],
        sizes: List[int],
        weights: List[float],
        seed: int = 0,
    ):
        self._dataset_sizes = sizes
        self._weights = weights
        self._seed = seed
        self._start = dist.get_rank()
        self._step = dist.get_world_size()

        check_weighted_sampler_params(dataset_names, sizes, weights)

    def __iter__(self):
        iterator = self._iterator()
        yield from itertools.islice(iterator, 0, None)

    def _iterator(self):
        generator = torch.Generator()
        generator.manual_seed(self._seed)

        while True:
            iterable = _weighted_generate_randperm_indices(
                sizes=self._dataset_sizes, weights=self._weights, generator=generator
            )
            yield from itertools.islice(iterable, self._start, None, self._step)


class ShardedInfiniteSampler(Sampler):
    def __init__(
        self,
        *,
        sample_count: int,
        seed: int = 0,
    ):
        self._sample_count = sample_count
        self._seed = seed
        self._start = dist.get_rank()
        self._step = dist.get_world_size()
        self._iter_count = 0
        self._shuffle_tensor_slice_fn = _new_shuffle_tensor_slice

    def __iter__(self):
        iterator = self._iterator()

        yield from itertools.islice(iterator, 0, None)

    def _iterator(self):
        # Instantiate a generator here (rather than in the ctor) to be keep the class
        # picklable (requirement of mp.spawn)
        generator = torch.Generator()

        # Always shuffle everything first
        generator.manual_seed(self._seed)
        dtype = _get_torch_dtype(self._sample_count)
        perm = torch.randperm(self._sample_count, dtype=dtype, generator=generator)

        while True:
            # Re-seed on each iteration to allow skipping whole permutations
            seed = _make_seed(self._seed, self._start, self._iter_count)
            generator.manual_seed(seed)

            iterable = self._shuffle_tensor_slice_fn(
                tensor=perm, start=self._start, step=self._step, generator=generator
            )
            yield from iterable
            self._iter_count += 1


class WeightedShardedInfiniteSampler(Sampler):
    def __init__(
        self,
        *,
        dataset_names: List[int],
        sizes: List[int],
        weights: List[float],
        seed: int = 0,
    ):
        self._dataset_sizes = sizes
        self._weights = weights
        self._seed = seed
        self._start = dist.get_rank()
        self._step = dist.get_world_size()
        self._iter_count = 0
        self._shuffle_tensor_slice_fn = _new_shuffle_tensor_slice

        check_weighted_sampler_params(dataset_names, sizes, weights)

    def __iter__(self):
        iterator = self._iterator()
        yield from itertools.islice(iterator, 0, None)

    def _iterator(self):
        generator = torch.Generator()

        # Normalize weights
        total_weight = sum(self._weights)
        normalized_weights = [weight / total_weight for weight in self._weights]

        while True:
            total_samples = sum(
                size * weight
                for size, weight in zip(self._dataset_sizes, normalized_weights)
            )

            sample_sizes = [
                int(total_samples * weight) for weight in normalized_weights
            ]
            sample_sizes = [
                min(size, s) for size, s in zip(self._dataset_sizes, sample_sizes)
            ]

            all_indices = []
            for dataset_idx, size in enumerate(self._dataset_sizes):
                if sample_sizes[dataset_idx] > 0:
                    perm = torch.randperm(size, generator=generator)
                    indices = perm[: sample_sizes[dataset_idx]] + sum(
                        self._dataset_sizes[:dataset_idx]
                    )
                    all_indices.append(indices)

            assert all_indices, "No samples to shuffle"

            all_indices = torch.cat(all_indices)
            seed = _make_seed(self._seed, self._start, self._iter_count)
            generator.manual_seed(seed)
            shuffled_indices = all_indices[
                torch.randperm(len(all_indices), generator=generator)
            ]

            iterable = self._shuffle_tensor_slice_fn(
                tensor=shuffled_indices,
                start=self._start,
                step=self._step,
                generator=generator,
            )
            yield from iterable
            self._iter_count += 1

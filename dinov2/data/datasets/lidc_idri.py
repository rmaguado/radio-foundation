# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import csv
from enum import Enum
import logging
import os
from typing import Callable, List, Optional, Tuple, Union, Any

from PIL import Image
import numpy as np
import json
import h5py

from .extended import ExtendedVisionDataset


logger = logging.getLogger("dinov2")
_Target = int

class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        # slices
        split_lengths = {
            _Split.TRAIN: 195_178,
            _Split.VAL: 24_454,
            _Split.TEST: 24_326,
        }
        return split_lengths[self]

    def get_image_relpath(self, image_id: str) -> str:
        return os.path.join(self.value, image_id)


class LidcIdri(ExtendedVisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "LidcIdri.Split",
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        enable_targets: bool = False
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split
        self.enable_targets = enable_targets

        self._entries = None

    @property
    def split(self) -> "LidcIdri.Split":
        return self._split

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        extra_full_path = self._get_extra_full_path(extra_path)
        os.makedirs(self._extra_root, exist_ok=True)
        np.save(extra_full_path, extra_array)

    @property
    def _entries_path(self) -> str:
        return f"entries-{self._split.value.upper()}.npy"

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    def get_image_data(self, index: int) -> np.ndarray:
        entries = self._get_entries()
        series_id = entries[index]["series_id"]
        slice_index = entries[index]["slice_index"]

        image_full_path = os.path.join(self.root, self.split.value, series_id, 'image.npy')
        
        image_mmap = np.load(image_full_path, mmap_mode="r")
        loaded_image = image_mmap[slice_index].astype(np.float32)

        return Image.fromarray(loaded_image, 'F')

    def get_target(self, index: int) -> Optional[Target]:
        entries = self._get_entries()
        
        series_id = entries[index]["series_id"]
        slice_index = entries[index]["slice_index"]

        mask_full_path = os.path.join(self.root, self.split.value, series_id, 'mask.h5')
        
        with h5py.File(mask_full_path, 'r') as f:
            data = f["data"]
            loaded_mask = data[slice_index]

        return Image.fromarray(loaded_mask)

    def get_targets(self) -> Optional[np.ndarray]:
        entries = self._get_entries()
        return [self.get_target(i) for i in range(len(entries))]

    def __len__(self) -> int:
        entries = self._get_entries()
        assert len(entries) == self.split.length
        return len(entries)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
            
        if self.enable_targets:
            target = self.get_target(index)
        else:
            target = None

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def _dump_entries(self) -> None:
        split = self.split
        
        base_dir = os.path.join(self.root, split.value)

        series_dirs = [
            d for d in os.listdir(base_dir) \
            if os.path.isdir(os.path.join(self.root, split.value, d))
        ]

        dtype = np.dtype(
            [
                ("series_id", "U256"),
                ("slice_index", "uint16"),
            ]
        )
        entries_array = np.empty(self.split.length, dtype=dtype)
        
        abs_slice_index = 0
        for series_id in series_dirs:
            with open(os.path.join(base_dir, series_id, "metadata.json"), 'r') as f:
                metadata = json.load(f)
            num_slices = metadata["shape"][0]
            for slice_index in range(num_slices):
                entries_array[abs_slice_index] = (
                    series_id,
                    slice_index
                )
                abs_slice_index += 1

        logger.info(f'saving entries to "{self._entries_path}"')
        self._save_extra(entries_array, self._entries_path)

    def dump_extra(self) -> None:
        self._dump_entries()

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Callable, Optional, Tuple, Any

import torch
import numpy as np
import h5py

from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov2")


class CtDataset(ExtendedVisionDataset):

    def __init__(
        self,
        *,
        split: str,
        root: str,
        extra: str,
        num_slices: int,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split
        self._num_slices = num_slices
        
        self._entries = None

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            extra_full_path = os.path.join(self._extra_root, f"entries-{self._split.upper()}.npy")
            self._entries = np.load(extra_full_path, mmap_mode="r")
        assert self._entries is not None
        return self._entries

    def get_image_data(self, index: int) -> np.ndarray:
        entries = self._get_entries()
        series_id = entries[index]["series_id"]
        slice_index = entries[index]["slice_index"]
        
        image_full_path = os.path.join(self.root, series_id, 'image.h5')
        
        with h5py.File(image_full_path, 'r') as f:
            data = f["data"]
            image = data[slice_index].astype(np.float32)
            
        return torch.from_numpy(image).unsqueeze(0)

    def get_target(self, index: int) -> Optional[Any]:
        return None

    def get_targets(self) -> Optional[np.ndarray]:
        entries = self._get_entries()
        return [self.get_target(i) for i in range(len(entries))]

    def __len__(self) -> int:
        entries = self._get_entries()
        return len(entries)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
            
        target = self.get_target(index)
            
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            
        return image, target

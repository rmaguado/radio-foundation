# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Callable, Optional, Any
import json

import numpy as np
import torch
import h5py
import torch.distributed as dist

from .base import BaseDataset

logger = logging.getLogger("dinov2")


class VolumeDataset(BaseDataset):

    def __init__(
        self,
        root_path: str,
        output_path: str,
        dataset_name: str,
        split: str,
        channels: int,
        transform: Optional[Callable] = lambda _: _,
        target_transform: Optional[Callable] = lambda _: _,
    ) -> None:
        super().__init__()

        self.root_path = root_path
        self.output_path = output_path
        self.dataset_name = dataset_name
        self.split = split
        self.channels = channels

        self.transform = transform
        self.target_transform = target_transform

        self.entries = self.get_entries()
        self.data_path = os.path.join(root_path, dataset_name, "data")

    def create_entries(self, entries_dataset_path):
        split_path = os.path.join(
            self.root_path, self.dataset_name, f"extra/{self.split}.json"
        )
        with open(split_path, "r", encoding="utf-8") as f:
            entries_data = json.load(f)

        total_slices = sum(
            [x["slices"] // self.channels for x in entries_data.values()]
        )

        dtype = np.dtype([("series_id", "U256"), ("slice_index", "uint16")])
        entries_array = np.empty(total_slices, dtype=dtype)

        counter = 0
        for series_id, series_data in entries_data.items():
            series_slices = series_data["slices"] // self.channels
            for series_slice_index in range(
                0, series_slices * self.channels, self.channels
            ):
                entries_array[counter] = (series_id, series_slice_index)
                counter += 1

        np.save(entries_dataset_path, entries_array)
        return np.load(entries_dataset_path, mmap_mode="r")

    def get_target(self, index: int) -> Optional[Any]:
        return None

    def get_image_data(self, index: int) -> torch.tensor:
        series_id = self.entries[index]["series_id"]
        slice_index = self.entries[index]["slice_index"]

        image_full_path = os.path.join(self.data_path, series_id, "image.h5")

        with h5py.File(image_full_path, "r") as f:
            data = f["data"]
            image_array = np.array(
                data[slice_index : slice_index + self.channels], dtype=np.float32
            )

        return torch.from_numpy(image_array)

    def __len__(self) -> int:
        return len(self.entries)

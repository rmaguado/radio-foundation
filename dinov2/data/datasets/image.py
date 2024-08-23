# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import json
from typing import Callable, Optional, Any

import numpy as np
import torch
import h5py

from .base import BaseDataset

logger = logging.getLogger("dinov2")


class ImageDataset(BaseDataset):

    def __init__(
        self,
        root_path: str,
        output_path: str,
        dataset_name: str,
        split: str,
        transform: Optional[Callable] = lambda _: _,
        target_transform: Optional[Callable] = lambda _: _,
    ) -> None:
        super().__init__()

        self.root_path = root_path
        self.output_path = output_path
        self.dataset_name = dataset_name
        self.split = split

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

        total_images = len(entries_data)

        dtype = np.dtype([("image_id", "U256")])
        entries_array = np.empty(total_images, dtype=dtype)

        for i, (series_id, _) in enumerate(entries_data.items()):
            entries_array[i] = series_id

        np.save(entries_dataset_path, entries_array)
        return np.load(entries_dataset_path, mmap_mode="r")

    def get_target(self, index: int) -> Optional[Any]:
        return None

    def get_image_data(self, index: int) -> torch.tensor:
        image_id = self.entries[index]["image_id"]

        image_full_path = os.path.join(self.data_path, image_id, "image.h5")

        with h5py.File(image_full_path, "r") as f:
            data = f["data"]
            image = np.array(data[:], dtype=np.float32)

        return torch.from_numpy(image).unsqueeze(0)

    def __len__(self) -> int:
        return len(self.entries)

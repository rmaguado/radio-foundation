# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Callable, Optional, Tuple, Any

import numpy as np
import torch
import json

logger = logging.getLogger("dinov2")


class ImageDataset:
    def __init__(
        self,
        root_path: str,
        output_path: str,
        dataset_name: str,
        split: str,
        transform: Optional[Callable] = lambda _:_,
        target_transform: Optional[Callable] = lambda _:_
    ) -> None:
        self.root_path = root_path
        self.output_path = output_path
        self.dataset_name = dataset_name
        self.split = split
        
        self.transform = transform
        self.target_transform = target_transform
        
        self.entries = self.get_entries()
        self.data_path = os.path.join(root_path, dataset_name, "data")
        
    def get_entries(self) -> np.ndarray:
        entries_path = os.path.join(self.output_path, "entries")
        os.makedirs(entries_path, exist_ok=True)
        
        entries_dataset_path = os.path.join(entries_path, f"{self.dataset_name}.npy")
        if os.path.exists(entries_dataset_path):
            return np.load(entries_dataset_path, mmap_mode="r")
        return self.create_entries(entries_dataset_path)
    
    def create_entries(self, entries_dataset_path):
        split_path = os.path.join(self.root_path, self.dataset_name, f"extra/{self.split}.json")
        with open(split_path, "r") as f:
            entries_data = json.load(f)
        
        total_images = len(entries_data)
        
        dtype = np.dtype([
            ("image_id", "U256")
        ])
        entries_array = np.empty(total_images, dtype=dtype)
        
        for i, (series_id, series_data) in enumerate(entries_data.items()):
            entries_array[i] = (series_id)
        
        np.save(entries_dataset_path, entries_array)
        return np.load(entries_dataset_path, mmap_mode="r")
        
    def get_image_data(self, index: int) -> torch.tensor:
        image_id = self.entries[index]["image_id"]
        
        image_full_path = os.path.join(self.data_path, series_id, 'image.h5')
        
        with h5py.File(image_full_path, 'r') as f:
            data = f["data"]
            image = data[:].astype(np.float32)
            
        return torch.from_numpy(image).unsqueeze(0)

    def __len__(self) -> int:
        return len(self.entries)

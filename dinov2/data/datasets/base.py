# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Any, Tuple, Optional
import torch
import numpy as np
import os
import sqlite3

import torch.distributed as dist


class BaseDataset:
    def __init__(self):
        self.transform = lambda _: _
        self.target_transform = lambda _: _
        self.dataset_name = "DatasetNotGiven"
        self.index_path = "IndexPathNotGiven"
        self.root_path = "RootPathNotGiven"
        self.output_path = "OutputPathNotGiven"
        self.entries_path = "path/to/entries"
        
        self.conn = None
        self.cursor = None

    def get_entries(self) -> np.ndarray:
        os.makedirs(self.entries_path, exist_ok=True)

        entries_dataset_path = os.path.join(
            self.entries_path, f"{self.dataset_name}.npy"
        )

        if dist.is_initialized() and dist.get_rank() == 0:
            if not os.path.exists(entries_dataset_path):
                self.create_entries()

        if dist.is_initialized():
            dist.barrier()

        return np.load(entries_dataset_path, mmap_mode="r")

    def open_db(self):
        self.conn = sqlite3.connect(self.index_path, uri=True)
        self.cursor = self.conn.cursor()

    def __del__(self):
        if isinstance(self.conn, sqlite3.Connection):
            self.conn.close()

    def create_entries(self) -> np.ndarray:
        raise NotImplementedError

    def get_image_data(self, index: int) -> torch.tensor:
        raise NotImplementedError

    def get_target(self, index: int) -> Optional[Any]:
        raise NotImplementedError

    def get_targets(self) -> Optional[any]:
        return [self.get_target(i) for i in len(self)]

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, image, target):
        return self.transform(image), self.target_transform(target)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e

        target = self.get_target(index)

        return self.apply_transforms(image, target)

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Any, Tuple, Optional
import torch
import numpy as np
import os
import sqlite3
from datetime import datetime

import torch.distributed as dist
from dinov2.distributed import is_main_process, is_enabled


class BaseDataset:
    def __init__(self):
        """
        Initializes the base dataset class for dicom storage of image or volumetric datasets.

        Attributes:
        - dataset_name (str): The name of the dataset.
        - index_path (str): The path to the index file.
        - root_path (str): The root path of the dataset.
        - entries_path (str): The path to the entries.
        - transform (function): A function to transform the data.
        - target_transform (function): A function to transform the target.
        - conn: The database connection.
        - cursor: The database cursor.
        - entries: The dataset entries.
        """
        self.dataset_name = "DatasetNotGiven"
        self.index_path = "IndexPathNotGiven"
        self.root_path = "RootPathNotGiven"
        self.entries_path = "path/to/entries"
        self.channels = 1

        self.transform = lambda _: _
        self.target_transform = lambda _: _

        self.conn = None
        self.cursor = None

        self.entries = None

    def get_entries_dir(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.entries_path, f"{timestamp}.npy")

    def get_entries(self) -> np.ndarray:
        """
        Get a numpy memmap object pointing to the sqlite database rows of dicom paths.
        If using distributed, only create new memmap on the main process.

        Returns:
            np.ndarray: The entries dataset.
        """
        os.makedirs(self.entries_path, exist_ok=True)

        entries_dir = self.get_entries_dir()

        if is_main_process():
            self.create_entries()

        if is_enabled():
            dist.barrier()

        return np.load(entries_dir, mmap_mode="r")

    def open_db(self) -> None:
        """
        Opens a connection to the SQLite database.

        Returns:
            None
        """
        self.conn = sqlite3.connect(self.index_path, uri=True)
        self.cursor = self.conn.cursor()

    def __del__(self):
        """
        Closes the SQLite connection if it is an instance of `sqlite3.Connection`.

        This method is automatically called when the object is about to be destroyed.
        """
        if isinstance(self.conn, sqlite3.Connection):
            self.conn.close()

    def create_entries(self) -> np.ndarray:
        """
        Generates a numpy memmap object pointing to the sqlite database rows of dicom paths.

        Returns:
            np.ndarray: The entries dataset (memmap).
        """
        raise NotImplementedError(
            "create_entries is an abstract method and needs to be implemented."
        )

    def get_image_data(self, index: int) -> torch.Tensor:
        """
        Retrieves the image data at the specified index.

        Parameters:
            index (int): The index of the image data to retrieve.

        Returns:
            torch.Tensor: The image data as a tensor.
        """
        raise NotImplementedError(
            "get_image_data is an abstract method and needs to be implemented."
        )

    def get_target(self, index: int) -> Optional[Any]:
        """
        Returns the target value for the given index.

        Parameters:
            index (int): The index of the target value to retrieve.

        Returns:
            Optional[Any]: The target value for the given index, or None if not available.
        """
        raise NotImplementedError(
            "get_target is an abstract method and needs to be implemented."
        )

    def get_targets(self) -> Optional[any]:
        return [self.get_target(i) for i in len(self)]

    def __len__(self) -> int:
        raise NotImplementedError(
            "The __len__ method must be implemented in the subclass."
        )

    def apply_transforms(self, image, target):
        return self.transform(image), self.target_transform(target)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        """
        Get the item at the specified index from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, Any]: A tuple containing the image data and the target for the specified index.
        """
        try:
            image = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e

        target = self.get_target(index)

        return self.apply_transforms(image, target)

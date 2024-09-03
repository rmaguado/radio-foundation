# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import Callable, Optional, Any

import os
import numpy as np
import torch
import sqlite3
import pydicom

from .base import BaseDataset

logger = logging.getLogger("dinov2")


class CtDataset(BaseDataset):

    def __init__(
        self,
        dataset_name: str,
        index_path: str,
        root_path: str,
        output_path: str,
        options: Optional[dict] = None,
        transform: Optional[Callable] = lambda _: _,
        target_transform: Optional[Callable] = lambda _: _,
    ) -> None:
        super().__init__()

        self.dataset_name = dataset_name
        self.index_path = index_path
        self.root_path = root_path
        self.output_path = output_path
        self.open_db()

        self.channels = options.get("channels", 1)
        self.lower_window = options.get("lower_window", -1000)
        self.upper_window = options.get("upper_window", 2000)

        self.transform = transform
        self.target_transform = target_transform
        
        self.entries_path = os.path.join(self.output_path, "entries")
        self.entries = self.get_entries()

    def create_entries(self) -> np.ndarray:
        slice_stack_num = self.channels

        self.cursor.execute(f"SELECT dataset, series_id, slice_index FROM global")
        global_rows = self.cursor.fetchall()

        entries = []

        for dataset, series_id, slice_index in global_rows:
            self.cursor.execute(
                f"SELECT num_slices FROM '{dataset}' WHERE series_id = '{series_id}'"
            )
            num_slices = self.cursor.fetchone()[0]
            if slice_index > num_slices - slice_stack_num:
                continue
            stack_slice_indexes = [slice_index + x for x in range(slice_stack_num)]

            self.cursor.execute(
                """
                SELECT rowid, slice_index 
                FROM global 
                WHERE series_id = ? 
                AND dataset = ? 
                AND slice_index IN ({})
                """.format(
                    ",".join("?" * len(stack_slice_indexes))
                ),
                (series_id, dataset, *stack_slice_indexes),
            )
            stack_rows = self.cursor.fetchall()
            if len(stack_rows) != slice_stack_num:
                raise ValueError(f"Not right amount of slices in stack: {stack_rows}.")

            stack_rows.sort(key=lambda x: x[1])
            entries.append([x[0] for x in stack_rows])

        entries_array = np.array(entries, dtype=np.uint32)
        
        entries_dataset_path = os.path.join(
            self.entries_path, f"{self.dataset_name}.npy"
        )
        np.save(entries_dataset_path, entries_array)
        return np.load(entries_dataset_path, mmap_mode="r")

    def process_ct(self, dcm) -> torch.tensor:
        if dcm.RescaleType == "HU":
            array_data = dcm.pixel_array
        else:
            array_data = dcm.RescaleSlope * dcm.pixel_array + dcm.RescaleIntercept
        array_data = np.clip(array_data, self.lower_window, self.upper_window)
        array_data = (array_data - self.lower_window) / (
            self.upper_window - self.lower_window
        )

        return torch.tensor(array_data, dtype=torch.float32)

    def get_target(self, index: int) -> Optional[Any]:
        return None

    def get_image_data(self, index: int) -> torch.tensor:
        stack_rowids = [int(x) for x in self.entries[index]]
        self.cursor.execute(
            """
            SELECT rowid, slice_index, dataset, dicom_path
            FROM global 
            WHERE rowid IN ({})
            """.format(
                ",".join("?" * self.channels)
            ),
            stack_rowids,
        )
        stack_rows = self.cursor.fetchall()
        stack_rows.sort(key=lambda x: x[1])
        
        

        stack_data = []
        for _, _, dataset, rel_dicom_path in stack_rows:
            abs_dicom_path = os.path.join(self.root_path, dataset, rel_dicom_path)
            dcm = pydicom.dcmread(abs_dicom_path)
            stack_data.append(self.process_ct(dcm))

        return torch.stack(stack_data)

    def __len__(self) -> int:
        return len(self.entries)

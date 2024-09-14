# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import Callable, Optional, Any

import os
import numpy as np
import torch
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
        channels: int,
        lower_window: int,
        upper_window: int,
        transform: Optional[Callable] = lambda _: _,
        target_transform: Optional[Callable] = lambda _: _,
    ) -> None:
        """
        Initializes a CTScan object.

        Args:
            dataset_name (str): The name of the dataset.
            index_path (str): The path to the index file.
            root_path (str): The root path of the dataset.
            output_path (str): The output path for the dataset.
            channels (int): The number of channels to use.
            lower_window (int): The lower window value.
            upper_window (int): The upper window value.
            transform (Optional[Callable], optional): A function to apply to the data. Defaults to lambda _: _.
            target_transform (Optional[Callable], optional): A function to apply to the target. Defaults to lambda _: _.
        """
        super().__init__()

        self.dataset_name = dataset_name
        self.index_path = index_path
        self.root_path = root_path
        self.output_path = output_path
        self.entries_path = os.path.join(os.path.dirname(self.index_path), "entries")
        self.open_db()

        self.transform = transform
        self.target_transform = target_transform

        self.channels = channels
        self.lower_window = lower_window
        self.upper_window = upper_window

        self.entries = self.get_entries()

    def create_entries(self) -> np.ndarray:
        """
        Generates a numpy memmap object pointing to the sqlite database rows of dicom paths.
        For collecting various slices of a CT scan (multi-channel) each memmap row contains the ordered rowids of the slices.

        Returns:
            np.ndarray: The entries dataset (memmap).
        """
        logger.info(f"Creating entries for {self.dataset_name}.")

        slice_stack_num = self.channels
        entries_dataset_path = os.path.join(
            self.entries_path, f"{slice_stack_num}_channels.npy"
        )

        dataset_names = self.cursor.execute(
            "SELECT dataset FROM sqlite_master WHERE type='table' AND name != 'global'"
        ).fetchall()

        series_ids = []
        for dataset_name in dataset_names:
            dataset_name = dataset_name[0]
            series_ids += self.cursor.execute(
                f"SELECT dataset, series_id, num_slices FROM '{dataset_name}'"
            ).fetchall()

        logger.info(f"Total number of series: {len(series_ids)}.")

        entries = []
        for dataset_name, series_id, num_slices in series_ids:

            if num_slices < slice_stack_num:
                continue

            self.cursor.execute(
                """
                SELECT rowid, slice_index 
                FROM global 
                WHERE series_id = ? 
                AND dataset = ? 
                ORDER BY slice_index
                """,
                (series_id, dataset_name),
            )
            slice_indexes = self.cursor.fetchall()

            slice_indexes.sort(key=lambda x: x[1])

            stack_rows = [
                slice_indexes[i : i + slice_stack_num, 0]
                for i in range(len(slice_indexes) - slice_stack_num + 1)
            ]

            entries += stack_rows

        entries_array = np.array(entries, dtype=np.uint32)

        entries_dataset_path = os.path.join(
            self.entries_path, f"{self.channels}_channels.npy"
        )
        logger.info(f"Saving entries to {entries_dataset_path}.")
        np.save(entries_dataset_path, entries_array)
        return np.load(entries_dataset_path, mmap_mode="r")

    def process_ct(self, dcm: pydicom.dataset.FileDataset) -> torch.tensor:
        """
        Process a CT scan by applying rescaling and windowing.

        Args:
            dcm (pydicom.dataset.FileDataset): The DICOM object representing the CT scan.

        Returns:
            torch.tensor: The processed CT scan as a tensor.

        """
        slope = getattr(dcm, "RescaleSlope", 1)
        intercept = getattr(dcm, "RescaleIntercept", 0)

        array_data = dcm.pixel_array * slope + intercept

        array_data = np.clip(array_data, self.lower_window, self.upper_window)
        array_data = (array_data - self.lower_window) / (
            self.upper_window - self.lower_window
        )

        return torch.tensor(array_data, dtype=torch.float32)

    def get_target(self, index: int) -> Optional[Any]:
        return None

    def get_image_data(self, index: int) -> torch.tensor:
        """
        Retrieves the image data for a given index.

        Args:
            index (int): The index of the image data to retrieve.

        Returns:
            torch.tensor: The image data as a torch tensor.
        """
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

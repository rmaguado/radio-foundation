import logging
import os
import numpy as np
import torch
import pydicom
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Any

from dinov2.data.datasets.dicoms import DicomCTVolumesFull


logger = logging.getLogger("dinov2")


class FullVolumeEval(DicomCTVolumesFull):
    def __init__(
        self,
        root_path: str,
        dataset_name: str,
        channels: int = 1,
        transform=None,
        max_workers=4,
    ):
        super().__init__(
            dataset_name=dataset_name,
            root_path=root_path,
            channels=channels,
            transform=transform,
        )
        self.max_workers = max_workers

    def create_entries(self) -> np.ndarray:
        """
        Generates a numpy memmap object pointing to the sqlite database rows of dicom paths.

        Returns:
            np.ndarray: The entries dataset (memmap).
        """
        logger.info(f"Creating entries for {self.dataset_name}.")

        dataset_names = self.cursor.execute(f"SELECT dataset FROM datasets").fetchall()

        entries_dtype = [
            ("dataset", "U256"),
            ("rowid", np.uint32),
            ("map_id", "U256"),
        ]
        entries = []
        for dataset_name in dataset_names:
            dataset_name = dataset_name[0]
            dataset_series = self.cursor.execute(
                f"SELECT rowid, map_id FROM '{dataset_name}'"
            ).fetchall()

            for rowid, map_id in dataset_series:
                entries.append((dataset_name, rowid, map_id))

        logger.info(f"Total number of scans: {len(entries)}.")

        entries_array = np.array(entries, dtype=entries_dtype)

        entries_dir = self.get_entries_dir()
        logger.info(f"Saving entries to {entries_dir}.")
        np.save(entries_dir, entries_array)
        return np.load(entries_dir, mmap_mode="r")

    def get_image_data(self, index: int) -> Tuple[torch.Tensor, str]:
        dataset_name, rowid, map_id = self.entries[index]

        self.cursor.execute(
            f"SELECT series_id, map_id FROM '{dataset_name}' WHERE rowid = {rowid}"
        )
        series_id, map_id = self.cursor.fetchone()

        self.cursor.execute(
            """
            SELECT slice_index, dataset, dicom_path
            FROM global 
            WHERE series_id = ? 
            AND dataset = ?
            """,
            (series_id, dataset_name),
        )
        stack_rows = self.cursor.fetchall()
        stack_rows.sort(key=lambda x: x[0])

        try:
            stack_data = self.create_stack_data(stack_rows)
        except Exception as e:
            logger.exception(f"Error processing stack. Seriesid: {series_id} \n{e}")
            stack_data = torch.zeros((10, 512, 512))

        return stack_data, map_id

    @staticmethod
    def load_dicom(row, root_path):
        _, dataset, rel_dicom_path = row
        abs_dicom_path = os.path.join(root_path, dataset, rel_dicom_path)
        dcm = pydicom.dcmread(abs_dicom_path)
        return dcm

    def create_stack_data(self, stack_rows):
        load_dicom_partial = partial(self.load_dicom, root_path=self.root_path)

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            dicom_files = list(executor.map(load_dicom_partial, stack_rows))

        stack_data = [self.process_ct(dcm) for dcm in dicom_files]

        return torch.stack(stack_data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        try:
            image, map_id = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e

        transformed_image = self.transform(image)

        return transformed_image, map_id

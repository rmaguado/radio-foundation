import os
import torch
from typing import Tuple, Any
import numpy as np
import logging

from dinov2.data.datasets.dicoms import DicomCTVolumesFull

logger = logging.getLogger("dinov2")


def rle_decode(run_lengths, shape):
    pos = np.cumsum(run_lengths)
    decoded = np.zeros(np.prod(shape), dtype=np.int32)
    current_value = 1
    for i in range(len(pos)):
        decoded[pos[i] : pos[i + 1] if i + 1 < len(pos) else None] = current_value
        current_value = 1 - current_value

    return decoded.reshape(shape)


class LidcIdriNodules(DicomCTVolumesFull):
    def __init__(
        self,
        mask_path: str,
        root_path: str,
        transform,
    ):
        super().__init__(
            dataset_name="LIDC-IDRI",
            root_path=root_path,
            transform=transform,
        )

        self.root_mask_path = mask_path

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
            ("scan_id", np.uint32),
        ]
        entries = []
        for dataset_name in dataset_names:
            dataset_name = dataset_name[0]
            dataset_series = self.cursor.execute(
                f"SELECT rowid, scan_id FROM '{dataset_name}'"
            ).fetchall()
            entries += [
                (dataset_name, rowid, scan_id) for rowid, scan_id in dataset_series
            ]
        logger.info(f"Total number of scans: {len(entries)}.")

        entries_array = np.array(entries, dtype=entries_dtype)

        entries_dir = self.get_entries_dir()
        logger.info(f"Saving entries to {entries_dir}.")
        np.save(entries_dir, entries_array)
        return np.load(entries_dir, mmap_mode="r")

    def get_image_data(self, index: int) -> torch.Tensor:
        dataset_name, rowid, scan_id = self.entries[index]
        self.cursor.execute(
            f"SELECT series_id, scan_id FROM '{dataset_name}' WHERE rowid = ?", (rowid,)
        )
        series_id = self.cursor.fetchone()

        self.cursor.execute(
            """
            SELECT slice_index, dataset, dicom_path
            FROM global 
            WHERE series_id = ? 
            AND dataset = ?
            """,
            (series_id, dataset_name),
        )
        slice_indexes_rowid = self.cursor.fetchall()
        slice_indexes_rowid.sort(key=lambda x: x[0])

        try:
            stack_data = self.create_stack_data(slice_indexes_rowid)
        except Exception as e:
            logger.exception(f"Error processing stack. Seriesid: {series_id} \n{e}")
            stack_data = torch.zeros((10, 512, 512))

        return stack_data, scan_id

    def get_target(self, scanid: str) -> torch.Tensor:

        mask_path = os.path.join(self.root_mask_path, f"{scanid}.npz")

        with np.load(mask_path) as data:
            rle_mask = data["rle_mask"]
            shape = data["shape"]

        mask = rle_decode(rle_mask, shape)

        return mask

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        try:
            image, scanid = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e

        target = self.get_target(scanid)

        return self.apply_transforms(image, target)

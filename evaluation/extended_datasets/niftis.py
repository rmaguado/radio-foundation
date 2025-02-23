import logging
import numpy as np
import torch
import nibabel as nib
from typing import Tuple, Any
import os

from dinov2.data.datasets.dicoms import NiftiCtVolumesFull


logger = logging.getLogger("dinov2")


class NiftiFullVolumeEval(NiftiCtVolumesFull):
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
            ("rowid", np.uint32),
            ("map_id", "U256"),
        ]
        entries = []
        for dataset_name in dataset_names:
            dataset_name = dataset_name[0]
            dataset_series = self.cursor.execute(
                f"SELECT rowid, map_id FROM '{self.dataset_name}'"
            ).fetchall()

            for rowid, map_id in dataset_series:
                entries.append((rowid, map_id))

        logger.info(f"Total number of scans: {len(entries)}.")

        entries_array = np.array(entries, dtype=entries_dtype)

        entries_dir = self.get_entries_dir()
        logger.info(f"Saving entries to {entries_dir}.")
        np.save(entries_dir, entries_array)
        return np.load(entries_dir, mmap_mode="r")

    def get_image_data(self, index: int) -> torch.Tensor:
        rowid, map_id = self.entries[index]
        self.cursor.execute(
            """
            SELECT dataset, axial_dim, nifti_path FROM global WHERE rowid = ?
            """,
            (int(rowid),),
        )
        dataset, axial_dim, nifti_path = self.cursor.fetchone()

        abs_path_to_nifti = os.path.join(self.root_path, dataset, nifti_path)
        nifti_file = nib.load(abs_path_to_nifti)

        volume_data = nifti_file.get_fdata().astype(np.float32)
        volume_data = np.moveaxis(volume_data, axial_dim, 0)

        return self.process_ct(volume_data), map_id

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        try:
            image, map_id = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e

        transformed_image = self.transform(image)

        return transformed_image, map_id

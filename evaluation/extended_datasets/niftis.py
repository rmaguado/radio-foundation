import logging
import numpy as np
import torch
import nibabel as nib
from typing import Tuple, Any
import os

from dinov2.data.datasets.niftis import NiftiCtVolumesFull


logger = logging.getLogger("dinov2")


class NiftiFullVolumeEval(NiftiCtVolumesFull):
    def __init__(
        self,
        root_path: str,
        dataset_name: str,
        channels: int = 1,
        transform=None,
    ):
        super().__init__(
            dataset_name=dataset_name,
            root_path=root_path,
            channels=channels,
            transform=transform,
        )

    def create_entries(self) -> np.ndarray:
        """
        Generates a numpy memmap object pointing to the sqlite database rows of dicom paths.

        Returns:
            np.ndarray: The entries dataset (memmap).
        """
        logger.info(f"Creating entries for {self.dataset_name}.")

        entries_dtype = [
            ("rowid", np.uint32),
            ("map_id", "U256"),
        ]
        entries = []
        row_map_id = self.cursor.execute(f"SELECT rowid, map_id FROM global").fetchall()

        for rowid, map_id in row_map_id:
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

        num_slices = volume_data.shape[0]
        num_stacks = num_slices // self.channels
        num_slices = num_stacks * self.channels

        image_width = volume_data.shape[1]
        image_height = volume_data.shape[2]

        volume_data = volume_data[:num_slices]
        volume_data = torch.tensor(volume_data).view(
            num_stacks, self.channels, image_width, image_height
        )

        return self.process_ct(volume_data), map_id

    def get_index_from_map_id(self, map_id: str) -> int:
        return np.where(self.entries["map_id"] == map_id)[0][0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        try:
            image, map_id = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e

        transformed_image = self.transform(image)

        return transformed_image, map_id

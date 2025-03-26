import torch
import numpy as np
import nibabel as nib
import os
from typing import Tuple, Any
from einops import rearrange

from dinov2.data.datasets.niftis import NiftiCtVolumesFull


class RadiologyReportDataset(NiftiCtVolumesFull):
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

        entries_dtype = [
            ("rowid", np.uint32),
        ]
        entries = []
        row_ids = self.cursor.execute(f"SELECT rowid FROM global").fetchall()

        for rowid in row_ids:
            entries.append((rowid,))

        entries_array = np.array(entries, dtype=entries_dtype)

        entries_dir = self.get_entries_dir()
        np.save(entries_dir, entries_array)
        return np.load(entries_dir, mmap_mode="r")

    def get_image_data(self, index: int) -> torch.Tensor:
        rowid = self.entries[index]
        self.cursor.execute(
            """
            SELECT dataset, axial_dim, nifti_path, report FROM global WHERE rowid = ?
            """,
            (int(rowid),),
        )
        dataset, axial_dim, nifti_path, report = self.cursor.fetchone()

        abs_path_to_nifti = os.path.join(self.root_path, dataset, nifti_path)
        nifti_file = nib.load(abs_path_to_nifti)

        volume_data = nifti_file.get_fdata().astype(np.float32)
        volume_data = np.moveaxis(volume_data, axial_dim, 0)

        num_slices = volume_data.shape[0]
        num_stacks = num_slices // self.channels
        num_slices = num_stacks * self.channels

        volume_data = volume_data[:num_slices]
        volume_data = torch.from_numpy(volume_data)
        volume_data = rearrange(
            volume_data, "(s c) w h -> s c w h", s=num_stacks, c=self.channels
        )

        return self.process_ct(volume_data), report

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        try:
            image, report = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image/report for sample {index}") from e

        transformed_image = self.transform(image)

        return transformed_image, report

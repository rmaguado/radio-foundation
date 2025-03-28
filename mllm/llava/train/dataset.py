import torch
from torchvision import transforms
import numpy as np
import nibabel as nib
import os
from typing import Tuple, Any
from einops import rearrange

from dinov2.data.datasets.niftis import NiftiCtVolumesFull


class ImageProcessor:
    def __init__(self, img_size, mean, std, min_zspacing=1.0, channels=10):
        self.img_size = img_size
        self.normalize = transforms.Normalize(mean=mean, std=std)

        self.min_zspacing = min_zspacing
        self.channels = channels

    def resize(self, image, slice_thickness):
        slices, w, h = image.shape

        target_height = self.img_size * h // w

        if slice_thickness < self.min_zspacing:
            target_slices = int(slices * self.min_zspacing / slice_thickness)
        else:
            target_slices = slices

        groups = target_slices // self.channels
        target_slices = groups * self.channels

        image = image.unsqueeze(0).unsqueeze(0)
        image = torch.nn.functional.interpolate(
            image, size=(target_slices, self.img_size, target_height), mode="trilinear"
        )
        return rearrange(image, "1 1 (g c) w h -> g c w h", g=groups, c=self.channels)

    def __call__(self, image, slice_thickness):
        image = self.resize(image, slice_thickness)
        image = self.normalize(image)
        return image


class RadiologyReportDataset(NiftiCtVolumesFull):
    def __init__(
        self,
        root_path: str,
        dataset_name: str,
        channels: int = 10,
        img_size: int = 504,
        mean: float = 0.0,
        std: float = 1.0,
        min_zspacing=1.0,
    ):
        super().__init__(
            dataset_name=dataset_name,
            root_path=root_path,
            channels=channels,
            transform=None,
        )

        self.image_processor = ImageProcessor(
            img_size, mean, std, min_zspacing, channels
        )

    def create_entries(self) -> np.ndarray:
        """
        Generates a numpy memmap object pointing to the sqlite database rows of dicom paths.

        Returns:
            np.ndarray: The entries dataset (memmap).
        """

        entries_dtype = [
            ("rowid", np.uint32),
            ("length", np.uint32),
        ]
        entries = []
        row_id_lengths = self.cursor.execute(
            f"SELECT rowid, length FROM global"
        ).fetchall()

        for rowid, length in row_id_lengths:
            entries.append((rowid, length))

        entries_array = np.array(entries, dtype=entries_dtype)

        entries_dir = self.get_entries_dir()
        np.save(entries_dir, entries_array)
        return np.load(entries_dir, mmap_mode="r")

    def get_lengths(self):
        return self.entries["length"]

    def process_report(self, report_text):
        return report_text

    def get_image_data(self, index: int) -> torch.Tensor:
        rowid, _ = self.entries[index]
        self.cursor.execute(
            """
            SELECT dataset, axial_dim, nifti_path, text, slice_thickness FROM global WHERE rowid = ?
            """,
            (int(rowid),),
        )
        dataset, axial_dim, nifti_path, report, slice_thickness = self.cursor.fetchone()

        abs_path_to_nifti = os.path.join(self.root_path, dataset, nifti_path)
        nifti_file = nib.load(abs_path_to_nifti)

        volume_data = nifti_file.get_fdata().astype(np.float32)
        volume_data = np.moveaxis(volume_data, axial_dim, 0)
        volume_data = torch.from_numpy(volume_data)
        volume_data = self.image_processor(volume_data, slice_thickness)

        return self.process_ct(volume_data), report

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        try:
            image, report = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image/report for sample {index}") from e

        return image, report

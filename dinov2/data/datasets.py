import os
import torch
from typing import List, Tuple, Any, Callable
import polars as pl
import numpy as np
import SimpleITK as sitk
import logging


logger = logging.getLogger("dinov2")


class VolumeDataset:
    def __init__(
        self,
        index_path: str,
        modality: str,
        transform: Callable = lambda x: x,
    ) -> None:
        self.df = pl.read_csv(index_path)
        self.modality = modality
        self.transform = transform

    def resample_to_isotropic(self, image, new_spacing):
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        new_size = [
            int(round(osz * ospc / nspc))
            for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkLinear)

        return resampler.Execute(image)

    def __len__(self) -> int:
        return len(self.df)

    def get_image_data(self, idx: int) -> sitk.Image:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = self.get_image_data(idx)

        original_spacing = image.GetSpacing()
        ratio = max(original_spacing) / min(original_spacing)

        if ratio > 3.0:
            new_spacing = (max(original_spacing) / 2,) * 3
        else:
            new_spacing = (min(original_spacing),) * 3

        image = self.resample_to_isotropic(image, new_spacing)

        # orientation = image.GetDirection()

        volume = sitk.GetArrayFromImage(image)  # (slices, height, width)
        volume_tensor = torch.tensor(volume, dtype=torch.float32)

        augmentations = self.transform(volume_tensor)

        return augmentations


class DicomVolumeDataset(VolumeDataset):

    def get_image_data(self, idx: int) -> sitk.Image:
        meta = self.df.row(idx)
        dicom_folder = meta["path"]

        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(dicom_folder)
        assert series_IDs, f"No DICOM series found in: {dicom_folder}"
        series_file_names = reader.GetGDCMSeriesFileNames(dicom_folder, series_IDs[0])
        reader.SetFileNames(series_file_names)
        image = reader.Execute()

        return image


class NiftiVolumeDataset(VolumeDataset):

    def get_image_data(self, idx: int) -> sitk.Image:
        meta = self.df.row(idx)
        nifti_file_path = meta["path"]

        image = sitk.ReadImage(nifti_file_path)

        return image


class MultiDataset:

    def __init__(self, datasets: list) -> None:
        """
        Initializes a MultiDataset object for collating various dataset objects.

        Args:
            datasets (list): A list of datasets.
        """
        self.datasets = datasets
        self.cumulative_sizes = np.cumsum([len(d) for d in datasets])

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def get_dataset_sizes(self) -> List[int]:
        return [len(d) for d in self.datasets]

    def get_dataset_names(self) -> List[str]:
        return [d.dataset_name for d in self.datasets]

    def _find_dataset_idx(self, idx: int) -> int:
        return int(np.searchsorted(self.cumulative_sizes, idx, side="right"))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        dataset_idx = self._find_dataset_idx(index)

        if dataset_idx > 0:
            dataset_index = index - self.cumulative_sizes[dataset_idx - 1]
        else:
            dataset_index = index

        return self.datasets[dataset_idx][dataset_index]

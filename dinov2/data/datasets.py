import os
import torch
from typing import List, Dict, Tuple, Any, Callable
import polars as pl
import numpy as np

import nibabel as nib
import pydicom
import logging


logger = logging.getLogger("dinov2")


class VolumeDataset:
    """
    Base class for volumetric medical image datasets.

    Handles loading of dataset metadata, resampling of images to isotropic spacing,
    and application of transformations. Subclasses should implement the get_image_data method.

    Args:
        dataset_name (str): Name of the dataset.
        index_path (str): Path to the CSV file containing dataset index/metadata.
        modality (str): Imaging modality (e.g., 'ct', 'mri').
        transform (Callable): Transformation function to apply to the loaded volume.
    """

    def __init__(
        self,
        dataset_name: str,
        index_path: str,
        modality: str,
        mean: float,
        std: float,
        bounds: Tuple[float, float] = (-1000, 1900),
    ) -> None:
        self.dataset_name = dataset_name
        self.df = pl.read_csv(index_path)
        self.modality = modality
        self.mean = mean
        self.std = std
        self.bounds = bounds

    def __len__(self) -> int:
        return len(self.df)

    def get_norm(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std

    def get_image_data(self, idx: int) -> Tuple[torch.Tensor, Tuple[float, ...]]:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[float, ...]]:

        image, spacing = self.get_image_data(idx)
        image = self.get_norm(image)

        return image, spacing


class DicomVolumeDataset(VolumeDataset):
    """
    Dataset class for loading volumetric DICOM image series.

    Inherits from VolumeDataset and implements get_image_data for DICOM folders.
    """

    def get_image_data(self, idx: int):
        meta = self.df.row(idx)
        dicom_folder_path = meta[0]
        spacing = meta[5:8]

        dicom_file_paths = [
            x for x in os.listdir(dicom_folder_path) if x.endswith(".dcm")
        ]

        dcms = [pydicom.dcmread(x) for x in dicom_file_paths]
        dcms.sort(key=lambda x: x.ImagePositionPatient[2])

        image_array = torch.stack(
            [torch.from_numpy(dcm.pixel_array).float() for dcm in dcms]
        )

        if self.modality == "ct":
            slope = dcms[0].RescaleSlope
            intercept = dcms[0].RescaleIntercept
            image_array = image_array * slope + intercept

        image_array = torch.clip(image_array, self.bounds[0], self.bounds[1])

        return image_array, spacing


class NiftiVolumeDataset(VolumeDataset):
    """
    Dataset class for loading volumetric NIfTI image files.

    Inherits from VolumeDataset and implements get_image_data for NIfTI files.
    """

    def get_image_data(self, idx: int) -> Tuple[torch.Tensor, Tuple[float, ...]]:
        meta = self.df.row(idx)
        nifti_file_path = meta[0]
        spacing = meta[5:8]

        image = nib.loadsave.load(nifti_file_path)

        image_memmap = image.dataobj  # pyright: ignore[reportAttributeAccessIssue]

        if self.modality == "ct":
            slope = image_memmap.slope
            intercept = image_memmap.inter

            slope = 1.0 if slope is None else slope
            intercept = 0.0 if intercept is None else intercept

            image_array = np.asarray(image_memmap, dtype=np.float32)
            image_array = slope * image_array + intercept
            image_array = torch.from_numpy(image_array)

        elif self.modality == "mri":
            image_array = torch.from_numpy(np.array(image_memmap))

        else:
            raise ValueError(f"Unrecognized modality: {self.modality}.")

        image_array = torch.clip(image_array, self.bounds[0], self.bounds[1])

        return image_array, spacing


class MultiDataset:
    """
    Collates multiple datasets into a single dataset interface.

    Allows indexing across multiple datasets as if they were a single dataset.

    Args:
        datasets (list): List of dataset objects to combine.
    """

    def __init__(self, datasets: list) -> None:
        """
        Initializes a MultiDataset object for collating various dataset objects.

        Args:
            datasets (list): A list of datasets.
        """
        self.datasets = datasets
        self.cumulative_sizes = np.cumsum([len(d) for d in datasets])

    def __len__(self) -> int:
        """
        Returns the total number of samples across all datasets.

        Returns:
            int: Total number of samples.
        """
        return self.cumulative_sizes[-1]

    def get_dataset_sizes(self) -> List[int]:
        """
        Returns the sizes of each individual dataset.

        Returns:
            List[int]: List of dataset sizes.
        """
        return [len(d) for d in self.datasets]

    def get_dataset_names(self) -> List[str]:
        """
        Returns the names of each individual dataset.

        Returns:
            List[str]: List of dataset names.
        """
        return [d.dataset_name for d in self.datasets]

    def _find_dataset_idx(self, idx: int) -> int:
        """
        Finds which dataset a global index belongs to.

        Args:
            idx (int): Global index.

        Returns:
            int: Index of the dataset in the datasets list.
        """
        return int(np.searchsorted(self.cumulative_sizes, idx, side="right"))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a sample from the appropriate dataset based on the global index.

        Args:
            index (int): Global index across all datasets.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of transformed image views.
        """
        dataset_idx = self._find_dataset_idx(index)

        if dataset_idx > 0:
            dataset_index = index - self.cumulative_sizes[dataset_idx - 1]
        else:
            dataset_index = index

        return self.datasets[dataset_idx][dataset_index]

import torch
from typing import List, Dict, Callable
import polars as pl
import numpy as np
import SimpleITK as sitk
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
        transform: Callable = lambda x: x,
    ) -> None:
        self.dataset_name = dataset_name
        self.df = pl.read_csv(index_path)
        self.modality = modality
        self.transform = transform

    def resample_to_isotropic(self, image, new_spacing):
        """
        Resample a SimpleITK image to isotropic voxel spacing.

        Args:
            image (sitk.Image): The input image to resample.
            new_spacing (tuple): The desired isotropic spacing (x, y, z).

        Returns:
            sitk.Image: The resampled image.
        """
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
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.df)

    def get_image_data(self, idx: int) -> sitk.Image:
        """
        Abstract method to retrieve the image data for a given index.
        Should be implemented by subclasses.

        Args:
            idx (int): Index of the sample.

        Returns:
            sitk.Image: The loaded image.
        """
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Loads and processes a sample from the dataset.
        Returns a dictionary of transformed image views.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of transformed image views.
        """

        image = self.get_image_data(idx)

        original_spacing = image.GetSpacing()
        ratio = max(original_spacing) / min(original_spacing)

        if ratio > 3.0:
            new_spacing = (max(original_spacing) / 2,) * 3
        else:
            new_spacing = (min(original_spacing),) * 3

        image = self.resample_to_isotropic(image, new_spacing)

        volume = sitk.GetArrayFromImage(image)  # (slices, height, width)
        volume_tensor = torch.tensor(volume, dtype=torch.float32)

        return self.transform(volume_tensor)


class DicomVolumeDataset(VolumeDataset):
    """
    Dataset class for loading volumetric DICOM image series.

    Inherits from VolumeDataset and implements get_image_data for DICOM folders.
    """

    def get_image_data(self, idx: int) -> sitk.Image:
        """
        Loads a DICOM image series as a SimpleITK image.

        Args:
            idx (int): Index of the sample.

        Returns:
            sitk.Image: The loaded DICOM image volume.
        """
        meta = self.df.row(idx)
        dicom_folder = meta[0]

        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(dicom_folder)
        assert series_IDs, f"No DICOM series found in: {dicom_folder}"
        series_file_names = reader.GetGDCMSeriesFileNames(dicom_folder, series_IDs[0])
        reader.SetFileNames(series_file_names)
        image = reader.Execute()

        return image


class NiftiVolumeDataset(VolumeDataset):
    """
    Dataset class for loading volumetric NIfTI image files.

    Inherits from VolumeDataset and implements get_image_data for NIfTI files.
    """

    def get_image_data(self, idx: int) -> sitk.Image:
        """
        Loads a NIfTI image file as a SimpleITK image.

        Args:
            idx (int): Index of the sample.

        Returns:
            sitk.Image: The loaded NIfTI image volume.
        """
        meta = self.df.row(idx)
        nifti_file_path = meta[0]

        image = sitk.ReadImage(nifti_file_path)

        return image


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

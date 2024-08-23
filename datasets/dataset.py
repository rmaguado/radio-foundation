import os
import h5py
import json
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from tqdm import tqdm
from abc import ABC, abstractmethod


class DatasetBase(ABC):
    def __init__(self, config: dict):
        self.config = config

        self.target_path = config["target_path"]
        self.chunk_size = config.get("chunk_size", (1, 512, 512))
        self.statistics = StatisticsManager()

        sitk.ProcessObject_SetGlobalWarningDisplay(False)

    @abstractmethod
    def get_patient_ids(self) -> list:
        pass

    @abstractmethod
    def extend_metadata(self, metadata: dict) -> None:
        pass

    def get_series_processor(self):
        return SeriesProcessorBase(self.config, self.statistics)

    def prepare_dataset(self):
        assert hasattr(
            self, "get_patient_ids"
        ), "The method 'get_patient_ids' must be implemented."

        patient_ids = self.get_patient_ids()
        processor = self.get_series_processor()
        series_number = 0
        for patient_id, series_path in tqdm(patient_ids):
            series_id = f"series_{series_number:04d}"

            image_array, metadata = processor.process_series(
                series_path, series_id, patient_id
            )
            self.extend_metadata(metadata)
            series_save_path = os.path.join(self.target_path, series_id)
            processor.save_array(image_array, series_save_path, "image.h5")
            processor.save_metadata(metadata, series_save_path, "metadata.json")
            series_number += 1

        self.statistics.save_global_metadata(self.target_path)


class SeriesProcessorBase:
    def __init__(self, config: dict, statistics_manager):
        self.config = config
        self.clip_min = config["clip_min"]
        self.clip_max = config["clip_max"]
        self.clip_window = self.clip_max - self.clip_min
        self.z_spacing = config["z_spacing"]
        self.statistics_manager = statistics_manager

        assert (
            self.clip_window > 0
        ), "Clip window must be positive (clip_max - clip_min > 0)."

    def process_series(self, path_to_series, series_id, patient_id):
        image = self.get_image(path_to_series, patient_id)
        original_spacing = self.get_spacing(image)
        image_array, resampled_spacing = self.preprocess_image(image)

        metadata = {
            "dataset": self.config["dataset"],
            "series_id": series_id,
            "patient_id": patient_id,
            "shape": list(image_array.shape),
            "spacing": {"original": original_spacing, "resampled": resampled_spacing},
            "other": dict(),
        }

        return image_array, metadata

    def get_image(self, path_to_series, patient_id):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path_to_series)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        return image

    def preprocess_image(self, image: sitk.Image):
        assert isinstance(image, sitk.Image), "Image must be a SimpleITK image."

        resampled_image, resampled_spacing = self.resample_image(image)
        image_array = sitk.GetArrayViewFromImage(resampled_image)
        clipped_hu_array = self.clip_hu(image_array)

        self.statistics_manager.update_statistics(clipped_hu_array)

        return clipped_hu_array, resampled_spacing

    def clip_hu(self, image_array):
        clipped_array = np.clip(image_array, self.clip_min, self.clip_max)
        standardized_array = (clipped_array - self.clip_min) / self.clip_window
        return standardized_array.astype(np.float32)

    def get_spacing(self, image):
        spacing = image.GetSpacing()
        return [spacing[i] for i in [2, 0, 1]]

    def resample_image(self, image):
        current_spacing = image.GetSpacing()
        current_size = image.GetSize()

        desired_spacing = (current_spacing[0], current_spacing[1], self.z_spacing)
        desired_size = [
            int(round(current_size[i] * (current_spacing[i] / desired_spacing[i])))
            for i in range(3)
        ]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(desired_spacing)
        resample.SetSize(desired_size)
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        resampled_image = resample.Execute(image)

        resampled_spacing = self.get_spacing(resampled_image)

        return resampled_image, resampled_spacing

    def save_array(self, array_data, path_to_save, filename="image.h5"):
        assert isinstance(array_data, np.ndarray), "Array data must be a numpy array."
        if not array_data.shape:
            raise ValueError("Array data must have a shape.")

        os.makedirs(path_to_save, exist_ok=True)
        with h5py.File(os.path.join(path_to_save, filename), "w") as f:
            f.create_dataset(
                "data",
                data=array_data,
                compression="lzf",
                chunks=self.config["chunk_size"],
            )

    def save_metadata(self, metadata, path_to_save, filename="metadata.json"):
        assert isinstance(metadata, dict), "Metadata must be a dictionary."

        os.makedirs(path_to_save, exist_ok=True)
        with open(os.path.join(path_to_save, filename), "w") as f:
            json.dump(metadata, f, indent=4)


class StatisticsManager:
    def __init__(self):
        self.total_voxels = 0
        self.sum_voxels = 0.0
        self.sum_squares_voxels = 0.0

    def update_statistics(self, image_array):
        """
        Update the running sum and sum of squares of voxel intensities.
        """
        self.total_voxels += image_array.size
        self.sum_voxels += np.sum(image_array)
        self.sum_squares_voxels += np.sum(np.square(image_array))

    def finalize_statistics(self):
        """
        Calculate and return the final mean and standard deviation.
        """
        mean = self.sum_voxels / self.total_voxels
        variance = (self.sum_squares_voxels / self.total_voxels) - (mean**2)
        std_dev = np.sqrt(variance)
        return mean, std_dev

    def save_global_metadata(self, target_path):
        """
        Save the global mean and standard deviation to the target path.
        """
        mean, std_dev = self.finalize_statistics()
        global_metadata = {
            "mean": mean,
            "std_dev": std_dev,
            "total_voxels": self.total_voxels,
        }
        with open(os.path.join(target_path, "metadata.json"), "w") as f:
            json.dump(global_metadata, f, indent=4)

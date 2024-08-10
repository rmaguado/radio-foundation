import os
import h5py
import json
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from tqdm import tqdm
from abc import ABC, abstractmethod

class CtDataset:
    def __init__(
        self,
        dataset: str,
        target_path: str,
        z_spacing: float = 2.0,
        clip_min: int = -1024,
        clip_max: int = 3072,
        chunk_size: tuple = (1, 512, 512)
    ):
        self.dataset = dataset
        self.target_path = target_path
        
        self.z_spacing = z_spacing
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.clip_window = self.clip_max - self.clip_min
        
        self.chunk_size = chunk_size
        
        self.total_voxels = 0
        self.sum_voxels = 0.0
        self.sum_squares_voxels = 0.0
        
        assert self.clip_window > 0
        
    def get_patient_ids(self) -> list:
        """
        Should return a list of patient identifiers used to find scans of that patient.
        """
        raise NotImplementedError
    
    def get_patient_series_paths(self, patient_id: str) -> list:
        """
        Should return a list of paths to the DICOM series of a patient.
        """
        raise NotImplementedError
        
    def get_image(self, path_to_series):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path_to_series)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        return image
        
    def clip_hu(self, image_array):
        clipped_array = np.clip(image_array, self.clip_min, self.clip_max)
        standardized_array = (clipped_array - self.clip_min) / self.clip_window
        return standardized_array.astype(np.float32)
    
    def preprocess_image(self, image):
        resampled_image, resampled_spacing = self.resample_image(image)
        image_array = sitk.GetArrayViewFromImage(resampled_image)
        clipped_hu_array = self.clip_hu(image_array)
        
        self.update_statistics(clipped_hu_array)
        
        return clipped_hu_array, resampled_spacing
    
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
        variance = (self.sum_squares_voxels / self.total_voxels) - (mean ** 2)
        std_dev = np.sqrt(variance)
        return mean, std_dev
    
    def save_global_metadata(self):
        """
        Save the global mean and standard deviation to the target path.
        """
        mean, std_dev = self.finalize_statistics()
        global_metadata = {
            "mean": mean,
            "std_dev": std_dev,
            "total_voxels": self.total_voxels
        }
        with open(os.path.join(self.target_path, "metadata.json"), 'w') as f:
            json.dump(global_metadata, f, indent=4)
    
    def get_spacing(self, image):
        spacing = image.GetSpacing()
        return [spacing[i] for i in [2,0,1]]
        
    def resample_image(self, image):
        current_spacing = image.GetSpacing()
        current_size = image.GetSize()

        desired_spacing = (current_spacing[0], current_spacing[1], self.z_spacing)
        desired_size = [
            int(round(current_size[i] * (current_spacing[i] / desired_spacing[i]))) \
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

        return resampled_ct_image, resampled_spacing
        
    def process_series(self, path_to_series, series_id, patient_id):
        image = self.get_image(path_to_series)
        original_spacing = self.get_spacing(image)
        
        image_array, resampled_spacing = self.preprocess_image(image)

        metadata = {
            "dataset": self.dataset,
            "series_id": series_id,
            "patient_id": patient_id,
            "shape" : list(image_array.shape),
            "spacing": {
                "original": original_spacing,
                "resampled": resampled_spacing
            },
            "other": dict()
        }

        return image_array, metadata
    
    def save_array(self, array_data, path_to_save, filename="image.h5"):
        os.makedirs(path_to_save, exist_ok=True)
        with h5py.File(os.path.join(path_to_save, filename), 'w') as f:
            f.create_dataset('data', data=array_data, compression='lzf', chunks=self.chunk_size)
            
    def save_metadata(self, metadata, path_to_save, filename="metadata.json"):
        os.makedirs(path_to_save, exist_ok=True)
        with open(os.path.join(path_to_save, filename), 'w') as f:
            json.dump(metadata, f, indent=4)
        
    def extend_metadata(self, metadata: dict) -> None:
        """
        Should add any additional information to the "other" section of metadata.
        """
        pass
        
    def prepare_dataset(self):
        patient_ids = self.get_patient_ids()
        
        series_number = 0
        for patient_id in tqdm(patient_ids):
            series_paths = self.get_patient_series_paths(patient_id)
            for series_path in series_paths:
                series_id = f"series_{series_number:04d}"
                image_array, metadata = self.process_series(series_path, series_id, patient_id)
                self.extend_metadata(metadata)
                series_save_path = os.path.join(self.target_path, series_id)
                self.save_array(image_array, series_save_path, "image.h5")
                self.save_metadata(metadata, series_save_path, "metadata.json")
                series_number += 1

        self.save_global_metadata()

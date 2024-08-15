import os
import h5py
import json
import numpy as np
from tqdm import tqdm
import pylidc as pl
from pylidc.utils import consensus
import SimpleITK as sitk

from datasets import DatasetBase, SeriesProcessorBase


np.int = int
np.bool = bool

class SeriesProcessor(SeriesProcessorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def resample_image(self, image, mask=None):
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
        
        resample.SetSize(desired_size)
        resample.SetOutputSpacing(desired_spacing)
        
        resampled_spacing = [desired_spacing[2], desired_spacing[0], desired_spacing[1]]
        
        if mask is not None:
            mask_image = sitk.GetImageFromArray(mask.astype(np.uint8))
            mask_image.SetSpacing(image.GetSpacing())
            mask_image.SetDirection(image.GetDirection())
            mask_image.SetOrigin(image.GetOrigin())
            
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
            resampled_mask = resample.Execute(mask_image)
            
            resampled_mask_array = sitk.GetArrayFromImage(resampled_mask).astype(bool)
            return resampled_image, resampled_mask_array, resampled_spacing
        
        return resampled_image, None, resampled_spacing

    def preprocess_image(self, image, mask=None):
        resampled_image, resampled_mask, resampled_spacing = self.resample_image(image, mask)
        image_array = sitk.GetArrayViewFromImage(resampled_image)
        clipped_hu_array = self.clip_hu(image_array)
        
        if mask is not None:
            assert clipped_hu_array.shape == resampled_mask.shape
        
        self.statistics_manager.update_statistics(clipped_hu_array)
        
        return clipped_hu_array, resampled_mask, resampled_spacing
    
    def process_series(self, scan, path_to_series, series_id, patient_id):
        image = self.get_image(path_to_series)
        original_spacing = self.get_spacing(image)
        
        segmentation_mask = np.zeros_like(
            sitk.GetArrayViewFromImage(image), dtype=bool
        )
        
        nods = scan.cluster_annotations(verbose=False)
        metadata = {
            "dataset": self.config['dataset'],
            "series_id": series_id,
            "patient_id": patient_id,
            "shape": None,
            "spacing": {
                "original": original_spacing,
                "resampled": None
            },
            "other": {
                "nodules_count": len(nods)
            }
        }
        
        for nodule_id, anns in enumerate(nods, 1):
            cmask, bbox, _ = consensus(anns, clevel=0.5)

            cmask = np.transpose(cmask, (2, 0, 1))
            bbox = (bbox[2], bbox[0], bbox[1])

            segmentation_mask[bbox] = cmask
        
        image_array, resampled_mask, resampled_spacing = self.preprocess_image(image, segmentation_mask)
        
        metadata["shape"] = image_array.shape
        metadata["spacing"]["resampled"] = resampled_spacing
        
        return image_array, resampled_mask, metadata


class LidcIdri(DatasetBase):
    def __init__(self, config):
        super().__init__(config)
        
    def get_patient_ids(self):
        scans = pl.query(pl.Scan).all()
        patient_ids = list(set(scan.patient_id for scan in scans))
        patient_ids.sort()
        
        patient_ids_series = []
        for patient_id in patient_ids:
            scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
            patient_ids_series += [(patient_id, scan.get_path_to_dicom_files(), scan) for scan in scans]
        
        return patient_ids_series
    
    def extend_metadata(self, metadata):
        pass
    
    def prepare_dataset(self):
        patient_ids = self.get_patient_ids()
        
        series_number = 0
        for patient_id, series_path, scan in tqdm(patient_ids):
            series_id = f"series_{series_number:04d}"
            processor = SeriesProcessor(self.config, self.statistics)
            image_array, mask_array, metadata = processor.process_series(scan, series_path, series_id, patient_id)

            series_save_path = os.path.join(self.target_path, series_id)
            processor.save_array(image_array, series_save_path, "image.h5")
            processor.save_array(mask_array, series_save_path, "mask.h5")
            processor.save_metadata(metadata, series_save_path, "metadata.json")
            series_number += 1
                
        self.statistics.save_global_metadata(self.target_path)

def main():
    config = {
        "dataset": "LIDC-IDRI",
        "target_path": "datasets/LIDC-IDRI/data",
        "z_spacing": 2.0,
        "clip_min": -1024,
        "clip_max": 3072,
        "chunk_size": (1, 512, 512)
    }
    
    dataprep = LidcIdri(config)
    dataprep.prepare_dataset()

if __name__ == "__main__":
    main()

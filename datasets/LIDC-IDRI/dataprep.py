import os
import h5py
import json
import numpy as np
from tqdm import tqdm
import pylidc as pl
from pylidc.utils import consensus
import SimpleITK as sitk

from datasets import CtDataset


np.int = int
np.bool = bool

class LidcIdri(CtDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_patient_ids(self):
        scans = pl.query(pl.Scan).all()
        patient_ids = set(scan.patient_id for scan in scans)

        return list(patient_ids)
        
    def get_patient_paths_scans(self, patient_id):
        scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
        
        return [(scan.get_path_to_dicom_files(), scan) for scan in scans]
    
    def preprocess_image(self, image, mask):
        resampled_image, resampled_mask, resampled_spacing = self.resample_image(image, mask)
        image_array = sitk.GetArrayViewFromImage(resampled_image)
        clipped_hu_array = self.clip_hu(image_array)
        
        assert clipped_hu_array.shape == resampled_mask.shape
        
        return clipped_hu_array, resampled_mask, resampled_spacing
        
    def resample_image(self, image, mask):
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
        
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        resampled_mask = resample.Execute(sitk.GetImageFromArray(mask.astype(np.int8)))

        resampled_mask_array = sitk.GetArrayFromImage(resampled_mask).astype(bool)

        return resampled_image, resampled_mask_array, resampled_spacing
        
    def process_series(self, scan, path_to_series, series_id, patient_id):
        image = self.get_image(path_to_series)
        original_spacing = self.get_spacing(image)
        
        
        segmentation_mask = np.zeros_like(
            sitk.GetArrayViewFromImage(image), dtype=bool
        )
        
        nods = scan.cluster_annotations(verbose=False)
        metadata = {
            "dataset": self.dataset,
            "series_id": series_id,
            "patient_id": patient_id,
            "shape" : None,
            "spacing": {
                "original": original_spacing,
                "resampled": None
            },
            "other": {
                "nodules": [],
                "has_nodule": len(nods) > 0
            }
        }
        
        for nodule_id, anns in enumerate(nods, 1):
            cmask, bbox, _ = consensus(anns, clevel=0.5)

            cmask = np.transpose(cmask, (2,0,1))
            bbox = (bbox[2], bbox[0], bbox[1])

            segmentation_mask[bbox] = cmask

            nodule_info = {"nodule_id": nodule_id}
            nodule_info["bbox"] = {
                "slice_range": [int(bbox[0].start), int(bbox[0].stop)],
                "row_range": [int(bbox[1].start), int(bbox[1].stop)],
                "column_range": [int(bbox[2].start), int(bbox[2].stop)]
            }
            metadata["other"]["nodules"].append(nodule_info)
            
        image_array, resampled_mask, resampled_spacing = self.preprocess_image(image, segmentation_mask)
        metadata["shape"] = image_array.shape
        metadata["spacing"]["resampled"] = resampled_spacing

        return image_array, resampled_mask, metadata
    
    def prepare_dataset(self):
        patient_ids = self.get_patient_ids()
        
        series_number = 0
        for patient_id in tqdm(patient_ids):
            for (series_path, scan) in self.get_patient_paths_scans(patient_id):
                series_id = f"series_{series_number:04d}"
                image_array, mask_array, metadata = self.process_series(scan, series_path, series_id, patient_id)
                
                series_save_path = os.path.join(self.target_path, series_id)
                self.save_array(image_array, series_save_path, "image.h5")
                self.save_array(mask_array, series_save_path, "mask.h5")
                self.save_metadata(metadata, series_save_path, "metadata.json")
                series_number += 1

def main():
    Dataprep = LidcIdri(
        dataset="datasets/LIDC-IDRI",
        target_path="datasets/LIDC-IDRI/data",
    )
    Dataprep.prepare_dataset()

if __name__ == "__main__":
    main()

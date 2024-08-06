import os
import h5py
import json
import numpy as np
from tqdm import tqdm
import pylidc as pl
from pylidc.utils import consensus
import SimpleITK as sitk
from sklearn.model_selection import train_test_split

np.int = int
np.bool = bool

def get_patient_ids():
    scans = pl.query(pl.Scan).all()
    patient_ids = set(scan.patient_id for scan in scans)
    
    return list(patient_ids)


def process_scan(scan, patient_id, series_id):
    dicom_dir = scan.get_path_to_dicom_files()
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    img_spacing = image.GetSpacing()
    img_size = image.GetSize()
    
    nods = scan.cluster_annotations(verbose=False)
    
    image_array = sitk.GetArrayViewFromImage(image)
    clip_hu = (np.clip(image_array, -1024, 3072) + 1024) / 4096
    
    segmentation_mask = np.zeros_like(clip_hu, dtype="bool")
    
    metadata = {
        "dataset": "LIDC-IDRI",
        "series_id": series_id,
        "patient_id": patient_id,
        "spacing": [img_spacing[i] for i in [2,0,1]],
        "shape" : [img_size[i] for i in [2,0,1]],
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

    return clip_hu, segmentation_mask, metadata


def save_data(series_target_dir, clip_hu, segmentation_mask, metadata):
    os.makedirs(series_target_dir, exist_ok=True)
    with h5py.File(os.path.join(series_target_dir, 'image.h5'), 'w') as f:
        f.create_dataset('data', data=clip_hu, compression='lzf', chunks=(1, 512, 512))
    with h5py.File(os.path.join(series_target_dir, 'mask.h5'), 'w') as f:
        f.create_dataset('data', data=segmentation_mask, compression='lzf', chunks=(1, 512, 512))

    with open(os.path.join(series_target_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)


def main():
    target_dir = "./datasets/LIDC-IDRI/data"

    patients = get_patient_ids()
    
    series_number = 0
    for patient_id in tqdm(patients):
        scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
        if not scans:
            continue

        for scan in scans:
            series_id = f'series_{series_number:04d}'
            clip_hu, segmentation_mask, metadata = process_scan(scan, patient_id, series_id)
            series_target_dir = os.path.join(target_dir, series_id)
            save_data(series_target_dir, clip_hu, segmentation_mask, metadata)
            
            series_number += 1
    
if __name__ == "__main__":
    main()

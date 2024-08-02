import os
import h5py
import json
import numpy as np
from tqdm import tqdm
import pylidc as pl
from pylidc.utils import consensus
import SimpleITK as sitk
from sklearn.model_selection import train_test_split


def create_directories(root_dir, subsets=['train', 'test', 'val']):
    for subset in subsets:
        os.makedirs(os.path.join(root_dir, subset), exist_ok=True)


def get_patient_ids():
    scans = pl.query(pl.Scan).all()
    patient_ids = set(scan.patient_id for scan in scans)
    
    return list(patient_ids)


def process_scan(scan, patient_id):
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
        "patient_id": patient_id,
        "scan_id": scan.id,
        "spacing": [img_spacing[i] for i in [2,0,1]],
        "shape" : [img_size[i] for i in [2,0,1]],
        "nodules": []
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
        metadata["nodules"].append(nodule_info)

    return clip_hu, segmentation_mask, metadata


def process_patient(patient_id, target_dir):
    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
    if not scans:
        print(f"No scans found for {patient_id}")
        return

    for scan in scans:
        clip_hu, segmentation_mask, metadata = process_scan(scan, patient_id)
        scan_folder = f'{patient_id}_{scan.id}'
        scan_target_dir = os.path.join(target_dir, scan_folder)
        os.makedirs(scan_target_dir, exist_ok=True)

        chunk_size = (1, 512, 512)
        with h5py.File(os.path.join(scan_target_dir, 'mask.h5'), 'w') as f:
            f.create_dataset('data', data=segmentation_mask, compression='gzip', compression_opts=1, chunks=chunk_size)

        np.save(os.path.join(scan_target_dir, 'image.npy'), clip_hu)

        with open(os.path.join(scan_target_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)


def main():
    target_dir = "./LIDC_IDRI"
    create_directories(target_dir)

    patients = get_patient_ids()
    train_patients, test_patients = train_test_split(patients, test_size=1/10, random_state=42)
    train_patients, val_patients = train_test_split(train_patients, test_size=1/9, random_state=42)

    for patient_id in tqdm(val_patients):
        process_patient(patient_id, os.path.join(target_dir, "val"))
    for patient_id in tqdm(train_patients):
        process_patient(patient_id, os.path.join(target_dir, "train"))
    for patient_id in tqdm(test_patients):
        process_patient(patient_id, os.path.join(target_dir, "test"))

if __name__ == "__main__":
    main()

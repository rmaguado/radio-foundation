import os
import fnmatch
import pandas as pd
import numpy as np
import SimpleITK as sitk
import pydicom
import h5py
import json
from tqdm import tqdm

sitk.ProcessObject_SetGlobalWarningDisplay(False)

sex_gender = {
    'female': 'F',
    'male': 'M'
}

def get_image(dicom_directory):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_directory)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image

def check_is_ct(patient_folder_path):
    sample_dcm = [file for file in os.listdir(patient_folder_path) if file.endswith('.dcm')][0]
    dcm = pydicom.dcmread(os.path.join(patient_folder_path, sample_dcm))
    return dcm.Modality == "CT"

def get_further_metadata(patient_id, meta_df):
    global sex_gender
    patient_row = meta_df[meta_df["PatientID"] == patient_id].iloc[0]
    try:
        age = int(patient_row['age'])
    except Exception as e:
        age = 'NA'
    sex_value = patient_row['gender']
    if sex_value in sex_gender.keys():
        sex = sex_gender[sex_value]
    else:
        sex = 'NA'
    survival_time = int(patient_row['Survival.time'])
    
    return {
        'sex': sex,
        'age': age,
        'survival_time': survival_time
    }

def process_series(series_path, patient_id, series_id, meta_df):
    if not check_is_ct(series_path):
        return
    image = get_image(series_path)
    image_array = sitk.GetArrayViewFromImage(image)
    clip_hu = (np.clip(image_array, -1024, 3072) + 1024) / 4096
    
    img_spacing = image.GetSpacing()
    img_size = image.GetSize()
    
    metadata = {
        "dataset": "NSCLC-Radiomics",
        "series_id": series_id,
        "patient_id": patient_id,
        "spacing": [img_spacing[i] for i in [2,0,1]],
        "shape" : [img_size[i] for i in [2,0,1]],
    }
    
    metadata["other"] = get_further_metadata(patient_id, meta_df)
    
    return clip_hu, metadata

def get_patient_series(patient_folder_path):
    patient_series_paths = set()
    for dirpath, dirnames, filenames in os.walk(patient_folder_path):
        for filename in filenames:
            if fnmatch.fnmatch(filename, '*.dcm'):
                patient_series_paths.add(dirpath)
                break
    return list(patient_series_paths)

def save_data(series_target_dir, clip_hu, metadata):
    os.makedirs(series_target_dir, exist_ok=True)

    with h5py.File(os.path.join(series_target_dir, 'image.h5'), 'w') as f:
        f.create_dataset('data', data=clip_hu, compression='lzf', chunks=(1, 512, 512))

    with open(os.path.join(series_target_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)

def main():
    meta_df = pd.read_csv("./NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv")
    
    root = "/home/rmaguado/rdt/DeepRDT/manifest-1722177583547/NSCLC-Radiomics"
    target_dir = "./data"
    
    patient_ids = [x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))]
    series_number = 0
    for patient_id in tqdm(patient_ids):
        patient_folder_path = os.path.join(root, patient_id)
        patient_series_paths = get_patient_series(patient_folder_path)

        for series_path in patient_series_paths:
            series_id = f'series_{series_number:04d}'
            clip_hu, metadata = process_series(series_path, patient_id, series_id, meta_df)

            series_target_dir = os.path.join(target_dir, series_id)
            save_data(series_target_dir, clip_hu, metadata)

            series_number += 1
    

if __name__ == "__main__":
    main()

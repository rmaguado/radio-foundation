import fnmatch
import os
import pandas as pd
import numpy as np
import pydicom

from datasets import CtDataset


class NsclcRadiomics(CtDataset):
    _sex_keys = {
        'female': 'F',
        'male': 'M'
    }
    _datapath = "/home/rmaguado/rdt/DeepRDT/manifest-1722177583547/NSCLC-Radiomics"
    _dataframe = pd.read_csv("/home/rmaguado/ruben/radio-foundation/datasets/NSCLC-Radiomics/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv")
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_patient_ids(self):
        return [
            x for x in os.listdir(NsclcRadiomics._datapath) \
            if os.path.isdir(os.path.join(NsclcRadiomics._datapath, x))
        ]
        
    def get_patient_series_paths(self, patient_id):
        patient_folder_path = os.path.join(NsclcRadiomics._datapath, patient_id)
        patient_series_paths = set()
        for dirpath, dirnames, filenames in os.walk(patient_folder_path):
            for filename in filenames:
                # check that .dcm files exist in path
                if fnmatch.fnmatch(filename, '*.dcm'):
                    dcm = pydicom.dcmread(os.path.join(dirpath, filename))
                    # check that dicom are CTs
                    if dcm.Modality == "CT":
                        patient_series_paths.add(dirpath)
                    break
        patient_series_paths = list(patient_series_paths)
        return patient_series_paths


    def extend_metadata(self, metadata):
        patient_id = metadata["patient_id"]
        patient_row = NsclcRadiomics._dataframe[NsclcRadiomics._dataframe["PatientID"] == patient_id].iloc[0]
        try:
            age = int(patient_row['age'])
        except Exception as e:
            age = 'NA'
        sex_value = patient_row['gender']
        if sex_value in NsclcRadiomics._sex_keys.keys():
            sex = NsclcRadiomics._sex_keys[sex_value]
        else:
            sex = 'NA'
        survival_time = int(patient_row['Survival.time'])

        metadata["other"] = {
            'sex': sex,
            'age': age,
            'survival_time': survival_time
        }

def main():
    Dataprep = NsclcRadiomics(
        dataset="datasets/NSCLC-Radiomics",
        target_path="datasets/NSCLC-Radiomics/data",
    )
    Dataprep.prepare_dataset()

if __name__ == "__main__":
    main()

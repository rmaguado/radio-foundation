import fnmatch
import os
import pandas as pd
import numpy as np
import pydicom

from datasets import DatasetBase, SeriesProcessorBase


class NsclcRadiomics(DatasetBase):
    def __init__(self, config):
        super().__init__(config)
        
        self.datapath = config["datapath"]
        self.df = pd.read_csv(config["dfpath"])
        
    def get_patient_ids(self):
        return [
            x for x in os.listdir(self.datapath) \
            if os.path.isdir(os.path.join(self.datapath, x))
        ]
        
    def get_patient_series_paths(self, patient_id):
        patient_folder_path = os.path.join(self.datapath, patient_id)
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
        patient_row = self.df[self.df["PatientID"] == patient_id].iloc[0]
        
        age = patient_row['age']
        age = round(float(age)) if age != "NA" else age

        sex = {
            'female': 'F',
            'male': 'M'
        }[patient_row['gender']]
        
        survival_time = patient_row['Survival.time']

        metadata["other"] = {
            'sex': sex,
            'age': age,
            'survival_time': survival_time
        }

def main():
    config = {
        "dataset": "NSCLC-Radiomics",
        "datapath": "/home/rmaguado/rdt/DeepRDT/manifest-1722177583547/NSCLC-Radiomics",
        "dfpath": "/home/rmaguado/ruben/radio-foundation/datasets/NSCLC-Radiomics/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv",
        "target_path": "datasets/NSCLC-Radiomics/data",
        "z_spacing": 2.0,
        "clip_min": -1024,
        "clip_max": 3072,
        "chunk_size": (1, 512, 512)
    }
    
    dataprep = NsclcRadiomics(config)
    dataprep.prepare_dataset()

if __name__ == "__main__":
    main()

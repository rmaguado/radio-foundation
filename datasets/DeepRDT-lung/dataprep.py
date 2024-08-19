import fnmatch
import os
import pandas as pd
import numpy as np
import pydicom

from datasets import DatasetBase, SeriesProcessorBase


class DeepRDTlung(DatasetBase):
    def __init__(self, config):
        super().__init__(config)
        
        self.datapath = config["datapath"]

        path_to_scans = os.path.join(self.datapath, "lung")
        patient_folders = [
            path for path in os.listdir(path_to_scans) \
            if os.path.isdir(os.path.join(path_to_scans, path))
        ]
        patient_folders.sort()

        path_mapping = os.path.join(self.datapath, "mapping_df_lung_noDupli_wNewPatient2022.csv")
        path_metadata_1 = os.path.join(self.datapath, "DeepRDTPulmon_DATA_2024-07-17_1215.csv")
        path_metadata_2 = os.path.join(self.datapath, "CodigoPacientes.csv")
        
        metadata_1 = pd.read_csv(path_metadata_1, delimiter=";")
        metadata_1 = metadata_1.rename(columns={'nhc_sap': 'PID'})
        metadata_1['PID'] = metadata_1['PID'].astype(int)
        metadata_2 = pd.read_csv(path_metadata_2)
        metadata_2 = metadata_2.rename(columns={'Num. SAP': 'PID'})
        metadata_2['PID'] = metadata_2['PID'].astype(int)
        mapping = pd.read_csv(path_mapping)
        combined = pd.merge(metadata_1, mapping, on='PID', how='inner')
        combined = pd.merge(combined, metadata_2, on='PID', how='inner')

        self.df = combined[combined['MAPID'].astype(str).isin(patient_folders)]
        
    def get_patient_ids(self):
        patient_folders = list(self.df['MAPID'].astype(str))
        raise NotImplementedError
        
    def extend_metadata(self, metadata):
        raise NotImplementedError

def main():
    config = {
        "dataset": "DeepRDT-lung",
        "datapath": "/home/rmaguado/ruben/datasets/DeepRDT_lung/lung",
        "dfpath": "/home/rmaguado/ruben/datasets/DeepRDT_lung",
        "target_path": "datasets/DeepRDT-lung/data",
        "z_spacing": 2.0,
        "clip_min": -1024,
        "clip_max": 3072,
        "chunk_size": (1, 512, 512)
    }
    
    dataprep = NsclcRadiomics(config)
    dataprep.prepare_dataset()

if __name__ == "__main__":
    main()

import fnmatch
import os
import pandas as pd
import numpy as np
import pydicom
import SimpleITK as sitk

from datasets import DatasetBase, SeriesProcessorBase


class DeepRDTlung(DatasetBase):
    def __init__(self, config):
        super().__init__(config)
        
        self.datapath = config["datapath"]
        self.series_uids_path = config["series_uids_path"]

        series_uids_list = [
            x.split("_")[-1].split(".")[0] \
            for x in os.listdir(self.series_uids_path) \
            if x.endswith(".csv")
        ]

        path_mapping = os.path.join(
            self.datapath, "mapping_df_lung_noDupli_wNewPatient2022.csv"
        )
        path_metadata = os.path.join(
            self.datapath, "DeepRDTPulmon_DATA_2024-07-17_1215.csv"
        )

        metadata = pd.read_csv(path_metadata, delimiter=";")
        metadata = metadata.rename(columns={'nhc_sap': 'PID'})
        metadata['PID'] = metadata['PID'].astype(int)
        mapping = pd.read_csv(path_mapping)

        combined = pd.merge(metadata, mapping, on='PID', how='inner')
        self.df = combined[combined['MAPID'].astype(str).isin(series_uids_list)]
        
    def get_patient_ids(self):
        patient_folders = list(self.df['MAPID'].astype(str))
        patient_ids_paths = [
            (p_id, os.path.join(self.datapath, "lung", p_id)) \
            for p_id in patient_folders
        ]
        return patient_ids_paths
        
    def extend_metadata(self, metadata):
        patient_id = metadata["patient_id"]
        
        df_row = self.df[self.df['MAPID'].astype(str) == patient_id].iloc[0]
        
        sex = {1: 'M', 2: 'F'}.get(df_row["sexo"], "NaN")
        age = int(df_row.get("edad_diag", -1))
        
        tabaco = int(df_row.get("tabaco", -1))
        response = int(df_row.get("respuesta", -1))
        
        metadata["other"] = {
            "sex": sex,
            "age": age,
            "tabaco": tabaco,
            "response": response
        }
        
    def get_series_processor(self):
        return SeriesProcessor(self.config, self.statistics, self.df)
        
class SeriesProcessor(SeriesProcessorBase):
    def __init__(self, config, statistics_manager, df):
        super().__init__(config, statistics_manager)
        self.df = df
        self.series_uids_path = config["series_uids_path"]
        
    def get_image(self, path_to_series, patient_id):
        def get_series_uid(mapid):
            filename = f"ct_df_filtered_{mapid}.csv"
            path_to_df = os.path.join(self.series_uids_path, filename)
            df = pd.read_csv(path_to_df)
            return df["SeriesInstanceUID"].iloc[0]
        
        good_series_uid = get_series_uid(patient_id)
        
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            path_to_series, good_series_uid
        )
        
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(series_file_names)
        image = reader.Execute()
        return image

def main():
    config = {
        "dataset": "DeepRDT-lung",
        "datapath": "/home/rmaguado/ruben/datasets/DeepRDT_lung",
        "series_uids_path": "/home/rmaguado/cuda/AI/alba/deepRDT_lung/data/dfs",
        "target_path": "datasets/DeepRDT-lung/data",
        "z_spacing": 2.0,
        "clip_min": -1024,
        "clip_max": 3072,
        "chunk_size": (1, 512, 512)
    }
    
    dataprep = DeepRDTlung(config)
    dataprep.prepare_dataset()

if __name__ == "__main__":
    main()

import os
from os.path import basename, normpath
import pandas as pd
import SimpleITK as sitk
import pydicom
from tqdm import tqdm

from data import DatasetBase


class DeepRDTlung(DatasetBase):
    def __init__(self, config):
        super().__init__(config)

    def get_series_paths(self):

        df_path = self.config["df_path"]
        patient_ids_list = [
            x.split("_")[-1].split(".")[0]
            for x in os.listdir(df_path)
            if x.endswith(".csv")
        ]

        datapath = self.config["dataset_path"]
        series_paths = []
        reader = sitk.ImageSeriesReader()
        
        print("Walking dataset directories.")
        total_dirs = sum(len(dirs) for _, dirs, _ in os.walk(datapath))
        for data_folder, dirs, files in tqdm(os.walk(datapath), total=total_dirs):
            series_ids = reader.GetGDCMSeriesIDs(data_folder)
            if series_ids:
                patient_id = basename(normpath(data_folder))
                if patient_id not in patient_ids_list:
                    continue
                df = pd.read_csv(os.path.join(df_path, f"ct_df_filtered_{patient_id}.csv"))
                good_series_id = df['SeriesInstanceUID'].iloc[0]
                
            for series_id in series_ids:
                if series_id == good_series_id:
                    series_paths.append((series_id, data_folder))

        return series_paths


def main():
    config = {
        "dataset_name": "DeepRDT-lung",
        "dataset_path": "/home/rmaguado/ruben/datasets/DeepRDT_lung",
        "target_path": "data/radiomics_datasets.db",
        "df_path": "/home/rmaguado/cuda/AI/alba/deepRDT_lung/data/dfs",
    }

    dataprep = DeepRDTlung(config)
    dataprep.prepare_dataset()


if __name__ == "__main__":
    main()

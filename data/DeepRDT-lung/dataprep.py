import os
from os.path import basename, normpath
import pandas as pd
import SimpleITK as sitk
import pydicom
from tqdm import tqdm


class DeepRDTlung:
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
            if not series_ids:
                continue

            patient_id = basename(normpath(data_folder))
            if patient_id not in patient_ids_list:
                continue
            df = pd.read_csv(os.path.join(df_path, f"ct_df_filtered_{patient_id}.csv"))
            dosisplan_series_id = df["SeriesInstanceUID"].iloc[0]

            series_file_names = reader.GetGDCMSeriesFileNames(
                data_folder, dosisplan_series_id
            )
            if not series_file_names:
                continue

            first_file = series_file_names[0]
            dcm = pydicom.dcmread(first_file, stop_before_pixels=True)
            series_paths.append((dosisplan_series_id, data_folder))

        return series_paths

import os
import pandas as pd
import SimpleITK as sitk
import pydicom

from data import DatasetBase


class DeepRDTlung(DatasetBase):
    def __init__(self, config):
        super().__init__(config)

    def get_series_paths(self):

        series_ids_list = [
            x.split("_")[-1].split(".")[0]
            for x in os.listdir(self.config["series_ids_path"])
            if x.endswith(".csv")
        ]

        datapath = self.config["dataset_path"]
        series_paths = []
        reader = sitk.ImageSeriesReader()

        for data_folder, dirs, files in os.walk(datapath):
            series_ids = reader.GetGDCMSeriesIDs(data_folder)
            for series_id in series_ids:
                if series_id in series_ids_list:
                    series_paths.append((series_id, data_folder))

        return series_paths


def main():
    config = {
        "dataset_name": "DeepRDT-lung",
        "dataset_path": "/home/rmaguado/ruben/datasets/DeepRDT_lung",
        "target_path": "data/radiomics_datasets.db",
        "series_ids_path": "/home/rmaguado/cuda/AI/alba/deepRDT_lung/data/dfs",
    }

    dataprep = DeepRDTlung(config)
    dataprep.prepare_dataset()


if __name__ == "__main__":
    main()

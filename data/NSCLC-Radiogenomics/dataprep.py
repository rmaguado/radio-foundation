from pathlib import Path
import os
import pandas as pd
import pydicom
import SimpleITK as sitk
from tqdm import tqdm

from data import DatasetBase, validate_ct_dicom


def get_description(dicom_file_path: str) -> str:
    ds = pydicom.dcmread(dicom_file_path)
    description = ds.get("SeriesDescription", None)
    if description:
        return description.value
    return ""


class NsclcRadiogenomics(DatasetBase):
    def __init__(self, config):
        super().__init__(config)

    def get_series_paths(self):
        datapath = self.config["dataset_path"]
        series_paths = []
        reader = sitk.ImageSeriesReader()

        filter_words = ["thin lung window", "thorax", "chest", "lung", "in reach"]

        print("Walking dataset directories.")
        total_dirs = sum(len(dirs) for _, dirs, _ in os.walk(datapath))
        for data_folder, dirs, files in tqdm(os.walk(datapath), total=total_dirs):
            series_ids = reader.GetGDCMSeriesIDs(data_folder)
            for series_id in series_ids:
                series_file_names = reader.GetGDCMSeriesFileNames(
                    data_folder, series_id
                )
                if series_file_names:
                    first_file = series_file_names[0]
                    dcm = pydicom.dcmread(first_file)
                    modality = dcm.get("Modality", None)
                    if modality == "CT":
                        description = get_description(first_file)
                        if any(
                            [x in description.lower() for x in filter_words]
                        ) and validate_ct_dicom(first_file):
                            if series_id not in [x[0] for x in series_paths]:
                                series_paths.append((series_id, data_folder))

        return series_paths


def main():
    config = {
        "dataset_name": "NSCLC-Radiogenomics",
        "dataset_path": "/home/rmaguado/rdt/DeepRDT/datasets/NSCLC-Radiogenomics",
        "target_path": "data/radiomics_datasets.db",
    }

    dataprep = NsclcRadiogenomics(config)
    dataprep.prepare_dataset()


if __name__ == "__main__":
    main()

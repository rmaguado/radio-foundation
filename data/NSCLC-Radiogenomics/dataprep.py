from pathlib import Path
import os
import pandas as pd
import pydicom
import SimpleITK as sitk
from tqdm import tqdm

from data import DatasetBase


def get_series_data(dicom_file_path):
    try:
        ds = pydicom.dcmread(dicom_file_path)
        description = ds.get("SeriesDescription", "No Series Description")
        orientation = ds.get("ImageOrientationPatient", None)
        slice_thickness = ds.get("SliceThickness", None)
        return description, orientation, slice_thickness
    except Exception as e:
        return "", None, None


def is_axial_orientation(orientation):
    if orientation is None:
        return False
    axial_orientation = [1, 0, 0, 0, 1, 0]
    return all(abs(orientation[i] - axial_orientation[i]) < 0.01 for i in range(6))


def is_thin_slice(slice_thickness):
    try:
        return slice_thickness is not None and float(slice_thickness) < 5.0
    except ValueError:
        return False


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
                        description, orientation, slice_thickness = get_series_data(
                            first_file
                        )
                        if (
                            any([x in description.lower() for x in filter_words])
                            and is_axial_orientation(orientation)
                            and is_thin_slice(slice_thickness)
                        ):
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

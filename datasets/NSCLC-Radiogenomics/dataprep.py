from pathlib import Path
import os
import pandas as pd
import pydicom
import SimpleITK as sitk

from datasets import DatasetBase


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

        self.datapath = config["datapath"]
        self.df = pd.read_csv(config["dfpath"])
        self.other_headers.update(
            [
                ("sex", "TEXT"),
                ("age", "INT"),
                ("smoking_status", "TEXT"),
                ("pack_years", "INT"),
            ]
        )

    def get_series_paths(self):
        series_paths = []
        reader = sitk.ImageSeriesReader()

        filter_words = ["thin lung window", "thorax", "chest", "lung", "in reach"]

        for data_folder, dirs, files in os.walk(self.datapath):
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
                            series_paths.append((series_id, data_folder))

        return series_paths

    def extend_metadata(self, metadata):
        series_path = metadata["other"]["series_path"]
        patient_id = series_path.split("NSCLC-Radiomics/")[1].split("/")[0]
        patient_row = self.df[self.df["Case ID"] == patient_id].iloc[0]

        age = round(float(patient_row["Age at Histological Diagnosis"]))
        sex = {"Female": "F", "Male": "M"}[patient_row["Gender"]]
        smoking_status = patient_row["Smoking status"]

        pack_years = patient_row["Pack Years"]
        if pack_years == float("nan") or pack_years == "Not Collected":
            pack_years = "NA"

        metadata["other"].update(
            {
                "sex": sex,
                "age": age,
                "smoking_status": smoking_status,
                "pack_years": pack_years,
            }
        )


def main():
    config = {
        "dataset": "NSCLC-Radiogenomics",
        "datapath": "/home/rmaguado/ruben/radio-foundation/datasets/NSCLC-Radiogenomics/source",
        "dfpath": "/home/rmaguado/ruben/radio-foundation/datasets/NSCLC-Radiogenomics/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv",
        "database_path": "radiomics_datasets.db",
    }

    dataprep = NsclcRadiogenomics(config)
    dataprep.prepare_dataset()


if __name__ == "__main__":
    main()

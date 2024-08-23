from pathlib import Path
import os
import pandas as pd
import pydicom

from datasets import DatasetBase, SeriesProcessorBase


class NsclcRadiogenomics(DatasetBase):
    def __init__(self, config):
        super().__init__(config)

        self.datapath = config["datapath"]
        self.df = pd.read_csv(config["dfpath"])

    def get_patient_ids(self):
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
            # Axial images typically have an orientation close to [1, 0, 0, 0, 1, 0]
            if orientation is None:
                return False

            axial_orientation = [1, 0, 0, 0, 1, 0]
            return all(
                abs(orientation[i] - axial_orientation[i]) < 0.01 for i in range(6)
            )

        def is_thin_slice(slice_thickness):
            try:
                return slice_thickness is not None and float(slice_thickness) < 5.0
            except ValueError:
                return False

        filter_words = ["thin lung window", "thorax", "chest", "lung", "in reach"]
        patient_ids_series = []

        dicom_folders = dicom_folders = []
        for dirpath, dirnames, filenames in os.walk(self.datapath):
            dicom_files = [f for f in filenames if f.lower().endswith(".dcm")]

            if dicom_files:
                first_dicom_file = os.path.join(dirpath, dicom_files[0])
                description, orientation, slice_thickness = get_series_data(
                    first_dicom_file
                )
                if (
                    any([x in description.lower() for x in filter_words])
                    and is_axial_orientation(orientation)
                    and is_thin_slice(slice_thickness)
                ):

                    patient_id = Path(dirpath).parts[7]
                    if patient_id not in [p_id for p_id, _ in patient_ids_series]:
                        patient_ids_series.append((patient_id, dirpath))

        return patient_ids_series

    def extend_metadata(self, metadata):
        patient_id = metadata["patient_id"]
        patient_row = self.df[self.df["Case ID"] == patient_id].iloc[0]

        age = round(float(patient_row["Age at Histological Diagnosis"]))
        sex = {"Female": "F", "Male": "M"}[patient_row["Gender"]]
        smoking_status = patient_row["Smoking status"]

        pack_years = patient_row["Pack Years"]
        if pack_years == float("nan"):
            pack_years = "NA"

        metadata["other"] = {
            "sex": sex,
            "age": age,
            "smoking_status": smoking_status,
            "pack_years": pack_years,
        }


def main():
    config = {
        "dataset": "NSCLC-Radiogenomics",
        "datapath": "/home/rmaguado/ruben/radio-foundation/datasets/NSCLC-Radiogenomics/source",
        "dfpath": "/home/rmaguado/ruben/radio-foundation/datasets/NSCLC-Radiogenomics/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv",
        "target_path": "datasets/NSCLC-Radiogenomics/data",
        "z_spacing": 2.0,
        "clip_min": -1024,
        "clip_max": 3072,
        "chunk_size": (1, 512, 512),
    }

    dataprep = NsclcRadiogenomics(config)
    dataprep.prepare_dataset()


if __name__ == "__main__":
    main()

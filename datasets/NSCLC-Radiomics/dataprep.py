import os
import pandas as pd
import SimpleITK as sitk
import pydicom
from typing import List, Tuple

from datasets import DatasetBase


class NsclcRadiomics(DatasetBase):
    def __init__(self, config):
        super().__init__(config)

        self.datapath = config["datapath"]
        self.df = pd.read_csv(config["dfpath"])
        self.other_headers.update(
            [
                ("patient_id", "TEXT"),
                ("sex", "TEXT"),
                ("age", "INT"),
                ("survival_time", "INT"),
            ]
        )

    def get_series_paths(self) -> List[Tuple[str, str]]:
        series_paths = []
        reader = sitk.ImageSeriesReader()

        for data_folder, dirs, files in os.walk(self.datapath):
            series_ids = reader.GetGDCMSeriesIDs(data_folder)
            for series_id in series_ids:
                series_file_names = reader.GetGDCMSeriesFileNames(
                    data_folder, series_id
                )
                if series_file_names:
                    first_file = series_file_names[0]
                    dcm = pydicom.dcmread(first_file)
                    modality = dcm.GetMetaData("0008|0060")
                    if modality == "CT":
                        series_paths.append((series_id, data_folder))

        return series_paths

    def extend_metadata(self, metadata):
        series_path = metadata["other"]["series_path"]
        patient_id = series_path.split("NSCLC-Radiomics/")[1].split("/")[0]

        patient_row = self.df[self.df["PatientID"] == patient_id].iloc[0]

        age = patient_row["age"]
        age = round(float(age)) if age != "NA" else age

        sex = {"female": "F", "male": "M"}.get(patient_row["gender"], "U")

        survival_time = patient_row["Survival.time"]

        metadata["other"].update(
            {"sex": sex, "age": age, "survival_time": survival_time}
        )


def main():
    config = {
        "dataset": "NSCLC-Radiomics",
        "datapath": "/home/rmaguado/rdt/DeepRDT/manifest-1722177583547/NSCLC-Radiomics",
        "dfpath": "/home/rmaguado/ruben/radio-foundation/datasets/NSCLC-Radiomics/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv",
        "database_path": "radiomics_datasets.db",
    }

    dataprep = NsclcRadiomics(config)
    dataprep.prepare_dataset()


if __name__ == "__main__":
    main()

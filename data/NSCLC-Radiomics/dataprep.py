import os
import SimpleITK as sitk
import pydicom
from tqdm import tqdm
from typing import List, Tuple

from data import DatasetBase


class NsclcRadiomics(DatasetBase):
    def __init__(self, config):
        super().__init__(config)

    def get_series_paths(self) -> List[Tuple[str, str]]:
        datapath = self.config["dataset_path"]
        series_paths = []
        reader = sitk.ImageSeriesReader()
        for data_folder, dirs, files in tqdm(os.walk(datapath)):
            series_ids = reader.GetGDCMSeriesIDs(data_folder)
            for series_id in series_ids:
                series_file_names = reader.GetGDCMSeriesFileNames(
                    data_folder, series_id
                )
                if series_file_names:
                    first_file = series_file_names[0]
                    dcm = pydicom.dcmread(first_file)
                    modality = dcm.get((0x0008, 0x0060))
                    if modality is None:
                        break
                    if modality.value == "CT":
                        series_paths.append((series_id, data_folder))

        return series_paths


def main():
    config = {
        "dataset_name": "NSCLC-Radiomics",
        "dataset_path": "/home/rmaguado/rdt/DeepRDT/datasets/NSCLC-Radiomics",
        "target_path": "data/radiomics_datasets.db",
    }

    dataprep = NsclcRadiomics(config)
    dataprep.prepare_dataset()


if __name__ == "__main__":
    main()

import os
import numpy as np
import pylidc as pl
import SimpleITK as sitk
from tqdm import tqdm
from typing import List, Tuple

from data import DatasetBase, validate_ct_dicom


np.int = int
np.bool = bool


class LidcIdri(DatasetBase):
    def __init__(self, config):
        super().__init__(config)

    def get_series_paths(self) -> List[Tuple[str, str]]:
        print("Getting series paths...")
        scans = pl.query(pl.Scan).all()
        patient_ids = list(set(scan.patient_id for scan in scans))
        print(f"Number of patients: {len(patient_ids)}")
        patient_ids.sort()
        reader = sitk.ImageSeriesReader()

        series_paths = []
        for patient_id in tqdm(patient_ids):
            scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
            patient_series_paths = [scan.get_path_to_dicom_files() for scan in scans]

            for series_path in patient_series_paths:
                series_ids = reader.GetGDCMSeriesIDs(series_path)
                for series_id in series_ids:
                    series_file_names = reader.GetGDCMSeriesFileNames(
                        series_path, series_id
                    )
                    first_file = series_file_names[0]
                    if validate_ct_dicom(first_file):
                        series_paths.append((series_id, series_path))

        return series_paths


def main():
    config = {
        "dataset_name": "LIDC-IDRI",
        "dataset_path": "/home/rmaguado/rdt/DeepRDT/datasets/LIDC",
        "target_path": "data/radiomics_datasets.db",
    }

    dataprep = LidcIdri(config)
    dataprep.prepare_dataset()


if __name__ == "__main__":
    main()

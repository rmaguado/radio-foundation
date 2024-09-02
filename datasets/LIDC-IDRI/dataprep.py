import os
import numpy as np
import pylidc as pl
import SimpleITK as sitk
from typing import List, Tuple

from datasets import DatasetBase


np.int = int
np.bool = bool


class LidcIdri(DatasetBase):
    def __init__(self, config):
        super().__init__(config)

    def get_series_paths(self) -> List[Tuple[str, str]]:
        scans = pl.query(pl.Scan).all()
        patient_ids = list(set(scan.patient_id for scan in scans))
        patient_ids.sort()
        reader = sitk.ImageSeriesReader()

        series_paths = []
        for patient_id in patient_ids:
            scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
            patient_series_paths = [scan.get_path_to_dicom_files() for scan in scans]

            for series_path in patient_series_paths:
                series_ids = reader.GetGDCMSeriesIDs(series_path)
                for series_id in series_ids:
                    series_paths.append((series_id, series_path))

        return series_paths


def main():
    config = {
        "dataset_name": "LIDC-IDRI",
        "dataset_path": "/home/rmaguado/rdt/DeepRDT/datasets/LIDC-IDRI",
        "target_path": "datasets/radiomics_datasets.db",
    }

    dataprep = LidcIdri(config)
    dataprep.prepare_dataset()


if __name__ == "__main__":
    main()

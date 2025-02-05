"""
Indexing script for evaluation with LIDC-IDRI
"""

from typing import List, Tuple
import os
import pylidc as pl
import argparse
import logging

from data.evaluation.base import DicomEvalBase, DicomProcessorBase


logger = logging.getLogger("dataprep")


class LidcIdriProcessor(DicomProcessorBase):
    def __init__(self, config, database):
        super().__init__(config, database)

    def get_paths_and_ids(self) -> List[Tuple[str, str]]:
        """
        Get paths to dicom files and their corresponding scans IDs used by LIDC-IDRI.
        """
        paths_ids = []

        scans = pl.query(pl.Scan).all()
        patient_ids = list(set(scan.patient_id for scan in scans))
        patient_ids.sort()

        for patient_id in patient_ids:
            patient_scans = (
                pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
            )
            for scan in patient_scans:
                try:
                    scan_path = scan.get_path_to_dicom_files()
                    scan_id = scan.id

                    paths_ids.append((scan_path, scan_id))
                except Exception as e:
                    logger.exception(f"failed to get path for patient {patient_id}.")

        return paths_ids


def get_argpase():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--root_path", type=str, required=True, help="The root path of the dataset."
    )
    return parser


def main(args):
    dataset_name = "LIDC-IDRI"
    db_name = "LIDC-IDRI_eval"
    dataset_path = os.path.join(args.root_path, dataset_name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    db_dir = os.path.join("data/index", db_name)
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "index.db")

    config = {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "db_path": db_path,
    }
    database = DicomEvalBase(config)
    processor = LidcIdriProcessor(config, database)
    processor.prepare_dataset()


if __name__ == "__main__":
    parser = get_argpase()
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        logger.exception(e)
        raise e

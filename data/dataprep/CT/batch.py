import sqlite3
import argparse
import os

from .dicoms import DicomProcessor
from .niftis import NiftiProcessor

import logging
import warnings


logger = logging.getLogger("dataprep")


def get_argpase():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--root_path", type=str, required=True, help="The root path of the dataset."
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="data/database/radiomics_datasets.db",
        required=False,
        help="The path to the database.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Specify if only validation should be performed.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        help="The storage type for the dataset. Options are 'nifti' and 'dicom'.",
    )

    return parser


def get_processed_datasets(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    datasets = [table[0] for table in cursor.fetchall() if table[0] != "global"]

    conn.close()
    return datasets


def get_unprocessed_datasets(root_path, db_path):
    datasets_in_root = [
        x
        for x in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, x)) and not x.startswith(".")
    ]
    if not os.path.exists(db_path):
        return datasets_in_root

    processed_datasets = get_processed_datasets(db_path)
    if processed_datasets:
        logger.info(
            f"Skipping already processed datasets: {', '.join(processed_datasets)}"
        )
    unprocessed_datasets = [x for x in datasets_in_root if x not in processed_datasets]

    return unprocessed_datasets


def main(args):
    root_path = args.root_path
    db_path = args.db_path
    storage = args.storage

    unprocessed_datasets = get_unprocessed_datasets(root_path, db_path)
    logger.info(f"Unprocessed datasets: {', '.join(unprocessed_datasets)}")

    for dataset in unprocessed_datasets:
        dataset_path = os.path.join(root_path, dataset)
        config = {
            "dataset_name": dataset,
            "dataset_path": dataset_path,
            "target_path": db_path,
            "validate_only": False,
            "storage": storage,
        }
        if storage == "dicom":
            processor = DicomProcessor(config)
        elif storage == "nifti":
            processor = NiftiProcessor(config)
        processor.prepare_dataset()
        processor.close_db()


if __name__ == "__main__":
    args = get_argpase().parse_args()
    try:
        main(args)
    except Exception as e:
        logger.exception(e)
        raise e

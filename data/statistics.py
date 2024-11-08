"""
Samples the mean and standard deviation of an image dataset.
"""

import os
import argparse
import pydicom
import numpy as np
import sqlite3
from tqdm import tqdm
from typing import Tuple
import logging

from data.utils import set_logging


logger = logging.getLogger("dataprep")


def read_dicom_image(
    root_path: str,
    dataset_name: str,
    dcm_path: str,
    lower: float = -1000,
    upper: float = 1900,
) -> np.ndarray:
    try:
        abs_path = os.path.join(root_path, dataset_name, dcm_path)
        dcm = pydicom.dcmread(abs_path)
        rescale_slope = dcm.RescaleSlope
        rescale_intercept = dcm.RescaleIntercept
        array_data = dcm.pixel_array * rescale_slope + rescale_intercept
        array_data = np.clip(array_data, lower, upper)
        return array_data
    except Exception as e:
        logger.error(f"Error reading {dcm_path}: {e}")
        return None


def get_mean_std(pixel_array: np.ndarray) -> tuple:
    mean = np.mean(pixel_array)
    std = np.std(pixel_array)
    return mean, std


def sample_paths_dicom(conn, cursor, n: int) -> Tuple[float, flaot]:
    raise RuntimeError(
        "Outdated function. Needs to be rewritten for new database format."
    )

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    datasets = [table[0] for table in cursor.fetchall() if table[0] != "global"]
    logger.info(f"Sampling from datasets: {', '.join(datasets)}")

    n_dataset = n // len(datasets)

    paths = []
    for dataset in datasets:
        cursor.execute(
            f"SELECT dicom_path FROM global WHERE dataset = '{dataset}' ORDER BY RANDOM() LIMIT {n_dataset}"
        )
        dcm_paths = cursor.fetchall()
        paths += [(dataset, dcm_path[0]) for dcm_path in dcm_paths]

    means = []
    variances = []

    for dataset, dcm_path in tqdm(dcm_paths):
        pixel_array = read_dicom_image(root_path, dataset, dcm_path)
        if pixel_array is not None:
            mean, std = get_mean_std(pixel_array)
            means.append(mean)
            variances.append(std**2)

    mean = np.mean(means)
    std = np.mean(variances) ** 0.5

    return mean, std


def sample_paths_nifti(conn, cursor, n: int) -> Tuple[float, float]:
    raise NotImplementedError


def get_database_storate(conn, cursor):
    raise NotImplementedError


def get_argpase():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--root_path",
        type=str,
        required=True,
        help="Root path to the image dataset.",
    )
    parser.add_argument(
        "--db_name",
        type=str,
        default="dicom_datasets",
        required=False,
        help="The name of the database. Will look in data/index/<db_name>/index.db",
    )
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Number of samples to take from the dataset.",
    )
    return parser


def main(args):
    set_logging("data/log/statistics.log")
    root_path = args.root_path
    n = args.n

    db_path = os.path.join("data/index", args.db_name, "index.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Cannot find database at {db_path}.")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    storage_type = get_database_storate(conn, cursor)

    if storage_type == "dicom":
        mean, std = sample_paths_dicom(conn, cursor, n)
    elif storage_type == "nifti":
        mean, std = sample_paths_nifti(conn, cursor, n)

    conn.close()

    logger.info(f"Mean: {mean}")
    logger.info(f"Standard Deviation: {std}")


if __name__ == "__main__":
    parser = get_argpase()
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        logger.exception(e)
        raise e

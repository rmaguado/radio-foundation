"""
Samples the mean and standard deviation of an image dataset.
"""

import os
import argparse
import pydicom
import numpy as np
import sqlite3
from tqdm import tqdm
import logging
import warnings


logger = logging.getLogger("dataprep")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("data/log/statistics.log")
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

warnings.filterwarnings("ignore")


def get_image(
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


def sample_dcm_paths(db_path: str, n: int) -> list:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    datasets = [table[0] for table in cursor.fetchall() if table[0] != "global"]

    n_dataset = n // len(datasets)

    paths = []
    for dataset in datasets:
        cursor.execute(
            f"SELECT dicom_path FROM global WHERE dataset = '{dataset}' ORDER BY RANDOM() LIMIT {n_dataset}"
        )
        dcm_paths = cursor.fetchall()
        paths += [(dataset, dcm_path[0]) for dcm_path in dcm_paths]

    conn.close()
    return paths


def get_argpase():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--root_path",
        type=str,
        required=True,
        help="Root path to the image dataset.",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        required=True,
        help="Path to the SQLite database.",
    )
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Number of samples to take from the dataset.",
    )
    return parser


def main(args):
    root_path = args.root_path
    db_path = args.db_path
    n = args.n

    dcm_paths = sample_dcm_paths(db_path, n)
    means = []
    variances = []

    for dataset, dcm_path in tqdm(dcm_paths):
        pixel_array = get_image(root_path, dataset, dcm_path)
        if pixel_array is not None:
            mean, std = get_mean_std(pixel_array)
            means.append(mean)
            variances.append(std**2)

    mean = np.mean(means)
    std = np.mean(variances) ** 0.5

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

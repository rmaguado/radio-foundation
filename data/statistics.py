"""
Samples the mean and standard deviation of an image dataset.
"""

import os
import argparse
import pydicom
import nibabel as nib
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
        logger.exception(f"Error reading {abs_path}: {e}")
    return None


def read_nifti_image(
    root_path: str,
    dataset_name: str,
    nii_path: str,
    lower: float = -1000,
    upper: float = 1900,
) -> np.ndarray:
    try:
        abs_path = os.path.join(root_path, dataset_name, nii_path)
        nii = nib.load(abs_path)
        array_data = nii.get_fdata()

        rescale_slope = nii.dataobj.slope
        rescale_intercept = nii.dataobj.inter

        if np.isnan(rescale_slope):
            rescale_slope = 1.0
        if np.isnan(rescale_intercept):
            rescale_intercept = 0.0

        array_data = array_data * rescale_slope + rescale_intercept

        array_data = np.clip(array_data, lower, upper)
        return array_data
    except Exception as e:
        logger.exception(f"Error reading {abs_path}: {e}")
    return None


def get_mean_std(pixel_array: np.ndarray) -> tuple:
    mean = np.mean(pixel_array)
    std = np.std(pixel_array)
    return mean, std


def sample_paths_dicom(cursor, root_path: str, n: int) -> Tuple[float, float]:
    cursor.execute("SELECT dataset, dcm_path FROM datasets;")
    datasets = cursor.fetchall()

    if len(datasets) == 0:
        raise ValueError("No datasets found in the database.")

    samples_per_dataset = n // len(datasets)

    dcm_paths = []
    for dataset_name in datasets:
        cursor.execute(
            f"""
            SELECT dicom_path FROM global WHERE dataset = '{dataset_name}'
            ORDER BY RANDOM()
            LIMIT {samples_per_dataset};
            """
        )
        dcm_paths.extend([(dataset_name, dcm_path) for dcm_path in cursor.fetchall()])

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


def sample_paths_nifti(cursor, root_path: str, n: int) -> Tuple[float, float]:
    cursor.execute("SELECT dataset, dcm_path FROM datasets;")
    datasets = cursor.fetchall()

    if len(datasets) == 0:
        raise ValueError("No datasets found in the database.")

    samples_per_dataset = n // len(datasets)

    nii_paths = []
    for dataset_name in datasets:
        cursor.execute(
            f"""
            SELECT nifti_path FROM global WHERE dataset = '{dataset_name}'
            ORDER BY RANDOM()
            LIMIT {samples_per_dataset};
            """
        )
        nii_paths.extend([(dataset_name, nii_path) for nii_path in cursor.fetchall()])

    means = []
    variances = []
    for dataset, nii_path in tqdm(nii_paths):
        pixel_array = read_nifti_image(root_path, dataset, nii_path)
        if pixel_array is not None:
            mean, std = get_mean_std(pixel_array)
            means.append(mean)
            variances.append(std**2)

    mean = np.mean(means)
    std = np.mean(variances) ** 0.5

    return mean, std


def get_database_storate(cursor):
    cursor.execute("SELECT value FROM metadata WHERE key = 'storage';")
    storage_type = cursor.fetchone()[0]

    if storage_type not in ["dicom", "nifti"]:
        raise ValueError(f"Unknown storage type: {storage_type}")

    return storage_type


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

    storage_type = get_database_storate(cursor)

    if storage_type == "dicom":
        mean, std = sample_paths_dicom(cursor, root_path, n)
    elif storage_type == "nifti":
        mean, std = sample_paths_nifti(cursor, root_path, n)

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

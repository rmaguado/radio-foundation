import argparse
import os

import numpy as np
import pandas as pd
import nibabel as nib

from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import logging


def process_row(row_args):
    row, args = row_args

    VolumeName = row["VolumeName"]
    dir1 = VolumeName.rsplit("_", 1)[0]
    dir2 = VolumeName.rsplit("_", 2)[0]
    foldername = "CT-RATE_valid" if args.dataset == "validation" else "CT-RATE_train"
    filepath = os.path.join(args.root_path, foldername, dir2, dir1, VolumeName)

    try:
        img_nib = nib.load(filepath, mmap=True)
        data = img_nib.get_fdata()
        affine = img_nib.affine
        header = img_nib.header
    except Exception as e:
        logging.exception(f"Error reading {filepath}: {e}")
        return

    if b"fixed" in header["descrip"]:
        logging.info(f"Already fixed {filepath}. Skipping.")
        return

    try:
        RescaleIntercept = float(row["RescaleIntercept"])
        RescaleSlope = float(row["RescaleSlope"])
        adjusted_data = data * RescaleSlope + RescaleIntercept
        adjusted_data = adjusted_data.astype(np.int16)
    except Exception as e:
        logging.exception(f"Error fixing Rescale for {filepath}: {e}")
        return

    try:
        header["descrip"] = b"fixed"

        dirpath = os.path.dirname(filepath)
        Path(dirpath).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(dirpath, os.path.basename(filepath))

        new_img_nib = nib.Nifti1Image(adjusted_data, affine, header=header)
        new_img_nib.to_filename(output_path)
    except Exception as e:
        logging.exception(f"Error writing {output_path}: {e}")
        return

    logging.info(f"Fixed {filepath} to {output_path}")


def main(args):

    metadata = pd.read_csv(
        os.path.join(args.root_path, f"metadata/{args.dataset}_metadata.csv")
    )

    rows = metadata.to_dict(orient="records")

    num_workers = max(1, cpu_count() - 2)

    with Pool(num_workers) as pool:
        list(
            tqdm(
                pool.imap_unordered(process_row, [(row, args) for row in rows]),
                total=len(rows),
            )
        )


def processed_missed_rows(args, missed_rows=[]):
    metadata = pd.read_csv(
        os.path.join(args.root_path, f"metadata/{args.dataset}_metadata.csv")
    )

    metadata = metadata[metadata["VolumeName"].isin(missed_rows)]

    rows = metadata.to_dict(orient="records")

    for row in rows:
        process_row((row, args))


def get_args():
    parser = argparse.ArgumentParser(
        description="Process a part of a DataFrame.", add_help=True
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="/path/to/CT-RATE",
        help="The root directory of the data.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="train",
        help="train or validation",
    )
    return parser.parse_args()


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename="fix_metadata.log",
    )

    args = get_args()
    main(args)

import os
from typing import Dict
import argparse
import logging
import nibabel as nib
from tqdm import tqdm
import numpy as np

from ..utils import walk, set_logging
from .ct_database import CtDatabase


logger = logging.getLogger("dataprep")


class NiftiCtValidation:
    def __init__(self, config: dict):
        self.skip_validation = config["skip_validation"]

    def test_image_shape(self, metadata: Dict) -> str:
        rows = metadata["rows"]
        columns = metadata["columns"]
        if rows != columns:
            return f"Rows and columns are not equal: {rows} x {columns}\n"

        return ""

    def test_slice_thickness(self, metadata: Dict) -> str:
        slice_thickness = metadata["slice_thickness"]
        if slice_thickness > 4.0:
            return f"\tSlice thickness is too high: ({slice_thickness}).\n"
        return ""

    def test_spacing(self, metadata: Dict) -> str:
        spacing_x = metadata["spacing_x"]
        spacing_y = metadata["spacing_y"]
        slice_thickness = metadata["slice_thickness"]
        if spacing_x > slice_thickness or spacing_y > slice_thickness:
            return f"\tSpacing is greater than slice thickness: ({spacing_x}, {spacing_y}).\n"
        return ""

    def test_rescale(self, header: nib.Nifti1Header) -> str:
        slope = header.get("scl_slope")
        intercept = header.get("scl_inter")

        if np.isnan(slope) or np.isnan(intercept):
            return "\tRescale slope or intercept is NaN.\n"
        if slope == 0:
            return "\tRescale slope is 0.\n"
        if slope is None or intercept is None:
            return "\tRescale slope or intercept is None.\n"
        return ""

    def __call__(self, nifti_file: nib.Nifti1Image, metadata: Dict):
        if self.skip_validation:
            return
        self.validate_nifti_for_ct(nifti_file, metadata)

    def validate_nifti_for_ct(self, nifti_file: nib.Nifti1Image, metadata: Dict):
        """
        Validates a NIfTI file for CT scans.
        Args:
            metadata (Dict): Metadata extracted from the NIfTI file.
        """
        volume_data = nifti_file.get_fdata()
        header = nifti_file.header

        issues = ""
        issues += self.test_slice_thickness(metadata)
        issues += self.test_image_shape(metadata)
        issues += self.test_spacing(metadata)
        issues += self.test_rescale(header)

        assert len(issues) == 0, issues


class NiftiDatabase(CtDatabase):
    def __init__(self, config):
        super().__init__(config, storage="nifti")

        self.create_global_table()
        self.add_dataset(config["dataset_name"])

    def create_global_table(self) -> None:
        """
        Creates a global table in the database for NIfTI files.
        """
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS global (
                dataset TEXT,
                num_slices INTEGER,
                slice_thickness REAL,
                spacing_x REAL,
                spacing_y REAL,
                axial_dim INTEGER,
                nifti_path TEXT
            )
            """
        )

    def insert_global_data(
        self, dataset_name: str, metadata: dict, nifti_path: str
    ) -> None:
        self.cursor.execute(
            """
            INSERT INTO "global" (dataset, num_slices, slice_thickness, spacing_x, spacing_y, axial_dim, nifti_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_name,
                metadata["num_slices"],
                metadata["slice_thickness"],
                metadata["spacing_x"],
                metadata["spacing_y"],
                metadata["axial_dim"],
                nifti_path,
            ),
        )


class NiftiProcessor:
    def __init__(self, config: dict):
        self.dataset_name = config["dataset_name"]
        self.absolute_dataset_path = os.path.abspath(config["dataset_path"])
        self.validate_only = config["validate_only"]

        self.validator = NiftiCtValidation(config)
        if not self.validate_only:
            self.database = NiftiDatabase(config)

        log_path = f"data/log/{self.dataset_name}.log"
        set_logging(log_path)

    def get_metadata(self, nifti_file: nib.Nifti1Image) -> dict:
        """
        Extracts metadata from a NIfTI file, including information on which dimension is axial.

        Args:
            nifti_file (nib.Nifti1Image): The NIfTI image object.

        Returns:
            dict: A dictionary with metadata including shape, voxel sizes, and axial orientation dimension.
        """
        header = nifti_file.header
        shape = header.get_data_shape()
        voxel_sizes = header.get_zooms()

        affine = nifti_file.affine
        abs_affine = np.abs(affine[:3, :3])
        # The axial dimension corresponding will have the highest value in the 3rd row of affine
        axial_dim = np.argmax(abs_affine[:, 2])

        rows_columns_dims = [0, 1, 2]
        rows_columns_dims.remove(axial_dim)

        rows = shape[rows_columns_dims[0]]
        columns = shape[rows_columns_dims[1]]

        spacing_x = voxel_sizes[rows_columns_dims[0]]
        spacing_y = voxel_sizes[rows_columns_dims[1]]

        return {
            "num_slices": int(shape[axial_dim]),
            "rows": int(rows),
            "columns": int(columns),
            "slice_thickness": float(voxel_sizes[axial_dim]),
            "spacing_x": float(spacing_x),
            "spacing_y": float(spacing_y),
            "axial_dim": int(axial_dim),
        }

    def process_volume(self, nifti_path: str) -> None:
        """
        Process a NIfTI volume, extract metadata, and store it in the database.

        Args:
            nifti_path (str): Path to the NIfTI file.
        """
        nifti_file = nib.load(nifti_path)
        try:
            metadata = self.get_metadata(nifti_file)
        except Exception as e:
            logger.error(f"Metadata extraction failed for {nifti_path}: {e}")
            return

        try:
            self.validator(nifti_file, metadata)
        except Exception as e:
            logger.error(f"Validation failed for {nifti_path}: {e}")
            return

        if self.validate_only:
            return

        rel_nifti_path = os.path.relpath(nifti_path, self.absolute_dataset_path)
        self.database.insert_global_data(self.dataset_name, metadata, rel_nifti_path)

        logger.info(f"Processed nifti: {nifti_path}.")

    def prepare_dataset(self) -> None:
        """
        Prepares the dataset by scanning the directories and processing NIfTI files.
        """

        logger.info(f"Processing dataset: {self.dataset_name}")
        print("Walking dataset directories.")
        total_dirs = sum(1 for _ in walk(self.absolute_dataset_path))
        logger.info(f"{self.dataset_name} total directories: {total_dirs}")

        for data_folder, dirs, files in tqdm(
            walk(self.absolute_dataset_path), total=total_dirs
        ):
            nii_paths = [
                os.path.join(data_folder, f)
                for f in files
                if f.endswith(".nii.gz") or f.endswith(".nii")
            ]
            if not nii_paths:
                continue

            for nii_path in nii_paths:

                if not self.validate_only:
                    try:
                        self.process_volume(nii_path)
                    except Exception as e:
                        logger.exception(f"Error processing nifti {nii_path}: {e}")
                        continue

        logger.info(f"Finished processing {self.dataset_name}. ")

    def close_db(self) -> None:
        """
        Closes the database connection.
        """
        self.database.close()


def get_argparse():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--root_path", type=str, required=True, help="The root path of the dataset."
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="The name of the dataset."
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Specify if only validation should be performed.",
    )
    parser.add_argument(
        "--db_name",
        type=str,
        default="nifti_datasets",
        required=False,
        help="The name of the database. Will be saved in data/index/<db_name>/index.db",
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Specify if validation should be skipped.",
    )
    return parser


def main(args):
    dataset_path = os.path.join(args.root_path, args.dataset_name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    db_dir = os.path.join("data/index", args.db_name)
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "index.db")

    config = {
        "dataset_name": args.dataset_name,
        "dataset_path": dataset_path,
        "db_path": db_path,
        "validate_only": args.validate_only,
        "skip_validation": args.skip_validation,
    }
    processor = NiftiProcessor(config)
    processor.prepare_dataset()


if __name__ == "__main__":
    parser = get_argparse()
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        logger.exception(e)
        raise e

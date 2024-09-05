from typing import List, Dict, Tuple, Any

import sqlite3
import argparse
from tqdm import tqdm
import pydicom
import ast
import os

import logging


logger = logging.getLogger("dataprep")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("data/log.txt")
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


class CtValidation:
    def __init__(self, config: dict):
        self.config = config
        self.derived_okay = config["derived_okay"]

        self.required_fields = [
            "SeriesInstanceUID",
            "Modality",
            "ImageOrientationPatient",
            "ImagePositionPatient",
            "SliceThickness",
            "PixelSpacing",
            "Rows",
            "Columns",
            "ImageType",
            "RescaleSlope",
            "RescaleIntercept",
        ]

    def get_fields(
        self, dcm: pydicom.dataset.FileDataset
    ) -> Tuple[Dict[str, Any], List[str]]:
        fields = {field: dcm.get(field, None) for field in self.required_fields}
        missing_fields = [field for field, value in fields.items() if value is None]
        return fields, missing_fields

    def is_axial_orientation(self, orientation: list):
        axial_orientation = [1, 0, 0, 0, 1, 0]
        return all(abs(orientation[i] - axial_orientation[i]) < 0.01 for i in range(6))

    def test_modality(self, fields: Dict[str, Any]) -> str:
        modality = fields["Modality"]
        if modality != "CT":
            return f"\tModality is not CT: ({modality}).\n"
        return ""

    def test_orientation(self, fields: Dict[str, Any]) -> str:
        orientation = fields["ImageOrientationPatient"]
        if not self.is_axial_orientation(orientation):
            return f"\tOrientation is not axial: ({orientation}).\n"
        return ""

    def test_slice_thickness(self, fields: Dict[str, Any]) -> str:
        slice_thickness = float(fields["SliceThickness"])
        if not slice_thickness <= 4.0:
            return f"\tSlice thickness is too high: ({slice_thickness}).\n"
        return ""

    def test_image_type(self, fields: Dict[str, Any]) -> str:
        image_type = fields["ImageType"]
        if isinstance(image_type, str):
            image_type = ast.literal_eval(image_type)
        if len(image_type) < 2:
            return f"\tImage type is too short: ({image_type}).\n"
        else:
            if image_type[0] != "ORIGINAL" and not self.derived_okay:
                return f"\tImage type is not ORIGINAL: ({image_type}).\n"
            if image_type[1] != "PRIMARY":
                return f"\tImage type is not PRIMARY: ({image_type[1]}).\n"
        if len(image_type) > 2:
            if image_type[2] == "LOCALIZER":
                return f"\tImage type is LOCALIZER.\n"
        return ""

    def __call__(
        self, dcm: pydicom.dataset.FileDataset, dicom_folder_path: str
    ) -> bool:
        """
        Validates a DICOM file for CT scans.

        Args:
            dcm: The DICOM object to validate.
            dicom_folder_path: The folder path of the DICOM file.

        Returns:
            bool: True if the DICOM file is valid for CT scans, False otherwise.
        """
        fields, missing_fields = self.get_fields(dcm)
        if missing_fields:
            logger.info(
                f"{dicom_folder_path}: Missing fields: {', '.join(missing_fields)}. Skipping."
            )
            return False

        issues = ""
        issues += self.test_modality(fields)
        issues += self.test_orientation(fields)
        issues += self.test_slice_thickness(fields)
        issues += self.test_image_type(fields)

        if issues:
            logger.info(f"Skipping {dicom_folder_path}:\n{issues}")
            return False
        return True


def walk(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "ignore" in filenames:
            dirnames[:] = []

        yield dirpath, dirnames, filenames


class Database:
    def __init__(self, config: dict):
        self.conn = sqlite3.connect(config["target_path"])
        self.cursor = self.conn.cursor()

        self.dataset_name_str = f'"{config["dataset_name"]}"'

        self.create_global_table()
        self.create_dataset_table()

    def create_global_table(self) -> None:
        """
        Creates a global table in the database if it doesn't already exist.
        """
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS global (
                dataset TEXT,
                series_id TEXT,
                slice_index INT,
                dicom_path TEXT,
                PRIMARY KEY (dataset, series_id, slice_index)
            )
            """
        )

    def create_dataset_table(self) -> None:
        """
        Create a dataset table in the database.
        """
        self.cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.dataset_name_str} (
                series_id TEXT PRIMARY KEY,
                num_slices INTEGER,
                image_shape_x INTEGER,
                image_shape_y INTEGER,
                slice_thickness REAL,
                spacing_x REAL,
                spacing_y REAL,
            )
            """
        )

    def insert_global_data(
        self,
        dataset_name: str,
        series_id: str,
        slice_index: int,
        dicom_path: str,
    ) -> None:
        """
        Insert global data into the database.

        Parameters:
            dataset_name (str): The name of the dataset.
            series_id (str): The ID of the series.
            slice_index (int): The index of the slice.
            dicom_path (str): The path to the DICOM file.

        Returns:
            None
        """
        self.cursor.execute(
            """
            INSERT INTO "global" (
                dataset, series_id, slice_index, dicom_path
            )
            VALUES (?, ?, ?, ?)
            """,
            (dataset_name, series_id, slice_index, dicom_path),
        )

    def insert_dataset_data(self, metadata: Tuple) -> None:
        """
        Insert dataset data into the specified table.

        Args:
            metadata (dict): The metadata dictionary containing information about the dataset.
        """
        self.cursor.execute(
            f"""
            INSERT INTO {self.dataset_name_str} (
                series_id, num_slices, image_shape_x, image_shape_y,
                slice_thickness, spacing_x, spacing_y
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            metadata,
        )

    def close(self):
        """
        Closes the database connection.
        """
        self.conn.commit()
        self.conn.close()

    def __del__(self):
        self.close()


class Processor:
    def __init__(self, config: dict):
        self.dataset_name = config["dataset_name"]
        self.absolute_dataset_path = os.path.abspath(config["dataset_path"])
        self.validate_only = config["validate_only"]

        self.ct_validator = CtValidation(config)
        if not self.validate_only:
            self.database = Database(config)

    def get_shape(self, dcm: pydicom.dataset.FileDataset) -> Tuple[int, int]:
        """
        Returns the shape of the image array.

        Parameters:
            dcm (pydicom.dataset.FileDataset): The dicom object.

        Returns:
            Tuple[int, int]: A tuple containing the shape of the image array.
        """
        return dcm.Rows, dcm.Columns

    def get_spacing(
        self, dcm: pydicom.dataset.FileDataset
    ) -> Tuple[float, float, float]:
        """
        Get the spacing of the given image.

        Args:
            dcm (pydicom.dataset.FileDataset): The input image.

        Returns:
            Tuple[float, float, float]: The spacing of the images and and slice thickness of the series.
        """
        return *dcm.PixelSpacing, dcm.SliceThickness

    def load_series(
        self, dicom_paths: List[str]
    ) -> List[Tuple[str, pydicom.dataset.FileDataset]]:
        """
        Loads the dicoms in the series and sorts them by position in patient.

        Args:
            dicom_paths (List[str]): A list of paths to the dicom files.

        Returns:
            List[Tuple[str, pydicom.dataset.FileDataset]]: A list of tuples containing the file path and the dicom object.
        """
        dicoms = [(path, pydicom.dcmread(path)) for path in dicom_paths]
        dicoms.sort(key=lambda x: x[1].ImagePositionPatient[2])
        return dicoms

    def process_series(self, dicom_paths: List[str], series_id: str) -> None:
        """
        Process a series and return metadata and paths to dicom files.

        Args:
            dicom_paths (List[str]): A list of paths to the dicom files.
            series_id (str): Series ID.
        """
        sorted_dicoms = self.load_series(dicom_paths)
        first_dicom = sorted_dicoms[0][1]
        num_slices = len(sorted_dicoms)
        image_shape = self.get_shape(first_dicom)
        (spacing_x, spacing_y), slice_thickness = self.get_spacing(first_dicom)

        metadata = (
            series_id,
            num_slices,
            image_shape[0],
            image_shape[1],
            slice_thickness,
            spacing_x,
            spacing_y,
        )

        for slice_index, (dicom_path, dicom) in enumerate(sorted_dicoms):
            rel_dicom_path = os.path.relpath(dicom_path, self.absolute_dataset_path)
            self.insert_global_data(
                self.dataset_name, series_id, slice_index, rel_dicom_path
            )

        self.insert_dataset_data(metadata)

    def prepare_dataset(self) -> None:
        """
        Prepares the dataset by creating necessary tables in the database and inserting data.
        """
        included_series_ids = []

        print("Walking dataset directories.")
        total_dirs = sum(len(dirs) for _, dirs, _ in walk(self.absolute_dataset_path))
        print("Total dicom directories: ", total_dirs)
        for data_folder, dirs, files in tqdm(
            walk(self.absolute_dataset_path), total=total_dirs
        ):
            dcm_paths = [
                os.path.join(data_folder, f) for f in files if f.endswith(".dcm")
            ]
            if not dcm_paths:
                continue

            first_dcm = pydicom.dcmread(dcm_paths[0], stop_before_pixels=True)
            if not self.ct_validator(first_dcm, data_folder):
                continue
            series_id = first_dcm.get("SeriesInstanceUID", None)

            if series_id in included_series_ids:
                logger.info(f"{data_folder}: Series ID already in database. Skipping.")
                continue

            if self.validate_only:
                continue
            self.process_series(dcm_paths, series_id)
            included_series_ids.append(series_id)


def get_argpase():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--root_path", type=str, required=True, help="The root path of the dataset."
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="The name of the dataset."
    )
    parser.add_argument(
        "--derived_okay",
        action="store_true",
        help="Specify if derived images are allowed.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Specify if only validation should be performed.",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="data/radiomics_datasets.db",
        required=False,
        help="The path to the database.",
    )
    return parser


def main(args):
    dataset_path = os.path.join(args.root_path, args.dataset_name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    config = {
        "dataset_name": args.dataset_name,
        "dataset_path": dataset_path,
        "target_path": args.db_path,
        "derived_okay": args.derived_okay,
        "validate_only": args.validate_only,
    }
    processor = Processor(config)
    processor.prepare_dataset()


if __name__ == "__main__":
    parser = get_argpase()
    args = parser.parse_args()
    main(args)

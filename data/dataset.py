import sqlite3
import argparse
from tqdm import tqdm
import SimpleITK as sitk
from tqdm import tqdm
import pydicom
from typing import List, Dict, Tuple, Any
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
            "SliceThickness",
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
        
    def __call__(self, dcm: pydicom.dataset.FileDataset, dicom_folder_path: str) -> bool:
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


class DatasetBase:
    def __init__(self, config: dict):
        sitk.ProcessObject_SetGlobalWarningDisplay(False)
        self.config = config
        self.ct_validator = CtValidation(config)

    def get_series_paths(self) -> List[Tuple[str, str]]:
        """
        Return a list of tuples with series_id and path to series.

        Returns:
            List[Tuple[str, str]]: A list of tuples with series_id and path to series.
        """
        datapath = self.config["dataset_path"]
        series_paths = []

        print("Walking dataset directories.")
        total_dirs = sum(len(dirs) for _, dirs, _ in walk(datapath))
        print("Total dicom directories: ", total_dirs)
        for data_folder, dirs, files in tqdm(walk(datapath), total=total_dirs):

            dcm_files = [f for f in files if f.endswith(".dcm")]
            if not dcm_files:
                continue

            first_dcm = os.path.join(data_folder, dcm_files[0])
            dcm = pydicom.dcmread(first_dcm, stop_before_pixels=True)

            if not self.ct_validator(dcm, data_folder):
                continue

            series_id = dcm.get("SeriesInstanceUID", None)

            if series_id in [s[0] for s in series_paths]:
                logger.info(f"{data_folder}: Series ID already in list. Skipping.")
                continue
            series_paths.append((series_id, data_folder))
        
        logger.info(f"Found {len(series_paths)} validated series.")

        return series_paths

    def get_spacing(self, image: sitk.Image) -> List[float]:
        """
        Get the spacing of the given image.

        Parameters:
            image (sitk.Image): The input image.

        Returns:
            List[float]: The spacing of the image in the order [z, x, y].
        """
        spacing = image.GetSpacing()
        return [spacing[i] for i in [2, 0, 1]]

    def get_shape(self, image: sitk.Image) -> Tuple[Tuple[int, int], int]:
        """
        Returns the shape of the image array and the number of slices.

        Parameters:
            image (sitk.Image): The input image.

        Returns:
            Tuple[Tuple[int, int], int]: A tuple containing the shape of the image array and the number of slices.
        """
        image_array = sitk.GetArrayFromImage(image)
        return image_array.shape[1:], image_array.shape[0]

    def load_image(self, dicom_paths: List[str]) -> sitk.Image:
        """
        Loads and returns a SimpleITK image from a list of DICOM file paths.

        Args:
            dicom_paths (List[str]): A list of file paths to DICOM files.

        Returns:
            sitk.Image: The loaded SimpleITK image.

        """

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dicom_paths)
        image = reader.Execute()

        return image

    def process_series(
        self, path_to_series: str, series_id: str
    ) -> Tuple[dict, List[str]]:
        """
        Process a series and return metadata and paths to dicom files.

        Args:
            path_to_series (str): Path to the series.
            series_id (str): Series ID.

        Returns:
            Tuple[dict, List[str]]: Metadata and paths to dicom files.
        """
        dicom_paths = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            path_to_series, series_id
        )
        image = self.load_image(dicom_paths)
        image_spacing = self.get_spacing(image)
        image_shape, num_slices = self.get_shape(image)

        metadata = {
            "num_slices": num_slices,
            "image_shape": image_shape,
            "spacing": image_spacing,
        }

        return metadata, dicom_paths

    def prepare_dataset(self) -> None:
        """
        Prepares the dataset by creating necessary tables in the database and inserting data.
        """
        dataset_name = self.config["dataset_name"]
        absolute_dataset_path = os.path.abspath(self.config["dataset_path"])

        conn = sqlite3.connect(self.config["target_path"])
        cursor = conn.cursor()

        self.create_global_table(cursor)
        self.create_dataset_table(cursor, dataset_name)

        series_paths = self.get_series_paths()
        logger.info(f"Processing {len(series_paths)} series.")
        for series_id, series_path in tqdm(series_paths):
            print(series_id)
            print(series_path)
            metadata, dicom_paths = self.process_series(series_path, series_id)

            for slice_index, dicom_path in enumerate(dicom_paths):
                relative_path = os.path.relpath(dicom_path, absolute_dataset_path)
                self.insert_global_data(
                    cursor, dataset_name, series_id, slice_index, relative_path
                )
            self.insert_dataset_data(cursor, dataset_name, series_id, metadata)

        conn.commit()
        conn.close()

    def create_global_table(self, cursor: sqlite3.Cursor) -> None:
        """
        Creates a global table in the database if it doesn't already exist.

        Parameters:
            cursor (sqlite3.Cursor): The cursor object used to execute SQL statements.

        Returns:
            None
        """
        cursor.execute(
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

    def create_dataset_table(self, cursor: sqlite3.Cursor, dataset_name: str) -> None:
        """
        Create a dataset table in the database.

        Args:
            cursor (sqlite3.Cursor): The cursor object to execute SQL statements.
            dataset_name (str): The name of the dataset.

        Returns:
            None
        """
        dataset_name = f'"{dataset_name}"'
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {dataset_name} (
                series_id TEXT PRIMARY KEY,
                num_slices INTEGER,
                image_shape_x INTEGER,
                image_shape_y INTEGER,
                spacing_x REAL,
                spacing_y REAL,
                spacing_z REAL
            )
            """
        )

    def insert_global_data(
        self,
        cursor: sqlite3.Cursor,
        dataset_name: str,
        series_id: str,
        slice_index: int,
        dicom_path: str,
    ) -> None:
        """
        Insert global data into the database.

        Parameters:
            cursor (sqlite3.Cursor): The cursor object to execute SQL queries.
            dataset_name (str): The name of the dataset.
            series_id (str): The ID of the series.
            slice_index (int): The index of the slice.
            dicom_path (str): The path to the DICOM file.

        Returns:
            None
        """
        cursor.execute(
            """
            INSERT INTO "global" (
                dataset, series_id, slice_index, dicom_path
            )
            VALUES (?, ?, ?, ?)
            """,
            (dataset_name, series_id, slice_index, dicom_path),
        )

    def insert_dataset_data(
        self, cursor: sqlite3.Cursor, dataset_name: str, series_id: str, metadata: dict
    ) -> None:
        """
        Insert dataset data into the specified table.

        Args:
            cursor (sqlite3.Cursor): The cursor object for executing SQL statements.
            dataset_name (str): The name of the dataset table.
            series_id (str): The series ID.
            metadata (dict): The metadata dictionary containing information about the dataset.

        Returns:
            None
        """
        dataset_name = f'"{dataset_name}"'
        cursor.execute(
            f"""
            INSERT INTO {dataset_name} (
                series_id, num_slices, image_shape_x, image_shape_y,
                spacing_x, spacing_y, spacing_z
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                series_id,
                metadata["num_slices"],
                metadata["image_shape"][0],
                metadata["image_shape"][1],
                metadata["spacing"][0],
                metadata["spacing"][1],
                metadata["spacing"][2],
            ),
        )


def get_argpase():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--derived_okay", type=bool, default=False, required=False)
    parser.add_argument("--validate_only", type=bool, default=False, required=False)
    parser.add_argument(
        "--db_path", type=str, default="data/radiomics_datasets.db", required=False
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
        "derived_okay": args.derived_okay
    }
    dataset = DatasetBase(config)
    if args.validate_only:
        dataset.get_series_paths()
    else:
        dataset.prepare_dataset()


if __name__ == "__main__":
    parser = get_argpase()
    args = parser.parse_args()
    main(args)

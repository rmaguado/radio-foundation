from typing import List, Dict, Tuple, Any

import argparse
from tqdm import tqdm
import pydicom
import ast
import os
import logging

from ..utils import walk, set_logging
from .ct_database import CtDatabase


logger = logging.getLogger("dataprep")


class DicomCtValidation:
    def __init__(self, config: Dict):
        self.derived_okay = config["derived_okay"]
        self.skip_validation = config["skip_validation"]

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
        orientation = [round(float(x), 1) for x in orientation]
        if len(orientation) != 6:
            return False
        if orientation[2] != 0 and orientation[2] != -1:
            return False
        return True

    def test_image_size(self, fields: Dict[str, Any]) -> str:
        rows = fields["Rows"]
        cols = fields["Columns"]
        if rows != 512 or cols != 512:
            return f"\tImage size is not 512x512: ({rows}, {cols}).\n"
        return ""

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
        if self.skip_validation:
            return
        self.validate_dicom_for_ct(dcm, dicom_folder_path)

    def validate_dicom_for_ct(self, dcm, dicom_folder_path):
        """
        Validates a DICOM file for CT scans.
        Args:
            dcm: The DICOM object to validate.
            dicom_folder_path: The folder path of the DICOM file.
        """
        fields, missing_fields = self.get_fields(dcm)
        assert (
            not missing_fields
        ), f"{dicom_folder_path}: Missing fields: {', '.join(missing_fields)}. Skipping."

        issues = ""
        issues += self.test_modality(fields)
        issues += self.test_orientation(fields)
        issues += self.test_slice_thickness(fields)
        issues += self.test_image_type(fields)

        assert len(issues) == 0, issues


class DicomDatabase(CtDatabase):
    def __init__(self, config):
        super().__init__(config, storage="dicom")

        self.create_global_table()
        self.add_dataset(config["dataset_name"])
        self.create_dataset_info_table(config["dataset_name"])

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

    def create_dataset_info_table(self, dataset_name) -> None:
        """
        Create a dataset table in the database.
        """
        dataset_name_str = f'"{dataset_name}"'
        self.cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {dataset_name_str} (
                series_id TEXT PRIMARY KEY,
                num_slices INTEGER,
                image_shape_x INTEGER,
                image_shape_y INTEGER,
                slice_thickness REAL,
                spacing_x REAL,
                spacing_y REAL
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

    def insert_dataset_data(self, metadata: Tuple, dataset_name: str) -> None:
        """
        Insert dataset data into the specified table.

        Args:
            metadata (dict): The metadata dictionary containing information about the dataset.
        """
        dataset_name_str = f'"{dataset_name}"'
        self.cursor.execute(
            f"""
            INSERT INTO {dataset_name_str} (
                series_id, num_slices, image_shape_x, image_shape_y,
                slice_thickness, spacing_x, spacing_y
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            metadata,
        )


class DicomProcessor:
    def __init__(self, config: dict):
        self.dataset_name = config["dataset_name"]
        self.absolute_dataset_path = os.path.abspath(config["dataset_path"])
        self.validate_only = config["validate_only"]

        self.validator = DicomCtValidation(config)
        if not self.validate_only:
            self.database = DicomDatabase(config)

        log_path = f"data/log/{self.dataset_name}.log"
        set_logging(log_path)

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
        spacing_x, spacing_y = dcm.PixelSpacing
        return spacing_x, spacing_y, dcm.SliceThickness

    def load_series(self, dicom_paths: List[str]):
        """
        Loads the dicoms in the series, groups them by series id and sorts them by position in patient.

        Args:
            dicom_paths (List[str]): A list of paths to the dicom files.

        Returns:
            Dict[str, List[Tuple[str, pydicom.dataset.FileDataset]]]: A dictionary containing the series id as the key and a list of tuples containing the path and the dicom object as the value.
        """
        paths_dicoms = [
            (path, pydicom.dcmread(path, stop_before_pixels=True))
            for path in dicom_paths
        ]

        grouped_dicoms = {}

        for path, dcm in paths_dicoms:
            series_id = dcm.get("SeriesInstanceUID")
            if series_id not in grouped_dicoms:
                grouped_dicoms[series_id] = []
            grouped_dicoms[series_id].append((path, dcm))

        for series_id, series_dicoms in list(grouped_dicoms.items()):
            try:
                series_dicoms.sort(key=lambda x: x[1].ImagePositionPatient[2])
            except AttributeError:
                logger.warning(
                    f"Series {series_id} does not have ImagePositionPatient attribute."
                )
                del grouped_dicoms[series_id]

        return grouped_dicoms

    def process_series(
        self, dicom_paths: List[Tuple[str, pydicom.dataset.FileDataset]], series_id: str
    ) -> None:
        """
        Process a series and return metadata and paths to dicom files.

        Args:
            dicom_paths (List[Tuple[str, pydicom.dataset.FileDataset]]): A list of tuples containing the path and the dicom object.
            series_id (str): Series ID.
        """
        first_dicom = dicom_paths[0][1]
        num_slices = len(dicom_paths)
        image_shape = self.get_shape(first_dicom)
        spacing_x, spacing_y, slice_thickness = self.get_spacing(first_dicom)

        metadata = (
            series_id,
            num_slices,
            image_shape[0],
            image_shape[1],
            slice_thickness,
            spacing_x,
            spacing_y,
        )

        for slice_index, (dicom_path, dicom) in enumerate(dicom_paths):
            rel_dicom_path = os.path.relpath(dicom_path, self.absolute_dataset_path)
            self.database.insert_global_data(
                self.dataset_name, series_id, slice_index, rel_dicom_path
            )

        self.database.insert_dataset_data(metadata, self.dataset_name)
        logger.info(f"Processed series {series_id}.")

    def prepare_dataset(self) -> None:
        """
        Prepares the dataset by creating necessary tables in the database and inserting data.
        """
        included_series_ids = []

        logger.info(f"Processing dataset: {self.dataset_name}")
        print("Walking dataset directories.")
        total_dirs = sum(1 for _ in walk(self.absolute_dataset_path))
        logger.info(f"{self.dataset_name} total directories: {total_dirs}")
        for data_folder, dirs, files in tqdm(
            walk(self.absolute_dataset_path), total=total_dirs
        ):
            dcm_paths = [
                os.path.join(data_folder, f) for f in files if f.endswith(".dcm")
            ]
            if not dcm_paths:
                continue

            try:
                grouped_series = self.load_series(dcm_paths)
            except Exception as e:
                logger.exception(f"Error loading series from {data_folder}: {e}")
                continue

            for series_id, dicoms in grouped_series.items():

                try:
                    for dicom_path, dicom in dicoms:
                        self.validator(dicom, data_folder)
                except AssertionError as e:
                    logger.error(f"Error validating {data_folder}: {e}")
                    continue
                except Exception as e:
                    logger.exception(f"Error validating {data_folder}: {e}")
                    continue

                if series_id in included_series_ids:
                    logger.info(
                        f"{data_folder}: Series ID already in database. Skipping."
                    )
                    continue

                if not self.validate_only:
                    try:
                        self.process_series(dicoms, series_id)
                    except Exception as e:
                        logger.exception(f"Error processing series {series_id}: {e}")
                        continue
                included_series_ids.append(series_id)

        logger.info(
            f"Finished processing {self.dataset_name}. {len(included_series_ids)} series included."
        )

    def close_db(self) -> None:
        """
        Closes the database connection.
        """
        self.database.close()


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
        "--db_name",
        type=str,
        default="dicom_datasets",
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
        "derived_okay": args.derived_okay,
        "validate_only": args.validate_only,
        "skip_validation": args.skip_validation,
    }
    processor = DicomProcessor(config)
    processor.prepare_dataset()


if __name__ == "__main__":
    parser = get_argpase()
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        logger.exception(e)
        raise e

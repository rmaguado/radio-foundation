"""
Base indexing classes for evaluating a dataset.
Extends scripts in data/dataprep.
Includes map_id in the SQLite database to reference metadata.
"""

from typing import List, Tuple
import os
from tqdm import tqdm
import pydicom
import nibabel as nib
import argparse
import logging

from data.utils import set_logging
from data.dataprep import (
    DicomProcessor,
    DicomDatabase,
    NiftiProcessor,
    NiftiDatabase,
    NpzProcessor,
    NpzDatabase,
)


logger = logging.getLogger("dataprep")


class DicomEvalBase(DicomDatabase):
    def __init__(self, config):
        super().__init__(config)

    def create_dataset_info_table(self, dataset_name) -> None:
        """
        Create a dataset table in the database.
        Includes additional column, 'map_id', to find metadata.
        """
        dataset_name_str = f'"{dataset_name}"'
        self.cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {dataset_name_str} (
                series_id TEXT PRIMARY KEY,
                map_id TEXT,
                num_slices INTEGER,
                image_shape_x INTEGER,
                image_shape_y INTEGER,
                slice_thickness REAL,
                spacing_x REAL,
                spacing_y REAL
            )
            """
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
                series_id, map_id, num_slices, image_shape_x, image_shape_y,
                slice_thickness, spacing_x, spacing_y
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            metadata,
        )


class DicomProcessorBase(DicomProcessor):
    def __init__(self, config: dict, database):
        self.dataset_name = config["dataset_name"]
        self.absolute_dataset_path = os.path.abspath(config["dataset_path"])

        self.database = database

        log_path = f"data/log/{self.dataset_name}_eval.log"
        set_logging(log_path)

    def process_series(
        self,
        dicom_paths: List[Tuple[str, pydicom.dataset.FileDataset]],
        series_id: str,
        map_id: str,
    ) -> None:
        """
        Process a series and return metadata and paths to dicom files.

        Args:
            dicom_paths (List[Tuple[str, pydicom.dataset.FileDataset]]): A list of tuples containing the path and the dicom object.
            series_id (str): Series ID.
            map_id (str): ID to link the scan to its metadata.
        """
        first_dicom = dicom_paths[0][1]
        num_slices = len(dicom_paths)
        image_shape = self.get_shape(first_dicom)
        spacing_x, spacing_y, slice_thickness = self.get_spacing(first_dicom)

        metadata = (
            series_id,
            map_id,
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

    def get_paths_and_mapids(self) -> List[Tuple[str, str]]:
        """
        Get paths to dicom files and their corresponding mapids.
        Mapids are taken from the folder in the first level of the dataset.
        """
        paths_ids = []

        mapids = [
            folder
            for folder in os.listdir(self.absolute_dataset_path)
            if os.path.isdir(os.path.join(self.absolute_dataset_path, folder))
        ]

        for mapid in mapids:
            for scan_path, dirs, files in os.walk(
                os.path.join(self.absolute_dataset_path, mapid)
            ):

                if len(files) > 0:
                    if files[0].endswith(".dcm"):
                        paths_ids.append((scan_path, mapid))
                        break

        return paths_ids

    def get_dcm_paths(self, scan_path: str) -> List[str]:
        dcm_paths = [
            os.path.join(scan_path, f)
            for f in os.listdir(scan_path)
            if f.endswith(".dcm")
        ]
        return dcm_paths

    def prepare_dataset(self) -> None:
        """
        Prepares the dataset by creating necessary tables in the database and inserting data.
        """

        included_series_ids = []

        paths_ids = self.get_paths_and_mapids()

        for scan_path, mapid in tqdm(paths_ids):
            dcm_paths = self.get_dcm_paths(scan_path)
            if not dcm_paths:
                logger.warning(f"No dicom files found in {scan_path}. Skipping.")
                continue

            try:
                grouped_series = self.load_series(dcm_paths)
            except Exception as e:
                logger.exception(f"Error loading series from {scan_path}: {e}")
                continue

            if len(grouped_series.keys()) != 1:
                logger.warning(
                    f"Found {len(grouped_series.keys())} series in {scan_path}. Skipping."
                )
                continue

            for series_id, dicoms in grouped_series.items():

                try:
                    self.process_series(dicoms, series_id, mapid)
                    included_series_ids.append(series_id)
                except Exception as e:
                    logger.exception(f"Error processing series {series_id}: {e}")
                    continue

        logger.info(
            f"Finished processing {self.dataset_name}. {len(included_series_ids)} series included."
        )


class NiftiEvalBase(NiftiDatabase):
    def __init__(self, config):
        super().__init__(config)

    def create_global_table(self) -> None:
        """
        Creates a global table in the database for NIfTI files.
        """
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS global (
                dataset TEXT,
                map_id TEXT,
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
            INSERT INTO "global" (dataset, map_id, num_slices, slice_thickness, spacing_x, spacing_y, axial_dim, nifti_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_name,
                metadata["map_id"],
                metadata["num_slices"],
                metadata["slice_thickness"],
                metadata["spacing_x"],
                metadata["spacing_y"],
                metadata["axial_dim"],
                nifti_path,
            ),
        )


class NiftiProcessorBase(NiftiProcessor):
    def __init__(self, config: dict, database):
        self.dataset_name = config["dataset_name"]
        self.absolute_dataset_path = os.path.abspath(config["dataset_path"])

        self.database = database

        log_path = f"data/log/{self.dataset_name}_eval.log"
        set_logging(log_path)

    def process_volume(self, nifti_path: str, map_id: str) -> None:
        """
        Process a NIfTI volume, extract metadata, and store it in the database.

        Args:
            nifti_path (str): Path to the NIfTI file.
            map_id (str): ID to link the scan to its metadata.
        """

        nifti_file = nib.load(nifti_path)
        try:
            metadata = self.get_metadata(nifti_file)
            metadata["map_id"] = map_id
        except Exception as e:
            logger.exception(
                f"Metadata extraction failed for {nifti_path} (mad_id: {map_id}): {e}"
            )
            return

        rel_nifti_path = os.path.relpath(nifti_path, self.absolute_dataset_path)
        self.database.insert_global_data(self.dataset_name, metadata, rel_nifti_path)

        logger.info(f"Processed nifti: {nifti_path} (map_id: {map_id}).")

    def get_paths_and_mapids(self) -> List[Tuple[str, str]]:
        """
        Get paths to nifti files and their corresponding mapids.
        Mapids are taken from the name of the nifti file.
        """
        paths_ids = []

        for data_folder, dirs, files in os.walk(self.absolute_dataset_path):
            nii_paths = [
                os.path.join(data_folder, f)
                for f in files
                if f.endswith(".nii.gz") or f.endswith(".nii")
            ]

            if not nii_paths:
                continue

            for nii_path in nii_paths:
                map_id = os.path.basename(nii_path).split(".")[0]
                paths_ids.append((nii_path, map_id))

        return paths_ids

    def prepare_dataset(self) -> None:
        """
        Prepares the dataset by scanning the directories and processing NIfTI files.
        """

        logger.info(f"Processing dataset: {self.dataset_name}")

        paths_ids = self.get_paths_and_mapids()
        total_niftis = len(paths_ids)
        logger.info(f"Total NIfTI files: {total_niftis}")

        volume_counts = 0

        for nii_path, map_id in tqdm(paths_ids):
            try:
                self.process_volume(nii_path, map_id)
                volume_counts += 1
            except Exception as e:
                logger.exception(
                    f"Error processing nifti {nii_path} (map_id: {map_id}): {e}"
                )

        logger.info(
            f"Finished processing {self.dataset_name}. Added {volume_counts} volumes."
        )


class NpzEvalBase(NpzDatabase):
    def __init__(self, config):
        super().__init__(config)

    def create_global_table(self) -> None:
        """
        Creates a global table in the database for NIfTI files.
        """
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS global (
                dataset TEXT,
                map_id TEXT,
                volume_path TEXT
            )
            """
        )

    def insert_global_data(
        self, dataset_name: str, map_id: str, volume_path: str
    ) -> None:
        self.cursor.execute(
            """
            INSERT INTO "global" (dataset, map_id, volume_path)
            VALUES (?, ?, ?)
            """,
            (dataset_name, map_id, volume_path),
        )


class NpzProcessorBase(NpzProcessor):
    def __init__(self, config: dict, database):
        self.dataset_name = config["dataset_name"]
        self.absolute_dataset_path = os.path.abspath(config["dataset_path"])

        self.database = database

        log_path = f"data/log/{self.dataset_name}_eval.log"
        set_logging(log_path)

    def process_volume(self, volume_path: str, map_id: str) -> None:
        """
        Process a Npz volume, extract metadata, and store it in the database.

        Args:
            volume_path (str): Path to the Npz file.
            map_id (str): ID to link the scan to its metadata.
        """
        rel_nifti_path = os.path.relpath(volume_path, self.absolute_dataset_path)
        self.database.insert_global_data(self.dataset_name, map_id, rel_nifti_path)

        logger.info(f"Processed npz volume: {volume_path} (map_id: {map_id}).")

    def get_paths_and_mapids(self) -> List[Tuple[str, str]]:
        """
        Get paths to files and their corresponding mapids.
        Mapids are taken from the name of the file.
        """
        paths_ids = []

        for data_folder, dirs, files in os.walk(self.absolute_dataset_path):
            paths = [os.path.join(data_folder, f) for f in files if f.endswith(".npz")]

            if not paths:
                continue

            for path in paths:
                map_id = os.path.basename(path).split(".")[0]
                paths_ids.append((path, map_id))

        return paths_ids

    def prepare_dataset(self) -> None:
        """
        Prepares the dataset by scanning the directories and processing files.
        """

        logger.info(f"Processing dataset: {self.dataset_name}")

        paths_ids = self.get_paths_and_mapids()
        total_files = len(paths_ids)
        logger.info(f"Total files: {total_files}")

        volume_counts = 0

        for path, map_id in tqdm(paths_ids):
            try:
                self.process_volume(path, map_id)
                volume_counts += 1
            except Exception as e:
                logger.exception(f"Error processing {path} (map_id: {map_id}): {e}")

        logger.info(
            f"Finished processing {self.dataset_name}. Added {volume_counts} volumes."
        )


def get_argpase():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--root_path", type=str, required=True, help="The root path of the dataset."
    )
    return parser


def main(args):
    dataset_name = "DemoDatasetName"
    db_name = "DemoDatasetName-Eval"
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
    processor = DicomProcessorBase(config, database)
    processor.prepare_dataset()


if __name__ == "__main__":
    parser = get_argpase()
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        logger.exception(e)
        raise e

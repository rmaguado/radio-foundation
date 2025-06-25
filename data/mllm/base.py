"""
Base indexing classes for text and CT datasets.
Extends scripts in data/dataprep.
Includes the report field in the SQLite database.
"""

from typing import List, Tuple
import os
from tqdm import tqdm
import nibabel as nib
import argparse
import logging

from data.utils import set_logging
from data.dataprep import (
    NiftiProcessor,
    NiftiDatabase,
)


logger = logging.getLogger("dataprep")


class NiftiReportBase(NiftiDatabase):
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
                nifti_path TEXT,
                text TEXT,
                length INTEGER
            )
            """
        )

    def insert_global_data(
        self, dataset_name: str, metadata: dict, nifti_path: str, text: str
    ) -> None:
        self.cursor.execute(
            """
            INSERT INTO "global" (dataset, map_id, num_slices, slice_thickness, spacing_x, spacing_y, axial_dim, nifti_path, text, length)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                text,
                len(text),
            ),
        )


class NiftiProcessorBase(NiftiProcessor):
    def __init__(self, config: dict, database):
        self.dataset_name = config["dataset_name"]
        self.absolute_dataset_path = os.path.abspath(config["dataset_path"])

        self.database = database

        log_path = f"data/log/{self.dataset_name}_mllm.log"
        set_logging(log_path)

    def get_text_data(self, map_id: str) -> str:
        """
        Get the text data for a given map_id.
        """
        raise NotImplementedError

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
            return 0

        try:
            text = self.get_text_data(map_id)
        except Exception as e:
            logger.exception(
                f"Text extraction failed for {nifti_path} (mad_id: {map_id}): {e}"
            )
            return 0

        rel_nifti_path = os.path.relpath(nifti_path, self.absolute_dataset_path)
        self.database.insert_global_data(
            self.dataset_name, metadata, rel_nifti_path, text
        )

        logger.info(f"Processed nifti: {nifti_path} (map_id: {map_id}).")

        return 1

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
                volume_counts += self.process_volume(nii_path, map_id)
            except Exception as e:
                logger.exception(
                    f"Error processing nifti {nii_path} (map_id: {map_id}): {e}"
                )

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
    db_name = "DemoDatasetName-mllm"
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
    database = NiftiReportBase(config)
    processor = NiftiProcessorBase(config, database)
    processor.prepare_dataset()


if __name__ == "__main__":
    parser = get_argpase()
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        logger.exception(e)
        raise e

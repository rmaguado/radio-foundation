import sqlite3
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from tqdm import tqdm
from typing import List, Tuple
import os
from abc import ABC, abstractmethod


class DatasetBase(ABC):
    def __init__(self, config: dict):
        sitk.ProcessObject_SetGlobalWarningDisplay(False)
        self.config = config

    @abstractmethod
    def get_series_paths(self) -> List[Tuple[str, str]]:
        """
        Return a list of tuples with series_id and path to series.

        Returns:
            List[Tuple[str, str]]: A list of tuples with series_id and path to series.
        """
        pass

    def get_spacing(self, image: sitk.Image) -> List[float]:
        spacing = image.GetSpacing()
        return [spacing[i] for i in [2, 0, 1]]

    def get_shape(self, image: sitk.Image) -> Tuple[Tuple[int, int], int]:
        image_array = sitk.GetArrayFromImage(image)
        return image_array.shape[1:], image_array.shape[0]

    def load_image(self, dicom_paths: List[str]) -> sitk.Image:
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
        assert hasattr(
            self, "get_series_paths"
        ), "The method 'get_series_paths' must be implemented."

        dataset_name = self.config["dataset_name"]
        absolute_dataset_path = os.path.abspath(self.config["dataset_path"])
        
        conn = sqlite3.connect(self.config["target_path"])
        cursor = conn.cursor()

        self.create_global_table(cursor)
        self.create_dataset_table(cursor, dataset_name)

        series_paths = self.get_series_paths()
        print(f"Processing {len(series_paths)} series.")
        for series_id, series_path in tqdm(series_paths):
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

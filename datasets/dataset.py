import sqlite3
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from tqdm import tqdm
from typing import List, Tuple
from abc import ABC, abstractmethod


class DatasetBase(ABC):
    def __init__(self, config: dict):
        sitk.ProcessObject_SetGlobalWarningDisplay(False)

        self.config = config
        self.statistics = StatisticsManager()

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
        return image_array.shape[1, 2], image_array.shape[0]

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

        self.statistics.update_statistics(sitk.GetArrayFromImage(image))

        return metadata, dicom_paths

    def prepare_dataset(self) -> None:
        assert hasattr(
            self, "get_series_paths"
        ), "The method 'get_series_paths' must be implemented."

        dataset_name = self.config["dataset_name"]

        conn = sqlite3.connect(self.config["target_path"])
        cursor = conn.cursor()

        self.create_global_table(cursor)
        self.create_dataset_table(cursor, dataset_name)
        self.create_statistics_table(cursor)

        series_paths = self.get_series_paths()
        print(f"Processing {len(series_paths)} series.")
        for series_id, series_path in tqdm(series_paths):
            metadata, dicom_paths = self.process_series(series_path, series_id)

            for slice_index, dicom_path in enumerate(dicom_paths):
                self.insert_global_data(
                    cursor, dicom_path, series_id, dataset_name, slice_index
                )
            self.insert_dataset_data(cursor, dataset_name, series_id, metadata)

        statistics = self.statistics.gather_statistics()
        self.insert_statistics_data(cursor, dataset_name, statistics)

        conn.commit()
        conn.close()

    def create_global_table(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS global (
                dicom_path TEXT PRIMARY KEY,
                series_id TEXT,
                dataset TEXT,
                slice_index REAL
            )
            """
        )

    def create_dataset_table(self, cursor: sqlite3.Cursor, dataset_name: str) -> None:
        dataset_name = f'"{dataset_name}"'
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS (?) (
                series_id TEXT PRIMARY KEY,
                num_slices INTEGER,
                image_shape_x INTEGER,
                image_shape_y INTEGER,
                image_shape_z INTEGER,
                spacing_x REAL,
                spacing_y REAL,
                spacing_z REAL
            )
            """,
            (dataset_name),
        )

    def create_statistics_table(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS "statistics" (
                dataset TEXT PRIMARY KEY,
                mean REAL,
                std_dev REAL,
                total_slices INTEGER,
                total_series INTEGER
            )
            """
        )

    def insert_global_data(
        self,
        cursor: sqlite3.Cursor,
        dicom_path: str,
        series_id: str,
        dataset_name: str,
        slice_index: int,
    ) -> None:
        cursor.execute(
            """
            INSERT INTO "global" (
                dicom_path, series_id, dataset, slice_index
            )
            VALUES (?, ?, ?, ?)
            """,
            (dicom_path, series_id, dataset_name, slice_index),
        )

    def insert_dataset_data(
        self, cursor: sqlite3.Cursor, dataset_name: str, series_id: str, metadata: dict
    ) -> None:

        cursor.execute(
            f"""
            INSERT INTO (?) (
                series_id, series_id, num_slices, image_shape_x, image_shape_y, image_shape_z,
                spacing_x, spacing_y, spacing_z
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_name,
                series_id,
                metadata["num_slices"],
                metadata["image_shape"][0],
                metadata["image_shape"][1],
                metadata["image_shape"][2],
                metadata["spacing"][0],
                metadata["spacing"][1],
                metadata["spacing"][2],
            ),
        )

    def insert_statistics_data(
        self, cursor: sqlite3.Cursor, dataset_name: str, statistics: dict
    ) -> None:
        cursor.execute(
            """
            INSERT INTO "statistics" (dataset, mean, std_dev, total_slices, total_series)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                dataset_name,
                statistics["mean"],
                statistics["std_dev"],
                statistics["total_slices"],
                statistics["total_series"],
            ),
        )


class StatisticsManager:
    def __init__(self):
        self.total_voxels: int = 0
        self.sum_voxels: float = 0.0
        self.sum_squares_voxels: float = 0.0
        self.total_slices: int = 0
        self.total_series: int = 0

    def update_statistics(self, image_array: np.ndarray) -> None:
        """
        Update the running sum and sum of squares of voxel intensities.
        """
        self.total_voxels += image_array.size
        self.sum_voxels += np.sum(image_array)
        self.sum_squares_voxels += np.sum(np.square(image_array))
        self.total_slices += image_array.shape[0]
        self.total_series += 1

    def gather_statistics(self) -> Tuple[float, float]:
        """
        Calculate and the global mean and standard deviation and return the collected statistics.
        """
        mean = self.sum_voxels / self.total_voxels
        variance = (self.sum_squares_voxels / self.total_voxels) - (mean**2)
        std_dev = np.sqrt(variance)

        return {
            "mean": mean,
            "std_dev": std_dev,
            "total_slices": self.total_slices,
            "total_series": self.total_series,
        }

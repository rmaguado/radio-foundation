import logging
from typing import Optional, Any, Callable
import os
import torch
import pydicom
import numpy as np

from .base import BaseDataset


logger = logging.getLogger("dinov2")


class DicomVolumes(BaseDataset):
    def __init__(self) -> None:
        super().__init__()

    def create_entries(self) -> np.ndarray:
        """
        Generates a numpy memmap object pointing to the sqlite database rows of dicom paths.
        For collecting various slices of a CT scan (multi-channel) each memmap row contains the ordered rowids of the slices.

        Returns:
            np.ndarray: The entries dataset (memmap).
        """
        logger.info(f"Creating entries for {self.dataset_name}.")

        slice_stack_num = self.channels

        dataset_names = self.cursor.execute(f"SELECT dataset FROM datasets").fetchall()

        series_ids = []
        for dataset_name in dataset_names:
            dataset_name = dataset_name[0]
            dataset_series = self.cursor.execute(
                f"SELECT series_id, num_slices FROM '{dataset_name}'"
            ).fetchall()
            series_ids += [
                (dataset_name, series_id, num_slices)
                for series_id, num_slices in dataset_series
            ]

        logger.info(f"Total number of series: {len(series_ids)}.")

        entries = []
        for dataset_name, series_id, num_slices in series_ids:

            if num_slices < slice_stack_num:
                continue

            self.cursor.execute(
                """
                SELECT rowid, slice_index 
                FROM global 
                WHERE series_id = ? 
                AND dataset = ? 
                ORDER BY slice_index
                """,
                (series_id, dataset_name),
            )
            slice_indexes_rowid = self.cursor.fetchall()
            slice_indexes_rowid.sort(key=lambda x: x[1])

            sorted_rowids = [x[0] for x in slice_indexes_rowid]

            stack_rows = [
                [x for x in sorted_rowids[i : i + slice_stack_num]]
                for i in range(len(sorted_rowids) - slice_stack_num + 1)
            ]

            entries += stack_rows

        entries_array = np.array(entries, dtype=np.uint32)

        entries_dir = self.get_entries_dir()

        logger.info(f"Saving entries to {entries_dir}.")
        np.save(entries_dir, entries_array)
        return np.load(entries_dir, mmap_mode="r")

    def get_target(self, index: int) -> Optional[Any]:
        return None

    def __len__(self) -> int:
        return len(self.entries)


class DicomCtDataset(DicomVolumes):

    def __init__(
        self,
        dataset_name: str,
        root_path: str,
        channels: int = 1,
        lower_window: int = -1000,
        upper_window: int = 1900,
        transform: Optional[Callable] = lambda _: _,
        target_transform: Optional[Callable] = lambda _: _,
    ) -> None:
        """
        Initializes the DicomCtDataset.

        Args:
            dataset_name (str): The name of the dataset.
            root_path (str): The root path of the dataset.
            channels (int): The number of channels to use.
            lower_window (int): The lower window value.
            upper_window (int): The upper window value.
            transform (Optional[Callable], optional): A function to apply to the data. Defaults to lambda _: _.
            target_transform (Optional[Callable], optional): A function to apply to the target. Defaults to lambda _: _.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.index_path = os.path.join(
            "file:data/index", self.dataset_name, "index.db?mode=ro"
        )
        self.entries_path = os.path.join("data/index", self.dataset_name, "entries")

        self.root_path = root_path
        self.channels = channels

        self.transform = transform
        self.target_transform = target_transform

        self.lower_window = lower_window
        self.upper_window = upper_window

        self.open_db()
        self.entries = self.get_entries()

    def get_image_data(self, index: int) -> torch.Tensor:
        """
        Retrieves the image data for a given index.

        Args:
            index (int): The index of the image data to retrieve.

        Returns:
            torch.Tensor: The image data as a torch tensor.
        """
        stack_rowids = [int(x) for x in self.entries[index]]
        self.cursor.execute(
            """
            SELECT rowid, slice_index, dataset, dicom_path
            FROM global 
            WHERE rowid IN ({})
            """.format(
                ",".join("?" * self.channels)
            ),
            stack_rowids,
        )
        stack_rows = self.cursor.fetchall()
        stack_rows.sort(key=lambda x: x[1])

        try:
            stack_data = self.create_stack_data(stack_rows)
        except Exception as e:
            logger.exception(f"Error processing stack. Rowids: {stack_rowids} \n{e}")
            stack_data = torch.zeros((self.channels, 512, 512))

        return stack_data

    def create_stack_data(self, stack_rows):
        stack_data = []
        for _, _, dataset, rel_dicom_path in stack_rows:
            abs_dicom_path = os.path.join(self.root_path, dataset, rel_dicom_path)
            dcm = pydicom.dcmread(abs_dicom_path)
            stack_data.append(self.process_ct(dcm))

        return torch.stack(stack_data)

    def process_ct(self, dcm: pydicom.dataset.FileDataset) -> torch.Tensor:
        """
        Process a CT scan by applying rescaling and windowing.

        Args:
            dcm (pydicom.dataset.FileDataset): The DICOM object representing the CT scan.

        Returns:
            torch.Tensor: The processed CT scan as a tensor.

        """
        slope = getattr(dcm, "RescaleSlope", 1)
        intercept = getattr(dcm, "RescaleIntercept", 0)

        array_data = dcm.pixel_array.astype(np.float32) * slope + intercept

        array_data = np.clip(array_data, self.lower_window, self.upper_window)
        array_data = (array_data - self.lower_window) / (
            self.upper_window - self.lower_window
        )

        return torch.tensor(array_data, dtype=torch.float32)


class DicomCTVolumesFull(DicomCtDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_entries_dir(self) -> np.ndarray:
        return os.path.join(self.entries_path, "full.npy")

    def create_entries(self) -> np.ndarray:
        """
        Generates a numpy memmap object pointing to the sqlite database rows of dicom paths.

        Returns:
            np.ndarray: The entries dataset (memmap).
        """
        logger.info(f"Creating entries for {self.dataset_name}.")

        dataset_names = self.cursor.execute(f"SELECT dataset FROM datasets").fetchall()

        entries_dtype = [
            ("dataset", "U256"),
            ("rowid", np.uint32),
        ]
        entries = []
        for dataset_name in dataset_names:
            dataset_name = dataset_name[0]
            dataset_series = self.cursor.execute(
                f"SELECT rowid FROM '{dataset_name}'"
            ).fetchall()
            entries += [(dataset_name, rowid) for rowid in dataset_series]
        logger.info(f"Total number of scans: {len(entries)}.")

        entries_array = np.array(entries, dtype=entries_dtype)

        entries_dir = self.get_entries_dir()
        logger.info(f"Saving entries to {entries_dir}.")
        np.save(entries_dir, entries_array)
        return np.load(entries_dir, mmap_mode="r")

    def get_image_data(self, index: int) -> torch.Tensor:
        dataset_name, rowid = self.entries[index]
        self.cursor.execute(
            f"SELECT series_id FROM '{dataset_name}' WHERE rowid = ?", (rowid,)
        )
        series_id = self.cursor.fetchone()

        self.cursor.execute(
            """
            SELECT slice_index, dataset, dicom_path
            FROM global 
            WHERE series_id = ? 
            AND dataset = ?
            """,
            (series_id, dataset_name),
        )
        slice_indexes_rowid = self.cursor.fetchall()
        slice_indexes_rowid.sort(key=lambda x: x[0])

        try:
            stack_data = self.create_stack_data(slice_indexes_rowid)
        except Exception as e:
            logger.exception(f"Error processing stack. Seriesid: {series_id} \n{e}")
            stack_data = torch.zeros((10, 512, 512))

        return stack_data

    def create_stack_data(self, stack_rows):
        stack_data = []
        for _, dataset, rel_dicom_path in stack_rows:
            abs_dicom_path = os.path.join(self.root_path, dataset, rel_dicom_path)
            dcm = pydicom.dcmread(abs_dicom_path)
            stack_data.append(self.process_ct(dcm))

        return torch.stack(stack_data)

    def get_target(self, index: int) -> Optional[Any]:
        """Maybe get it from a csv file"""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.entries)

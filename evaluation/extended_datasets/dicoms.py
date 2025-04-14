import logging
import os
import numpy as np
import torch
import pydicom
from typing import Tuple, Any

from dinov2.data.datasets.dicoms import DicomCTVolumesFull


logger = logging.getLogger("dinov2")


class DicomFullVolumeEval(DicomCTVolumesFull):
    def __init__(
        self,
        root_path: str,
        dataset_name: str,
        channels: int = 1,
        transform=None,
    ):
        super().__init__(
            dataset_name=dataset_name,
            root_path=root_path,
            channels=channels,
            transform=transform,
        )

    def create_entries(self) -> np.ndarray:
        """
        Generates a numpy memmap object pointing to the sqlite database rows of dicom paths.

        Returns:
            np.ndarray: The entries dataset (memmap).
        """
        logger.info(f"Creating entries for {self.dataset_name}.")

        dataset_names = self.cursor.execute(f"SELECT dataset FROM datasets").fetchall()
        assert len(dataset_names) == 1, "Must only have one dataset table in database."
        self.dataset = dataset_names[0][0]
        logger.info(f"dataset name (folder name with dicoms):{self.dataset}")

        entries_dtype = [
            ("series_id", "U256"),
            ("map_id", "U256"),
        ]
        entries = []
        dataset_series = self.cursor.execute(
            f"SELECT series_id, map_id FROM '{self.dataset}'"
        ).fetchall()

        for series_id, map_id in dataset_series:
            entries.append((series_id, map_id))

        logger.info(f"Total number of scans: {len(entries)}.")

        entries_array = np.array(entries, dtype=entries_dtype)

        entries_dir = self.get_entries_dir()
        logger.info(f"Saving entries to {entries_dir}.")
        np.save(entries_dir, entries_array)
        return np.load(entries_dir, mmap_mode="r")

    def get_image_data(self, index: int) -> Tuple[torch.Tensor, str]:
        series_id, map_id = self.entries[index]

        self.cursor.execute(
            """
            SELECT slice_index, dicom_path
            FROM global 
            WHERE series_id = ?
            """,
            (series_id,),
        )
        stack_rows = self.cursor.fetchall()
        stack_rows.sort(key=lambda x: x[0])

        slice_thickness = self.cursor.execute(
            f"SELECT slice_thickness FROM '{self.dataset}' WHERE series_id = ?",
            (series_id,),
        ).fetchone()[0]

        try:
            stack_data = self.create_stack_data(stack_rows)
        except Exception as e:
            logger.exception(f"Error processing stack (map_id: {map_id}) \n{e}")
            stack_data = torch.zeros((1, self.channels, 512, 512), dtype=torch.float32)

        return stack_data, map_id, slice_thickness

    def create_stack_data(self, stack_rows):
        dicom_files = []
        for i, rel_dicom_path in stack_rows:
            abs_dicom_path = os.path.join(self.root_path, self.dataset, rel_dicom_path)
            dicom_files.append(pydicom.dcmread(abs_dicom_path))

        stack_data = [self.process_ct(dcm) for dcm in dicom_files]

        w, h = stack_data[0].shape

        # num_slices = len(stack_data)
        # num_stacks = num_slices // self.channels

        # num_slices = num_stacks * self.channels

        stack_data = torch.stack(stack_data)

        # stack_data = stack_data[:num_slices].view(num_stacks, self.channels, w, h)

        return stack_data.type(torch.float32)

    def get_index_from_map_id(self, map_id: str) -> int:
        return np.where(self.entries["map_id"] == map_id)[0][0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        try:
            image, map_id, slice_thickness = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e

        transformed_image = self.transform(image, slice_thickness)

        return transformed_image, map_id

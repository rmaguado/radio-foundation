import logging
from typing import Optional, Any, Callable
import os
import torch
import numpy as np

from .base import BaseDataset


logger = logging.getLogger("dinov2")


class NpzVolumes(BaseDataset):
    def __init__(self) -> None:
        super().__init__()

    def get_target(self, index: int) -> Optional[Any]:
        return None

    def __len__(self) -> int:
        return len(self.entries)

    def get_image_data(self, index: int) -> torch.Tensor:
        raise NotImplementedError(
            "get_image_data is an abstract method and needs to be implemented."
        )

    def create_entries(self) -> np.ndarray:
        """
        Generates a numpy memmap object pointing to the sqlite database rows.
        For collecting various slices of a CT scan (multi-channel) each memmap row contains the ordered rowids of the slices.

        Returns:
            np.ndarray: The entries dataset (memmap).
        """
        logger.info(f"Creating entries for {self.dataset_name}.")

        slice_stack_num = self.channels

        volumes = self.cursor.execute("SELECT rowid, num_slices FROM global").fetchall()

        logger.info(f"Total number of series: {len(volumes)}.")

        entries = []
        for rowid, num_slices in volumes:

            if num_slices < slice_stack_num:
                continue

            slice_subgroups = [
                (rowid, i) for i in range(num_slices - slice_stack_num + 1)
            ]

            entries += slice_subgroups

        entries_dtype = np.dtype([("rowid", np.int32), ("slice_index", np.int32)])
        entries_array = np.array(entries, dtype=entries_dtype)

        entries_dir = self.get_entries_dir()
        logger.info(f"Saving entries to {entries_dir}.")
        np.save(entries_dir, entries_array)
        return np.load(entries_dir, mmap_mode="r")


class NpzCtDataset(NpzVolumes):

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
        Initializes the NpzCtDataset.

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
        rowid, slice_index = self.entries[index]
        self.cursor.execute(
            """
            SELECT dataset, path FROM global WHERE rowid = ?
            """,
            (int(rowid),),
        )
        dataset, volume_path = self.cursor.fetchone()

        abs_path_to_nifti = os.path.join(self.root_path, dataset, volume_path)
        npz_file = np.load(abs_path_to_nifti)
        volume = npz_file["image_data"]

        try:
            slice_data = volume[slice_index : slice_index + self.channels, :, :].astype(
                np.float32
            )
            slice_shape = slice_data.shape
            assert len(slice_shape) == 3, f"Slice shape is {slice_shape}."

            return self.process_ct(torch.tensor(slice_data, dtype=torch.float32))
        except Exception as e:
            logger.exception(
                f"Error in loading slice {slice_index} from {volume_path}."
            )

            return torch.zeros((self.channels, 512, 512), dtype=torch.float32)

    def process_ct(self, volume_data: torch.Tensor) -> torch.Tensor:
        """
        Processes the CT scan data.

        Args:
            volume_data (torch.Tensor): The CT scan data.

        Returns:
            torch.Tensor: The processed CT scan data.
        """
        return torch.clip(volume_data, self.lower_window, self.upper_window)


class NpzCtVolumesFull(NpzCtDataset):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_image_data(self, index: int) -> torch.Tensor:
        rowid = self.entries[index]
        self.cursor.execute(
            """
            SELECT dataset, volume_path FROM global WHERE rowid = ?
            """,
            (int(rowid),),
        )
        dataset, volume_path = self.cursor.fetchone()

        abs_path_to_nifti = os.path.join(self.root_path, dataset, volume_path)
        npz_file = np.load(abs_path_to_nifti)
        volume = npz_file["image_data"].astype(np.float32)

        return self.process_ct(volume)

    def get_target(self, index: int) -> Optional[Any]:
        raise NotImplementedError(
            "get_target is an abstract method and needs to be implemented."
        )

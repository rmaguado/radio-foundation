import logging
from typing import Optional, Any, Callable
import os
import torch
import numpy as np
import nibabel as nib
import indexed_gzip as igzip

from .base import BaseDataset


logger = logging.getLogger("dinov2")


class NiftiVolumes(BaseDataset):
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
        Generates a numpy memmap object pointing to the sqlite database rows of nifti paths.
        For collecting various slices of a CT scan (multi-channel) each memmap row contains the ordered rowids of the slices.

        Returns:
            np.ndarray: The entries dataset (memmap).
        """
        logger.info(f"Creating entries for {self.dataset_name}.")

        slice_stack_num = self.channels

        nifti_volumes = self.cursor.execute(
            "SELECT rowid, num_slices FROM global"
        ).fetchall()

        logger.info(f"Total number of series: {len(nifti_volumes)}.")

        entries = []
        for rowid, num_slices in nifti_volumes:

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


class NiftiCtDataset(NiftiVolumes):

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
        Initializes the NiftiCtDataset.

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
            SELECT dataset, axial_dim, nifti_path FROM global WHERE rowid = ?
            """,
            (int(rowid),),
        )
        dataset, axial_dim, nifti_path = self.cursor.fetchone()

        abs_path_to_nifti = os.path.join(self.root_path, dataset, nifti_path)

        # will automatically use indexed_gzip if nibabel>=2.3.0 is present (much faster than without)
        nifti_file = nib.load(abs_path_to_nifti)

        try:
            slope = nifti_file.dataobj.slope
            intercept = nifti_file.dataobj.inter

            if not slope or np.isnan(slope):
                slope = 1.0
            if not intercept or np.isnan(intercept):
                intercept = 0.0

        except AttributeError as e:
            slope = 1.0
            intercept = 0.0

        slice_obj = [slice(None)] * 3
        slice_obj[axial_dim] = slice(slice_index, slice_index + self.channels)
        slice_obj = tuple(slice_obj)

        try:
            slice_data = nifti_file.dataobj[slice_obj].astype(np.float32)
            slice_data = np.moveaxis(slice_data, axial_dim, 0)
            slice_data = slice_data * slope + intercept

            slice_shape = slice_data.shape
            assert len(slice_shape) == 3, f"Slice shape is {slice_shape}."

            return self.process_ct(torch.tensor(slice_data, dtype=torch.float32))
        except Exception as e:
            logger.exception(f"Error in loading slice {slice_index} from {nifti_path}.")

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


class NiftiCtVolumesFull(NiftiCtDataset):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_image_data(self, index: int) -> torch.Tensor:
        rowid = self.entries[index]
        self.cursor.execute(
            """
            SELECT dataset, axial_dim, nifti_path FROM global WHERE rowid = ?
            """,
            (int(rowid),),
        )
        dataset, axial_dim, nifti_path = self.cursor.fetchone()

        abs_path_to_nifti = os.path.join(self.root_path, dataset, nifti_path)
        nifti_file = nib.load(abs_path_to_nifti)

        volume_data = nifti_file.get_fdata().astype(np.float32)
        volume_data = np.moveaxis(volume_data, axial_dim, 0)

        return self.process_ct(torch.tensor(volume_data, dtype=torch.float32))

    def get_target(self, index: int) -> Optional[Any]:
        raise NotImplementedError(
            "get_target is an abstract method and needs to be implemented."
        )

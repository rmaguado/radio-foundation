import os
import torch
from typing import Tuple, Any
import numpy as np
import pydicom
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import logging
import pandas as pd

from dinov2.data.samplers import InfiniteSampler
from dinov2.data.datasets.dicoms import DicomCTVolumesFull
from torch.utils.data import DataLoader

logger = logging.getLogger("dinov2")


def get_dataloader(dataset, channels, split="train") -> DataLoader:

    def collate_fn(inputs):
        img = inputs[0][0]
        label = inputs[0][1]

        num_slices = img.shape[0]
        num_batches = num_slices // channels
        use_slice = num_batches * channels

        images = img[:use_slice].view(num_batches, channels, *img.shape[1:])

        return images, label

    loader_kwargs = {
        "batch_size": 1,
        "pin_memory": True,
        "collate_fn": collate_fn,
        "num_workers": 0,
    }

    if split == "train":
        sampler = InfiniteSampler(sample_count=len(dataset))
        return torch.utils.data.DataLoader(dataset, sampler=sampler, **loader_kwargs)
    elif split == "val":
        return torch.utils.data.DataLoader(dataset, **loader_kwargs)
    else:
        raise ValueError("train")


class DeepRDTSplit:
    def __init__(
        self,
        metadata_path: str,
        root_path: str,
        transform,
        max_workers: int,
        split: str = "train",
        train_val_split: float = 0.8,
    ):
        self.dataset = DeepRDT_Responses(
            metadata_path=metadata_path,
            root_path=root_path,
            transform=transform,
            max_workers=max_workers
        )

        self.train_len = int(len(self.dataset) * train_val_split)
        self.val_len = len(self.dataset) - self.train_len
        if split == "train":
            self.get_function = lambda index: self.dataset[index]
            self.subset_len = self.train_len
        elif split == "val":
            self.get_function = lambda index: self.dataset[index + self.train_len]
            self.subset_len = self.val_len
        else:
            raise ValueError('split should be either "train" or "val"')

    def __len__(self):
        return self.subset_len

    def __getitem__(self, index):
        return self.get_function(index)


class DeepRDT_Responses(DicomCTVolumesFull):
    def __init__(self, metadata_path: str, root_path: str, transform, max_workers: int):
        super().__init__(
            dataset_name="DeepRDT-lung",
            root_path=root_path,
            transform=transform,
        )
        self.max_workers = max_workers

        self.metadata = pd.read_csv(metadata_path)

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
            ("mapid", "U256"),
        ]
        entries = []
        for dataset_name in dataset_names:
            dataset_name = dataset_name[0]
            dataset_series = self.cursor.execute(
                f"SELECT rowid, mapid FROM '{dataset_name}'"
            ).fetchall()
            entries += [
                (dataset_name, rowid, mapid) for rowid, mapid in dataset_series
            ]
        logger.info(f"Total number of scans: {len(entries)}.")

        entries_array = np.array(entries, dtype=entries_dtype)

        entries_dir = self.get_entries_dir()
        logger.info(f"Saving entries to {entries_dir}.")
        np.save(entries_dir, entries_array)
        return np.load(entries_dir, mmap_mode="r")

    def get_image_data(self, index: int) -> Tuple[torch.Tensor, str]:
        dataset_name, rowid, mapid = self.entries[index]

        self.cursor.execute(
            f"SELECT series_id, mapid FROM '{dataset_name}' WHERE rowid = {rowid}"
        )
        series_id, mapid = self.cursor.fetchone()

        self.cursor.execute(
            """
            SELECT slice_index, dataset, dicom_path
            FROM global 
            WHERE series_id = ? 
            AND dataset = ?
            """,
            (series_id, dataset_name),
        )
        stack_rows = self.cursor.fetchall()
        stack_rows.sort(key=lambda x: x[0])

        try:
            stack_data = self.create_stack_data(stack_rows)
        except Exception as e:
            logger.exception(f"Error processing stack. Seriesid: {series_id} \n{e}")
            stack_data = torch.zeros((10, 512, 512))

        return stack_data, mapid

    @staticmethod
    def load_dicom(row, root_path):
        _, dataset, rel_dicom_path = row
        abs_dicom_path = os.path.join(root_path, dataset, rel_dicom_path)
        dcm = pydicom.dcmread(abs_dicom_path)
        return dcm

    def create_stack_data(self, stack_rows):
        load_dicom_partial = partial(self.load_dicom, root_path=self.root_path)

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            dicom_files = list(executor.map(load_dicom_partial, stack_rows))

        stack_data = [self.process_ct(dcm) for dcm in dicom_files]

        return torch.stack(stack_data)

    def get_target(self, mapid: str) -> torch.Tensor:

        matches = self.metadata[self.metadata["MAPID"] == int(mapid)]["respuesta"]

        if matches.shape[0] != 1:
            raise ValueError(f"Expected exactly one match for MAPID={mapid}, but found {matches.shape[0]}.")
        
        target = matches.iloc[0]

        return target

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        try:
            image, mapid = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e

        target = self.get_target(mapid)

        return self.transform(image, target)

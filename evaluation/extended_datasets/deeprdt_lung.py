import os
import torch
from typing import Tuple, Any
import numpy as np
import pydicom
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import logging
import pandas as pd

from dinov2.data.datasets.dicoms import DicomCTVolumesFull
from torch.utils.data import DataLoader

logger = logging.getLogger("dinov2")


def get_dataloaders(dataset_kwargs, channels, train_val_split=0.9) -> DataLoader:

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

    dataset = DeepRDT_Responses(**dataset_kwargs)

    total_entries = len(dataset)
    labels = dataset.entries["response"]

    indexes = np.arange(total_entries)
    positive_indexes = indexes[labels]
    negative_indexes = indexes[~labels]

    np.random.shuffle(positive_indexes)
    np.random.shuffle(negative_indexes)

    num_positives = np.sum(labels)
    num_negatives = total_entries - num_positives

    train_positives = int(num_positives * train_val_split)
    train_negatives = int(num_negatives * train_val_split)

    train_positives_indexes = positive_indexes[:train_positives]
    val_positives_indexes = positive_indexes[train_positives:]

    train_negatives_indexes = negative_indexes[:train_negatives]
    val_negatives_indexes = negative_indexes[train_negatives:]

    def get_loader(indexes):
        subset = DeepRDTSplit(dataset, indexes)
        torch.utils.data.DataLoader(subset, **loader_kwargs)

    return {
        "train_positives": get_loader(train_positives_indexes),
        "val_positives": get_loader(val_positives_indexes),
        "train_negatives": get_loader(train_negatives_indexes),
        "val_negatives": get_loader(val_negatives_indexes),
    }


class DeepRDTSplit:
    def __init__(
        self,
        dataset,
        indexes,
    ):
        self.dataset = dataset
        self.indexes = indexes
        self.subset_len = len(indexes)

    def __len__(self):
        return self.subset_len

    def __getitem__(self, index):
        return self.dataset[self.indexes[index]]


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
            ("response", np.bool),
        ]
        entries = []
        for dataset_name in dataset_names:
            dataset_name = dataset_name[0]
            dataset_series = self.cursor.execute(
                f"SELECT rowid, mapid FROM '{dataset_name}'"
            ).fetchall()

            for rowid, mapid in dataset_series:

                matches = self.metadata[self.metadata["MAPID"] == int(mapid)][
                    "respuesta"
                ]
                if matches.shape[0] != 1:
                    raise ValueError(
                        f"Expected exactly one match for MAPID={mapid}, but found {matches.shape[0]}."
                    )
                response_text = matches.iloc[0]

                # 1-Completa, 2-Parcial, 3-Estable, 4-Progresion
                response = response_text in ["1-Completa", "2-Parcial"]

                entries.append((dataset_name, rowid, mapid, response))

        logger.info(f"Total number of scans: {len(entries)}.")

        entries_array = np.array(entries, dtype=entries_dtype)

        entries_dir = self.get_entries_dir()
        logger.info(f"Saving entries to {entries_dir}.")
        np.save(entries_dir, entries_array)
        return np.load(entries_dir, mmap_mode="r")

    def get_image_data(self, index: int) -> Tuple[torch.Tensor, str]:
        dataset_name, rowid, mapid, response = self.entries[index]

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

        return stack_data, mapid, response

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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        try:
            image, mapid, response = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e

        transformed_image = self.transform(image)

        return transformed_image, response

import os
import torch
from typing import Tuple, Any
import numpy as np
import pydicom
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import logging

from dinov2.data.samplers import InfiniteSampler
from dinov2.data.datasets.dicoms import DicomCTVolumesFull
from torch.utils.data import DataLoader

logger = logging.getLogger("dinov2")


def get_lidcidri_loader(dataset, channels) -> DataLoader:

    def collate_fn(inputs):
        img = inputs[0][0]
        labels = inputs[0][1]

        num_slices = img.shape[0]
        num_batches = num_slices // channels
        use_slice = num_batches * channels

        images = img[:use_slice].view(num_batches, channels, *img.shape[1:])
        labels = labels[:use_slice].view(num_batches, channels, *labels.shape[1:])

        return images, labels

    loader_kwargs = {
        "batch_size": 1,
        "pin_memory": True,
        "collate_fn": collate_fn,
        "num_workers": 0,
    }

    sampler = InfiniteSampler(sample_count=len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, **loader_kwargs)
    return iter(dataloader)


def rle_decode(run_lengths, shape):
    pos = np.cumsum(run_lengths)
    decoded = np.zeros(np.prod(shape), dtype=np.int32)
    current_value = 1
    for i in range(len(pos)):
        decoded[pos[i] : pos[i + 1] if i + 1 < len(pos) else None] = current_value
        current_value = 1 - current_value

    return decoded.reshape(shape)


class LidcIdriTrain:
    def __init__(self, mask_path: str, root_path: str, transform, split: float = 0.8):
        self.dataset = LidcIdriNodules(mask_path, root_path, transform)

        self.index_split = int(len(self.dataset) * split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class LidcIdriVal:
    def __init__(self, mask_path: str, root_path: str, transform, split: float = 0.8):
        self.dataset = LidcIdriNodules(mask_path, root_path, transform)

        self.index_split = int(len(self.dataset) * split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[self.index_split + index]


class LidcIdriNodules(DicomCTVolumesFull):
    def __init__(self, mask_path: str, root_path: str, transform):
        super().__init__(
            dataset_name="LIDC-IDRI",
            root_path=root_path,
            transform=transform,
        )

        self.root_mask_path = mask_path

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
            ("scan_id", np.uint32),
        ]
        entries = []
        for dataset_name in dataset_names:
            dataset_name = dataset_name[0]
            dataset_series = self.cursor.execute(
                f"SELECT rowid, scan_id FROM '{dataset_name}'"
            ).fetchall()
            entries += [
                (dataset_name, rowid, scan_id) for rowid, scan_id in dataset_series
            ]
        logger.info(f"Total number of scans: {len(entries)}.")

        entries_array = np.array(entries, dtype=entries_dtype)

        entries_dir = self.get_entries_dir()
        logger.info(f"Saving entries to {entries_dir}.")
        np.save(entries_dir, entries_array)
        return np.load(entries_dir, mmap_mode="r")

    def get_image_data(self, index: int) -> torch.Tensor:
        dataset_name, rowid, scan_id = self.entries[index]

        self.cursor.execute(
            f"SELECT series_id, scan_id FROM '{dataset_name}' WHERE rowid = {rowid}"
        )
        series_id, scan_id = self.cursor.fetchone()

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

        return stack_data, scan_id

    @staticmethod
    def load_dicom(row, root_path):
        _, dataset, rel_dicom_path = row
        abs_dicom_path = os.path.join(root_path, dataset, rel_dicom_path)
        dcm = pydicom.dcmread(abs_dicom_path)
        return dcm

    def create_stack_data(self, stack_rows):
        load_dicom_partial = partial(self.load_dicom, root_path=self.root_path)

        with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            dicom_files = list(executor.map(load_dicom_partial, stack_rows))

        stack_data = [self.process_ct(dcm) for dcm in dicom_files]

        return torch.stack(stack_data)

    def get_target(self, scanid: str) -> torch.Tensor:

        mask_path = os.path.join(self.root_mask_path, f"{scanid}.npz")

        with np.load(mask_path) as data:
            rle_mask = data["rle_mask"]
            shape = data["shape"]

        mask = rle_decode(rle_mask, shape)

        return torch.tensor(mask, dtype=torch.float32)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        try:
            image, scanid = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e

        target = self.get_target(scanid)

        return self.transform(image, target)

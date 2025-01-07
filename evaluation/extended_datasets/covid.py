import os
import torch
from typing import Tuple, Any
import numpy as np
import logging

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
    }

    if split == "train":
        sampler = InfiniteSampler(sample_count=len(dataset))
        return torch.utils.data.DataLoader(dataset, sampler=sampler, **loader_kwargs)
    elif split == "val":
        return torch.utils.data.DataLoader(dataset, **loader_kwargs)
    else:
        raise ValueError("train")


class CovidDataset(DicomCTVolumesFull):
    def __init__(self, root_path: str, transform, covid_label: bool):
        super().__init__(
            dataset_name="LIDC-IDRI",
            root_path=root_path,
            transform=transform,
        )

        self.covid_label = covid_label

    def get_target(self, index: int) -> Optional[Any]:
        return self.covid_label

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import csv
from enum import Enum
import logging
import os
from typing import Callable, List, Optional, Tuple, Union, Any

from PIL import Image
import numpy as np
import pydicom

from .extended import ExtendedVisionDataset


logger = logging.getLogger("dinov2")
_Target = int

class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        # slices
        split_lengths = {
            _Split.TRAIN: 193_615,
            _Split.VAL: 24_331,
            _Split.TEST: 26_012,
        }
        return split_lengths[self]

    def get_dirname(self, volume_id: Optional[str] = None) -> str:
        return self.value if volume_id is None else os.path.join(self.value, volume_id)

    def get_image_relpath(self, slice_index: int, figures: int, acquisition_number: int, volume_id: str) -> str:
        dirname = self.get_dirname(volume_id)
        basename = f"{acquisition_number}-{str(slice_index).rjust(figures, '0')}.dcm"
        return os.path.join(dirname, basename)

    def parse_image_relpath(self, image_relpath: str) -> Tuple[str, int]:
        dirname, filename = os.path.split(image_relpath)
        volume_id = os.path.split(dirname)[-1]
        basename, _ = os.path.splitext(filename)
        slice_index = int(basename.split("-")[-1])
        return volume_id, slice_index


class LidcIdri(ExtendedVisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "LidcIdri.Split",
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split

        self._entries = None
        self._class_ids = None
        self._class_names = None

    @property
    def split(self) -> "LidcIdri.Split":
        return self._split

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        extra_full_path = self._get_extra_full_path(extra_path)
        os.makedirs(self._extra_root, exist_ok=True)
        np.save(extra_full_path, extra_array)

    @property
    def _entries_path(self) -> str:
        return f"entries-{self._split.value.upper()}.npy"

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    def get_image_data(self, index: int) -> np.ndarray:
        entries = self._get_entries()
        series_id = entries[index]["series_id"]
        acquisition_number = entries[index]["acquisition_number"]
        figures = entries[index]["figures"]
        slice_index = entries[index]["slice_index"]

        image_relpath = self.split.get_image_relpath(slice_index, figures, acquisition_number, series_id)
        image_full_path = os.path.join(self.root, image_relpath)
        dicom_data = pydicom.dcmread(image_full_path)
        
        image_data = dicom_data.pixel_array
        rescale_slope = dicom_data.RescaleSlope
        rescale_intercept = dicom_data.RescaleIntercept

        hu_array = image_data * rescale_slope + rescale_intercept

        windowed_hu = np.clip(hu_array, -1024, 3071)

        rescaled = (windowed_hu + 1024) / 4095
        
        image = Image.fromarray(rescaled, 'L')

        return image

    def get_target(self, index: int) -> Optional[Target]:
        entries = self._get_entries()
        
        # create volume with zeros
        
        # iterate over all nodues, add them to the volume
        
        # take a slice

        return None

    def get_targets(self) -> Optional[np.ndarray]:
        entries = self._get_entries()
        return None # [self.get_target(i) for i in range(len(entries))]

    def __len__(self) -> int:
        entries = self._get_entries()
        assert len(entries) == self.split.length
        return len(entries)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = None # self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def _dump_entries(self) -> None:
        split = self.split

        series_dirs = [
            d for d in os.listdir(os.path.join(self.root, split.value)) \
            if os.path.isdir(os.path.join(self.root, split.value, d))
        ]

        dtype = np.dtype(
            [
                ("series_id", "U256"),
                ("acquisition_number", "uint8"),
                ("figures", "uint8"),
                ("slice_index", "uint16"),
            ]
        )
        entries_array = np.empty(self.split.length, dtype=dtype)
        
        abs_slice_index = 0
        for series_id in series_dirs:
            slice_files = [
                f for f in os.listdir(os.path.join(self.root, split.value, series_id)) \
                if f.endswith('.dcm')
            ]
            for slice_file in slice_files:
                slice_labels = os.path.splitext(slice_file)[0].split('_')[-1].split("-")
                acquisition_number = int(slice_labels[0])
                figures = len(slice_labels[1])
                slice_index = int(slice_labels[1])
                entries_array[abs_slice_index] = (
                    series_id,
                    acquisition_number,
                    figures,
                    slice_index
                )
                abs_slice_index += 1

        logger.info(f'saving entries to "{self._entries_path}"')
        self._save_extra(entries_array, self._entries_path)

    def dump_extra(self) -> None:
        self._dump_entries()

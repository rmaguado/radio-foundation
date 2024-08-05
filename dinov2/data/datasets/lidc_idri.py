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
import json
import h5py

from .ct_dataset import CtDataset


logger = logging.getLogger("dinov2")
_Target = int


class LidcIdri(CtDataset):
    Target = Union[_Target]

    def __init__(
        self,
        *,
        split: "LidcIdri.Split",
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        enable_targets: bool = False
    ) -> None:
        super().__init__(
            split=split,
            root=root,
            extra=extra,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            enable_targets=enable_targets
        )

    def get_target(self, index: int) -> Optional[Target]:
        entries = self._get_entries()
        
        series_id = entries[index]["series_id"]
        slice_index = entries[index]["slice_index"]

        mask_full_path = os.path.join(self.root, series_id, 'mask.h5')
        
        with h5py.File(mask_full_path, 'r') as f:
            data = f["data"]
            loaded_mask = data[slice_index]

        return Image.fromarray(loaded_mask)

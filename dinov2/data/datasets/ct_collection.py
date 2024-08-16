# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Callable, Optional, Any

import torch
import numpy as np
import h5py

from .ct_dataset import CtDataset

logger = logging.getLogger("dinov2")


class CtCollection(CtDataset):

    def __init__(
        self,
        *,
        split: str,
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


    def get_image_data(self, index: int) -> np.ndarray:
        entries = self._get_entries()
        dataset = entries[index]["dataset"]
        series_id = entries[index]["series_id"]
        slice_index = entries[index]["slice_index"]
        
        image_full_path = os.path.join(self.root, dataset, "data", series_id, "image.h5")
        
        with h5py.File(image_full_path, 'r') as f:
            data = f["data"]
            image = data[slice_index].astype(np.float32)
            
        return torch.from_numpy(image).unsqueeze(0)

    def get_target(self, index: int) -> Optional[Any]:
        raise NotImplementedError

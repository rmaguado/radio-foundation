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

from .extended import ExtendedVisionDataset


logger = logging.getLogger("dinov2")
_Target = int


class NSCLC_Radiomics(ExtendedVisionDataset):
    Target = Union[_Target]

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
            split, root, extra, transforms, transform, target_transform, enable_targets
        )
        
    def get_target(self, index: int) -> Optional[Target]:
        raise NotImplementedError

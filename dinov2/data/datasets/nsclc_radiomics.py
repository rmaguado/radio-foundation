# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import Callable, Optional, Any

from .ct_dataset import CtDataset

logger = logging.getLogger("dinov2")


class NsclcRadiomics(CtDataset):

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
        
    def get_target(self, index: int) -> Optional[Any]:
        raise NotImplementedError

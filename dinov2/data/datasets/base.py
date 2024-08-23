# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Any, Tuple, Optional
import torch


class BaseDataset:
    def __init__(self):
        self.transform = lambda _: _
        self.target_transform = lambda _: _

    def get_image_data(self, index: int) -> torch.tensor:
        raise NotImplementedError

    def get_target(self, index: int) -> Optional[Any]:
        raise NotImplementedError

    def get_targets(self) -> Optional[any]:
        return [self.get_target(i) for i in len(self)]

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, image, target):
        return self.transform(image), self.target_transform(target)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e

        target = self.get_target(index)

        return self.apply_transforms(image, target)

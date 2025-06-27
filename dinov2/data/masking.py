# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import random
import math
import numpy as np
from typing import Optional


class MaskingGenerator:
    def __init__(
        self,
        input_size: int | tuple[int, int],
        num_masking_patches: Optional[int] = None,
        min_num_patches: int = 4,
        max_num_patches: int = None,  # type: ignore[assignment]
        min_aspect: float = 0.3,
        max_aspect: Optional[float] = None,
    ):
        """
        Masking generator for DINO. This generator creates a mask for the input image.


        Args:
            input_size (int or tuple): The size of the input.
            num_masking_patches (int): The number of patches to mask.
            min_num_patches (int): The minimum number of patches to mask.
            max_num_patches (int): The maximum number of patches to mask.
            min_aspect (float): The minimum aspect ratio.
            max_aspect (float): The maximum aspect ratio.
        """
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = (
            num_masking_patches if max_num_patches is None else max_num_patches
        )

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask: np.ndarray, max_mask_patches: int):
        """
        Mask the image with random patches

        Args:
            mask (np.ndarray): The mask to apply.
            max_mask_patches (int): The maximum number of patches to mask.
        """
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches: int = 0):
        """
        Generate a mask for the input image.

        Args:
            num_masking_patches (int): The number of patches to mask.

        Returns:
            np.ndarray: The mask.
        """
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask

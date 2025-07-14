import random
import math
import numpy as np
from typing import Optional, Sequence


class MaskingGenerator:
    """
    Generates a mask composed of multiple, smaller, non-overlapping blocks.

    This generator gives control over both the total percentage of the area
    to be masked and the size of the individual blocks used to create the mask.

    Args:
        min_aspect (float): Minimum aspect ratio for the mask blocks.
        max_aspect (Optional[float]): Maximum aspect ratio for the mask blocks. If None, set to 1/min_aspect.
    """

    def __init__(
        self,
        min_aspect: float = 0.3,
        max_aspect: Optional[float] = None,
    ):
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __call__(
        self,
        input_shape: Sequence[int],
        mask_ratio: float = 0.5,
    ) -> np.ndarray:
        """
        Generates the mask by iteratively adding small blocks.

        Args:
            input_shape (Sequence[int]): The shape of the input to be masked (e.g., [14, 14]).
            mask_ratio (float): The fraction of the total area to be masked (default: 0.5).

        Returns:
            np.ndarray: A boolean numpy array with multiple masked blocks.
        """
        num_total_patches = int(np.prod(input_shape))

        total_num_to_mask = int(num_total_patches * mask_ratio)

        mask = np.zeros(input_shape, dtype=bool)
        num_masked = 0

        max_attempts = total_num_to_mask * 10
        attempts = 0

        while num_masked < total_num_to_mask and attempts < max_attempts:
            attempts += 1

            block_mask = self._generate_single_block(input_shape, num_total_patches)

            new_mask = np.logical_or(mask, block_mask)

            newly_masked_count = np.sum(new_mask) - num_masked

            if newly_masked_count > 0:
                mask = new_mask
                num_masked += newly_masked_count

        return mask

    def _generate_single_block(
        self, input_shape: Sequence[int], num_total_patches: int
    ) -> np.ndarray:
        """
        Helper function to generate one small mask block.

        Args:
            input_shape (Sequence[int]): The shape of the input to be masked.
            num_total_patches (int): Total number of patches in the input.

        Returns:
            np.ndarray: A boolean numpy array representing a single mask block.
        """
        block_target_volume = num_total_patches * random.uniform(0, 0.5)

        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))

        n_dims = len(input_shape)
        base_side = block_target_volume ** (1 / n_dims)

        dims_float = [base_side] * n_dims
        dims_float[0] = base_side / math.sqrt(aspect_ratio)
        dims_float[1] = base_side * math.sqrt(aspect_ratio)

        side_lengths = [
            min(max_dim, max(1, int(round(dim_float))))
            for dim_float, max_dim in zip(dims_float, input_shape)
        ]

        starts = [
            random.randint(0, max_dim - side)
            for max_dim, side in zip(input_shape, side_lengths)
        ]

        block_mask = np.zeros(input_shape, dtype=bool)
        slices = tuple(
            slice(start, start + side) for start, side in zip(starts, side_lengths)
        )
        block_mask[slices] = True

        return block_mask

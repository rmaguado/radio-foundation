import random
import math
import numpy as np
from typing import Optional, Sequence


class MaskingGenerator:
    """
    Generates a mask composed of multiple, smaller, non-overlapping blocks.

    This generator works for 2D images and 3D (or higher-dimensional) volumes.

    Args:
        min_aspect (float): Minimum aspect ratio for the mask blocks.
        max_aspect (Optional[float]): Maximum aspect ratio for the mask blocks.
                                      If None, set to 1/min_aspect.
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
            input_shape (Sequence[int]): Shape of the input (e.g., [14, 14] or [16, 14, 14]).
            mask_ratio (float): Fraction of the total area to be masked (default: 0.5).

        Returns:
            np.ndarray: A boolean numpy array with multiple masked blocks.
        """
        num_total_patches = int(np.prod(input_shape))
        total_num_to_mask = int(num_total_patches * mask_ratio)

        mask = np.zeros(input_shape, dtype=bool)
        num_masked = 0

        max_attempts = total_num_to_mask * 2
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
        """
        n_dims = len(input_shape)

        block_target_volume = num_total_patches * random.uniform(0, 0.5)

        base_side = block_target_volume ** (1 / n_dims)

        log_ratios = np.random.uniform(*self.log_aspect_ratio, size=n_dims)
        scales = np.exp(log_ratios)

        scales = scales / (np.prod(scales) ** (1 / n_dims))

        dims_float = [base_side * s for s in scales]
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def visualize_3d_mask(mask: np.ndarray, title="3D Mask"):
        assert mask.ndim == 3, "Only for 3D masks"
        d, h, w = mask.shape

        slices = [
            mask[d // 2, :, :],  # axial
            mask[:, h // 2, :],  # coronal
            mask[:, :, w // 2],  # sagittal
        ]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, slc, name in zip(axes, slices, ["Axial", "Coronal", "Sagittal"]):
            ax.imshow(slc, cmap="gray")
            ax.set_title(name)
            ax.axis("off")
        plt.suptitle(title)
        plt.savefig("example3d")

    gen = MaskingGenerator(min_aspect=0.3)

    mask2d = gen([32, 32], mask_ratio=0.4)
    print(mask2d.mean())
    plt.imshow(mask2d, cmap="gray")
    plt.title("2D Mask")
    plt.axis("off")
    plt.savefig("example2d")

    mask3d = gen([32, 32, 32], mask_ratio=0.4)
    print(mask3d.mean())
    visualize_3d_mask(mask3d, title="3D Mask (32³)")

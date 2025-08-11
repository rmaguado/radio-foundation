import torch
import random
from typing import Tuple, List, Callable, Optional
from torchvision.transforms.functional import gaussian_blur
import logging

logger = logging.getLogger("dinov2")


class AnisotropicCrop:
    def __init__(self, size: int, scale: Tuple[float, float] = (0.7, 1.0)) -> None:
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"size must be a positive integer, but got {size}")
        if not (0 < scale[0] <= scale[1]):
            raise ValueError(
                f"Scale range must be positive and ordered, but got {scale}"
            )
        self.size = size
        self.crop_size = (size, size, size)
        self.scale = scale

    def _get_scale(self) -> float:
        """Generates a random scale factor from the specified range."""
        return torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()

    def __call__(
        self, img: torch.Tensor, spacing: Tuple[float, float, float]
    ) -> torch.Tensor:
        raise NotImplementedError


class BaseRandomCrop:
    """
    An abstract base class for random cropping operations.

    This class provides the core logic for calculating random crop parameters based on
    a specified scale range. Subclasses must implement the __call__ method to define
    the specific cropping and resampling behavior.
    """

    def __init__(self, scale: Tuple[float, float]) -> None:
        if not (0 < scale[0] <= scale[1]):
            raise ValueError(
                f"Scale range must be positive and ordered, but got {scale}"
            )
        self.scale = scale

    def _get_scale(self) -> float:
        """Generates a random scale factor from the specified range."""
        return torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()

    def _get_crop_params(
        self, img_shape: Tuple[int, ...]
    ) -> Tuple[List[int], List[int]]:
        """
        Calculates the start and end indices for a random crop.

        Args:
            img_shape: The shape of the image dimensions to be cropped.

        Returns:
            A tuple containing two lists: the start indices and end indices for the crop.
        """
        scale = self._get_scale()
        scaled_shape = [int(dim * scale) for dim in img_shape]

        crop_dims = [max(1, dim) for dim in scaled_shape]
        crop_dims = [min(img_shape[i], crop_dims[i]) for i in range(len(img_shape))]

        max_starts = [img_shape[i] - crop_dims[i] for i in range(len(img_shape))]

        starts = [random.randint(0, max_s) for max_s in max_starts]
        ends = [starts[i] + crop_dims[i] for i in range(len(img_shape))]

        return starts, ends

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """The main transformation method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the `__call__` method.")


class RandomCrop3D(BaseRandomCrop):
    """
    Crops a 3D image to a random size and resamples it to a target cubic 3D size.

    Takes a 3D tensor (D, H, W) and produces a cubic 3D tensor of (`size`, `size`, `size`).
    """

    def __init__(self, size: int, scale: Tuple[float, float] = (0.7, 1.0)) -> None:
        super().__init__(scale)
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"size must be a positive integer, but got {size}")
        self.crop_size = (size, size, size)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim != 3:
            raise ValueError(
                f"Input image must be 3D (D, H, W), but got shape {img.shape}"
            )

        min_spatial_dim = min(img.shape)

        min_required_size = int(max(self.crop_size) / 2)
        if min_spatial_dim * self.scale[0] >= min_required_size:
            scale = self._get_scale()
        elif min_spatial_dim >= min_required_size:
            scale = 1.0
        else:
            scale = 1.0
            logger.warning(
                f"Minimum spatial dimension {min_spatial_dim} is smaller than "
                f"the minimum required size for a cubic crop of {min_required_size}. "
                "The resulting crop may be upscaled or smaller than intended."
            )

        crop_dim = int(min_spatial_dim * scale)

        crop_dim = min(min_spatial_dim, max(1, crop_dim))

        max_start_d = img.shape[0] - crop_dim
        start_d = random.randint(0, max_start_d)
        max_start_h = img.shape[1] - crop_dim
        start_h = random.randint(0, max_start_h)
        max_start_w = img.shape[2] - crop_dim
        start_w = random.randint(0, max_start_w)

        cropped_img = img[
            start_d : start_d + crop_dim,
            start_h : start_h + crop_dim,
            start_w : start_w + crop_dim,
        ].float()

        resampled_img = torch.nn.functional.interpolate(
            cropped_img.unsqueeze(0).unsqueeze(0),
            size=self.crop_size,
            mode="trilinear",
            align_corners=False,
        )

        return resampled_img.squeeze(0).squeeze(0)


class RandomSliceCrop(BaseRandomCrop):
    """
    Extracts a thick 2D slice from a 3D image along a random axis and
    crops it, producing a square 2.5D output.
    """

    def __init__(
        self,
        size: int,
        channels: int,
        scale: Tuple[float, float] = (0.7, 1.0),
    ) -> None:
        super().__init__(scale)
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"size must be a positive integer, but got {size}")
        self.crop_size = (size, size)
        self.channels = channels

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim != 3:
            raise ValueError(
                f"Input image must be 3D (D, H, W), but got shape {img.shape}"
            )

        img_shape = img.shape

        min_required_size = int(max(self.crop_size) / 2)
        too_short_axes = [
            i for i, dim_size in enumerate(img_shape) if dim_size < min_required_size
        ]

        if len(too_short_axes) > 1:
            logger.warning(
                f"Image has more than one dimension shorter than the required spatial crop size "
                f"of {min_required_size}: {img_shape}. A valid 2D plane cannot be formed."
            )

        if too_short_axes:
            slice_axis = too_short_axes[0]
        else:
            slice_axis = random.choice([0, 1, 2])

        spatial_axes = [i for i in range(3) if i != slice_axis]

        if img_shape[slice_axis] < self.channels:
            raise ValueError(
                f"The chosen slice axis (index {slice_axis}) has size {img_shape[slice_axis]}, "
                f"which is smaller than the required depth of {self.channels}."
            )

        min_spatial_dim = min([img_shape[i] for i in spatial_axes])

        if min_spatial_dim * self.scale[0] >= min_required_size:
            scale = self._get_scale()
        elif min_spatial_dim >= min_required_size:
            scale = 1.0
        else:
            scale = 1.0
            logger.warning(
                f"Minimum spatial dimension {min_spatial_dim} is smaller than "
                f"the minimum required size for a cubic crop of {min_required_size}. "
                "The resulting crop may be upscaled or smaller than intended."
            )

        max_start = img_shape[slice_axis] - self.channels
        start_idx = random.randint(0, max_start)

        slicer = [slice(None)] * 3
        slicer[slice_axis] = slice(start_idx, start_idx + self.channels)
        sub_volume = img[tuple(slicer)]

        spatial_shape = (img_shape[spatial_axes[0]], img_shape[spatial_axes[1]])

        min_spatial_dim = min(spatial_shape)

        scaled_crop_dim = int(min_spatial_dim * scale)
        crop_dim = min(min_spatial_dim, max(1, scaled_crop_dim))

        max_start_0 = spatial_shape[0] - crop_dim
        start_0 = random.randint(0, max_start_0)
        max_start_1 = spatial_shape[1] - crop_dim
        start_1 = random.randint(0, max_start_1)

        spatial_slicer = [slice(None)] * 3
        spatial_slicer[slice_axis] = slice(None)
        spatial_slicer[spatial_axes[0]] = slice(start_0, start_0 + crop_dim)
        spatial_slicer[spatial_axes[1]] = slice(start_1, start_1 + crop_dim)

        cropped_plane = sub_volume[tuple(spatial_slicer)]
        cropped_plane = cropped_plane.movedim(slice_axis, 0).float()

        resampled_img = torch.nn.functional.interpolate(
            cropped_plane.unsqueeze(0),
            size=self.crop_size,
            mode="bilinear",
            align_corners=False,
        )

        return resampled_img.squeeze(0)


class RandomCrop2D(BaseRandomCrop):
    """
    Crops a 2D image (or 2.5D) to a random size and resamples it to a target square 2D size.

    Takes a tensor (C, H, W) and produces a tensor of (`C`, `size`, `size`).
    """

    def __init__(self, size: int, scale: Tuple[float, float] = (0.7, 1.0)) -> None:
        super().__init__(scale)
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"size must be a positive integer, but got {size}")
        self.crop_size = (size, size)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim != 3:
            raise ValueError(
                f"Input image must be 2.5D (C, H, W), but got shape {img.shape}"
            )

        spatial_shape = img.shape[1:]
        min_required_size = int(max(self.crop_size) / 2)

        min_spatial_dim = min(img.shape[1:])

        if min_spatial_dim * self.scale[0] >= min_required_size:
            scale = self._get_scale()
        elif min_spatial_dim >= min_required_size:
            scale = 1.0
        else:
            scale = 1.0
            logger.warning(
                f"Minimum spatial dimension {min_spatial_dim} is smaller than "
                f"the minimum required size for a cubic crop of {min_required_size}. "
                "The resulting crop may be upscaled or smaller than intended."
            )

        min_spatial_dim = min(spatial_shape)

        crop_dim = int(min_spatial_dim * scale)

        crop_dim = min(min_spatial_dim, max(1, crop_dim))

        max_start_h = spatial_shape[0] - crop_dim
        start_h = random.randint(0, max_start_h)
        max_start_w = spatial_shape[1] - crop_dim
        start_w = random.randint(0, max_start_w)

        cropped_img = img[
            :, start_h : start_h + crop_dim, start_w : start_w + crop_dim
        ].float()

        resampled_img = torch.nn.functional.interpolate(
            cropped_img.unsqueeze(0),
            size=self.crop_size,
            mode="bilinear",
            align_corners=False,
        )

        return resampled_img.squeeze(0)


class Resize:
    def __init__(self, output_size: Tuple[int, int, int]) -> None:
        self.output_size = output_size

    def _resize_2d(self, img: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=self.output_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _resize_3d(self, img: torch.Tensor) -> torch.Tensor:
        return (
            torch.nn.functional.interpolate(
                img.unsqueeze(0).unsqueeze(0),
                size=self.output_size,
                mode="trilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[0] == self.output_size[0]:
            return self._resize_2d(img)
        return self._resize_3d(img)


class Slice:
    def __init__(self, channels: int = 1) -> None:
        self.c = channels

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        axis = torch.randint(0, 3, (1,))
        idx = torch.randint(0, img.shape[axis] - self.c, (1,)).item()

        if axis == 0:
            return img[idx : idx + self.c, :, :]
        elif axis == 1:
            return img[:, idx : idx + self.c, :].permute(1, 2, 0)
        else:
            return img[:, :, idx : idx + self.c].permute(2, 0, 1)


class Permute:
    def __init__(self, skip_first: bool = True) -> None:
        self.skip_first = skip_first

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.skip_first:
            dims_order = [0] + torch.randperm(2).add(1).tolist()
        else:
            dims_order = torch.randperm(3).tolist()
        return img.permute(dims_order)


class Flip:
    def __init__(self, skip_first=True) -> None:
        self.flip_dims = [1, 2] if skip_first else [0, 1, 2]

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        flip_dims = [dim for dim in self.flip_dims if torch.rand(1).item() < 0.5]
        return img.flip(dims=flip_dims) if flip_dims else img


class GaussianBlur:
    def __init__(self, p: float = 1.0, sigma: List[float] = [0.1, 0.5]) -> None:
        self.p = p
        self.sigma = sigma

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return gaussian_blur(img, kernel_size=3, sigma=self.sigma)  # type: ignore
        return img


class Norm:
    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.mean) / self.std


class ImageTransforms:
    def __init__(self) -> None:
        self.transforms = []

    def __iadd__(self, new_transform: Callable):
        self.transforms.append(new_transform)
        return self

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        for i, transform in enumerate(self.transforms):
            img = transform(img)
        return img


if __name__ == "__main__":
    print("3D -> 3D Crop")
    transform_3d = RandomCrop3D(size=112, scale=(0.3, 1.0))
    input_3d = torch.randn(56, 128, 256)
    output_3d = transform_3d(input_3d)
    print(f"Input shape: {input_3d.shape}")
    print(f"Output shape: {output_3d.shape}\n")

    print("3D -> 2D Slice Crop")
    transform_slice = RandomSliceCrop(size=128, channels=5)
    input_3d_long = torch.randn(96, 5, 160)
    output_slice_1 = transform_slice(input_3d_long)
    print(f"Input shape: {input_3d_long.shape}")
    print(f"Output shape: {output_slice_1.shape}\n")

    print("2D -> 2D Crop")
    transform_2d = RandomCrop2D(size=112, scale=(0.5, 0.9))
    input_2d = torch.randn(3, 100, 256)
    output_2d = transform_2d(input_2d)
    print(f"Input shape: {input_2d.shape}")
    print(f"Output shape: {output_2d.shape}")

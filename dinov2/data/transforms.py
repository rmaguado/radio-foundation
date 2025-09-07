import torch
import random
from typing import Tuple, List, Callable
import logging

logger = logging.getLogger("dinov2")


class RandomCrop2D:
    def __init__(self, size: int, scale: Tuple[float, float]) -> None:
        if not (0 < scale[0] <= scale[1]):
            raise ValueError(
                f"Scale range must be positive and ordered, but got {scale}"
            )
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"size must be a positive integer, but got {size}")
        self.crop_size = (size, size, size)
        self.scale = scale

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim != 3:
            raise ValueError(
                f"Input image must be 3D (C, H, W), but got shape {img.shape}"
            )

        scale = torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
        min_spatial_size = min(img.shape)

        crop_size = int(min_spatial_size * scale)

        max_start_h = img.shape[1] - crop_size
        start_h = random.randint(0, max_start_h)
        max_start_w = img.shape[2] - crop_size
        start_w = random.randint(0, max_start_w)

        cropped_img = img[
            :,
            start_h : start_h + crop_size,
            start_w : start_w + crop_size,
        ].float()

        resampled_img = torch.nn.functional.interpolate(
            cropped_img.unsqueeze(0),
            size=self.crop_size,
            mode="bilinear",
            align_corners=False,
        )

        return resampled_img.squeeze(0)


class Resize:
    def __init__(self, output_size: int) -> None:
        self.output_size = (output_size, output_size, output_size)

    def _resize_2d(self, img: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=self.output_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self._resize_2d(img)


class Permute2D:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < 0.5:
            return img
        return img.permute([1, 2])


class Flip2D:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        flip_dims = [dim for dim in [1, 2] if torch.rand(1).item() < 0.5]
        return img.flip(dims=flip_dims) if flip_dims else img


class Norm:
    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.mean) / self.std


class Standardize:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        imin = img.min()
        imax = img.max()
        irange = imax - imin

        if irange == 0:
            return img - imin

        mult = 2 / irange
        scal = -2 * imin / irange - 1

        return img * mult + scal


class Window:
    def __init__(
        self,
        p: float = 0.5,
        percentiles: tuple[float, float] = (1.0, 99.0),
        level_std_ratio: float = 0.2,
        width_range_ratio: tuple[float, float] = (0.5, 2.0),
        hist_bins: int = 512,
        hist_range: tuple[int, int] = (-1000, 1900),
    ):
        self.p = p
        self.percentiles = torch.tensor(
            [percentiles[0] / 100.0, 0.25, 0.50, 0.75, percentiles[1] / 100.0],
            dtype=torch.float32,
        )
        self.level_std_ratio = level_std_ratio
        self.width_range_ratio = width_range_ratio
        self.hist_bins = hist_bins
        self.hist_range = hist_range

    @torch.no_grad()
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return img

        hist = torch.histogram(
            img.float(),
            bins=self.hist_bins,
            range=self.hist_range,
        ).hist

        cdf = torch.cumsum(hist, dim=0)
        total_pixels = cdf[-1]

        q_indices = torch.searchsorted(cdf, self.percentiles * total_pixels)
        q_indices = torch.clamp(q_indices, 0, self.hist_bins - 1)

        bin_width = (self.hist_range[1] - self.hist_range[0]) / self.hist_bins
        hu_values = self.hist_range[0] + q_indices * bin_width

        p_low, q1, median, q3, p_high = hu_values

        iqr = q3 - q1
        if iqr < 1.0:
            iqr = torch.clamp(p_high - p_low, min=1.0)

        level_std = iqr * self.level_std_ratio
        window_level = torch.normal(mean=median, std=level_std).item()

        min_width = iqr * self.width_range_ratio[0]
        max_width = iqr * self.width_range_ratio[1]
        window_width = (
            torch.empty(1).uniform_(min_width.item(), max_width.item()).item()
        )
        window_width = max(window_width, 10.0)

        window_min = window_level - (window_width / 2)
        window_max = window_level + (window_width / 2)

        return torch.clip(img, window_min, window_max)


class Identity:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return img


class ImageTransforms:
    def __init__(self) -> None:
        self.transforms = []

    def __iadd__(self, new_transform: Callable):
        self.transforms.append(new_transform)
        return self

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            img = transform(img)
        return img

import torch
from typing import Tuple, List, Callable, Optional
from torchvision.transforms.functional import gaussian_blur


class Crop:
    def __init__(
        self,
        scale: Tuple[float, float],
        size: Tuple[int, int, int],
        preserve_axis_order: bool = False,
    ) -> None:
        self.crop_size = torch.tensor(size, dtype=torch.float32)
        self.crop_scale = scale
        self.preserve_axis_order = preserve_axis_order

    def _get_scale(self) -> float:
        return torch.empty(1).uniform_(self.crop_scale[0], self.crop_scale[1]).item()

    def _get_crop_indices(
        self, original_shape: torch.Tensor, crop_dims_voxels: torch.Tensor
    ) -> Tuple[list, list, list]:
        permutation = list(range(len(original_shape)))
        if self.preserve_axis_order:
            permutation = torch.randperm(len(original_shape)).tolist()

        perm_crop_dims_voxels = crop_dims_voxels[permutation]
        perm_original_shape = original_shape[permutation]

        max_start = (perm_original_shape.to(torch.int32) - perm_crop_dims_voxels).clamp(
            min=0
        )
        start = [
            torch.randint(0, int(max_start[i]) + 1, (1,)).item()
            for i in range(len(original_shape))
        ]
        end = [start[i] + perm_crop_dims_voxels[i] for i in range(len(original_shape))]

        inverse_permutation = [0] * len(original_shape)
        for i, p in enumerate(permutation):
            inverse_permutation[p] = i

        start_orig_order = [
            start[inverse_permutation[i]] for i in range(len(original_shape))
        ]
        end_orig_order = [
            end[inverse_permutation[i]] for i in range(len(original_shape))
        ]

        return start_orig_order, end_orig_order, inverse_permutation

    def _process_image(
        self,
        img: torch.Tensor | torch.Tensor,
        start_indices: list,
        end_indices: list,
        inverse_permutation: Optional[list] = None,
    ) -> torch.Tensor:
        if len(start_indices) == 3:
            cropped = img[
                start_indices[0] : end_indices[0],
                start_indices[1] : end_indices[1],
                start_indices[2] : end_indices[2],
            ]
        elif len(start_indices) == 2:
            cropped = img[
                :, start_indices[0] : end_indices[0], start_indices[1] : end_indices[1]
            ]
        else:
            raise ValueError("Unsupported number of dimensions for cropping.")

        if self.preserve_axis_order and inverse_permutation is not None:
            cropped = cropped.permute(tuple(inverse_permutation))

        return cropped.float()

    def crop_anisotropic(
        self,
        img: torch.Tensor,
        spacing: Tuple[float, float, float],
    ) -> torch.Tensor:
        original_shape = torch.tensor(img.shape, dtype=torch.float32)
        spacing_tensor = torch.tensor(spacing, dtype=torch.float32)
        min_spacing = spacing_tensor.min()

        crop_physical_shape = self.crop_size * spacing_tensor / min_spacing
        scale = self._get_scale()
        scaled_physical_crop = crop_physical_shape * scale
        crop_dims_voxels = torch.floor(scaled_physical_crop / spacing_tensor).to(
            torch.int32
        )

        crop_dims_voxels = torch.minimum(
            crop_dims_voxels, original_shape.to(torch.int32)
        )
        crop_dims_voxels = torch.maximum(
            crop_dims_voxels, torch.tensor([1, 1, 1], dtype=torch.int32)
        )

        start, end, inverse_permutation = self._get_crop_indices(
            original_shape, crop_dims_voxels
        )
        cropped_tensor = self._process_image(img, start, end, inverse_permutation)

        resampled = torch.nn.functional.interpolate(
            cropped_tensor.unsqueeze(0).unsqueeze(0),
            size=tuple(self.crop_size.to(torch.int32).tolist()),
            mode="trilinear",
            align_corners=False,
        )
        return resampled.squeeze(0).squeeze(0)

    def _crop_and_resample(
        self,
        img: torch.Tensor,
        target_size: Tuple[int, ...],
        mode: str,
        is_3d: bool,
    ) -> torch.Tensor:
        img_shape = torch.tensor(img.shape, dtype=torch.float32)
        scale = self._get_scale()

        if is_3d:
            crop_shape = torch.floor(img_shape * scale)
        else:
            crop_shape = torch.floor(img_shape[1:] * scale)

        crop_dims_voxels = crop_shape.to(torch.int32)
        crop_dims_voxels = torch.maximum(
            crop_dims_voxels,
            torch.tensor([1] * len(crop_dims_voxels), dtype=torch.int32),
        )
        effective_original_shape = img_shape if is_3d else img_shape[1:]

        start, end, inverse_permutation = self._get_crop_indices(
            effective_original_shape, crop_dims_voxels
        )

        cropped_tensor = self._process_image(
            img, start, end, inverse_permutation if is_3d else None
        )

        if is_3d:
            cropped_tensor = cropped_tensor.unsqueeze(0).unsqueeze(0)
        else:
            cropped_tensor = cropped_tensor.unsqueeze(0)

        resampled = torch.nn.functional.interpolate(
            cropped_tensor,
            size=target_size,
            mode=mode,
            align_corners=False,
        )

        return resampled.squeeze(0).squeeze(0) if is_3d else resampled.squeeze(0)

    def __call__(
        self,
        img: torch.Tensor | torch.Tensor,
        spacing: Optional[Tuple[float, float, float]] = None,
    ) -> torch.Tensor:
        if spacing is not None:
            return self.crop_anisotropic(img, spacing)

        if len(img.shape) == 3 and img.shape[0] == self.crop_size[0]:
            target_size = tuple(self.crop_size.to(torch.int32).tolist()[1:])
            return self._crop_and_resample(img, target_size, "bilinear", is_3d=False)
        elif len(img.shape) == 3:
            target_size = tuple(self.crop_size.to(torch.int32).tolist())
            return self._crop_and_resample(img, target_size, "trilinear", is_3d=True)
        else:
            raise ValueError(
                f"Unsupported image dimensions: {img.shape}. Expected 3D (D,H,W) or 2D (C,H,W where C matches crop_size[0])."
            )


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
        axis = torch.randint(0, 3, (1,)).item()
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
            return gaussian_blur(img, kernel_size=3, sigma=self.sigma)
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

    def __call__(
        self, img: torch.Tensor, spacing: Optional[Tuple[float, ...]] = None
    ) -> torch.Tensor:
        for i, transform in enumerate(self.transforms):
            if i == 0 and spacing is not None:
                img = transform(img, spacing)
            else:
                img = transform(img)
        return img


def get_transform(name, kwargs) -> Callable:
    transform_keys = {
        "crop": Crop,
        "permute": Permute,
        "flip": Flip,
        "gaussian_blur": GaussianBlur,
        "norm": Norm,
    }
    if name not in transform_keys:
        raise ValueError(f"Transform name '{name}' not recognized.")
    return transform_keys[name](**kwargs)

import random
import torch
import numpy as np
from typing import Tuple, Callable
from torchvision.transforms.functional import gaussian_blur

class Crop:
    def __init__(
        self,
        scale: Tuple[float, float],
        size: Tuple[int, int, int],
        preserve_axis_order: bool = False
    ) -> None:
        self.crop_size = torch.tensor(size, dtype=torch.float32)
        self.crop_scale = scale
        self.preserve_axis_order = preserve_axis_order

    def _get_scale(self) -> float:
        """Returns a random scale factor within the initialized range."""
        return random.uniform(self.crop_scale[0], self.crop_scale[1])

    def _get_crop_indices(
        self,
        original_shape: torch.Tensor,
        crop_dims_voxels: torch.Tensor
    ) -> Tuple[list, list, list]:
        permutation = list(range(len(original_shape)))
        if self.preserve_axis_order:
            random.shuffle(permutation)

        perm_crop_dims_voxels = crop_dims_voxels[permutation]
        perm_original_shape = original_shape[permutation]

        max_start = (perm_original_shape.to(torch.int32) - perm_crop_dims_voxels).clamp(min=0)
        start = [random.randint(0, int(max_start[i])) for i in range(len(original_shape))]
        end = [start[i] + perm_crop_dims_voxels[i] for i in range(len(original_shape))]

        inverse_permutation = [0] * len(original_shape)
        for i, p in enumerate(permutation):
            inverse_permutation[p] = i

        start_orig_order = [start[inverse_permutation[i]] for i in range(len(original_shape))]
        end_orig_order = [end[inverse_permutation[i]] for i in range(len(original_shape))]

        return start_orig_order, end_orig_order, inverse_permutation

    def _process_image(
        self,
        img: torch.Tensor | np.ndarray,
        start_indices: list,
        end_indices: list,
        inverse_permutation: list = None
    ) -> torch.Tensor:
        """Crops, potentially permutes, and converts the image to a tensor for interpolation."""
        if len(start_indices) == 3: # 3D cropping
            cropped = img[start_indices[0] : end_indices[0],
                          start_indices[1] : end_indices[1],
                          start_indices[2] : end_indices[2]]
        elif len(start_indices) == 2: # 2D cropping (assuming channel-first for 2D)
            cropped = img[:, start_indices[0] : end_indices[0],
                          start_indices[1] : end_indices[1]]
        else:
            raise ValueError("Unsupported number of dimensions for cropping.")

        if self.preserve_axis_order and inverse_permutation:
            if isinstance(cropped, torch.Tensor):
                cropped = cropped.permute(tuple(inverse_permutation))
            else:
                cropped = np.transpose(cropped, tuple(inverse_permutation))

        if isinstance(cropped, np.ndarray):
            cropped = torch.tensor(cropped, dtype=torch.float32)
        return cropped

    def crop_anisotropic(
        self,
        img: torch.Tensor | np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> torch.Tensor:
        """
        Crops an image anisotropically based on spacing and then resamples it.
        """
        original_shape = torch.tensor(img.shape, dtype=torch.float32)
        spacing_tensor = torch.tensor(spacing, dtype=torch.float32)
        min_spacing = spacing_tensor.min()

        crop_physical_shape = self.crop_size * spacing_tensor / min_spacing
        scale = self._get_scale()
        scaled_physical_crop = crop_physical_shape * scale
        crop_dims_voxels = torch.floor(scaled_physical_crop / spacing_tensor).to(torch.int32)

        crop_dims_voxels = torch.minimum(
            crop_dims_voxels, original_shape.to(torch.int32)
        )
        crop_dims_voxels = torch.maximum(
            crop_dims_voxels, torch.tensor([1, 1, 1], dtype=torch.int32)
        )

        start, end, inverse_permutation = self._get_crop_indices(original_shape, crop_dims_voxels)
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
        img: torch.Tensor | np.ndarray,
        target_size: Tuple[int, ...],
        mode: str,
        is_3d: bool,
    ) -> torch.Tensor:
        """
        Generic helper for 2D and 3D cropping and resampling.
        """
        img_shape = torch.tensor(img.shape, dtype=torch.float32)
        scale = self._get_scale()

        if is_3d:
            crop_shape = torch.floor(img_shape * scale)
        else: # For 2D, assuming channel-first, so shape refers to spatial dims
            crop_shape = torch.floor(img_shape[1:] * scale)


        crop_dims_voxels = crop_shape.to(torch.int32)
        
        # Ensure crop dimensions are at least 1
        crop_dims_voxels = torch.maximum(crop_dims_voxels, torch.tensor([1] * len(crop_dims_voxels), dtype=torch.int32))

        # Adjust original_shape for _get_crop_indices depending on 2D or 3D
        effective_original_shape = img_shape if is_3d else img_shape[1:]

        start, end, inverse_permutation = self._get_crop_indices(effective_original_shape, crop_dims_voxels)
        
        cropped_tensor = self._process_image(img, start, end, inverse_permutation if is_3d else None)

        # Unsqueeze for batch and channel dimensions for interpolate
        if is_3d:
            cropped_tensor = cropped_tensor.unsqueeze(0).unsqueeze(0)
        else:
            cropped_tensor = cropped_tensor.unsqueeze(0) # For 2D, only batch dim needed

        resampled = torch.nn.functional.interpolate(
            cropped_tensor,
            size=target_size,
            mode=mode,
            align_corners=False,
        )

        return resampled.squeeze(0).squeeze(0) if is_3d else resampled.squeeze(0)

    def __call__(self, img: torch.Tensor | np.ndarray, spacing: Tuple[float, float, float] = None) -> torch.Tensor:
        """
        Applies cropping based on input image dimensions and optional spacing.
        """
        if spacing is not None:
            return self.crop_anisotropic(img, spacing)
        
        if len(img.shape) == 3 and img.shape[0] == self.crop_size[0]:
            target_size = tuple(self.crop_size.to(torch.int32).tolist()[1:])
            return self._crop_and_resample(img, target_size, "bilinear", is_3d=False)
        elif len(img.shape) == 3:
            target_size = tuple(self.crop_size.to(torch.int32).tolist())
            return self._crop_and_resample(img, target_size, "trilinear", is_3d=True)
        else:
            raise ValueError(f"Unsupported image dimensions: {img.shape}. Expected 3D (D,H,W) or 2D (C,H,W where C matches crop_size[0]).")


class Resize:
    def __init__(self, output_size: Tuple[int, int, int]) -> None:
        self.output_size = output_size
    def _resize_2d(self, img: torch.Tensor) -> torch.Tensor:
        mode = "bilinear"
        img = img.unsqueeze(0)
        return torch.nn.functional.interpolate(
            img,
            size=self.output_size,
            mode=mode,
            align_corners=False,
        ).squeeze(0)
    def _resize_3d(self, img: torch.Tensor) -> torch.Tensor:
        mode = "trilinear"
        img = img.unsqueeze(0).unsqueeze(0)
        return torch.nn.functional.interpolate(
            img,
            size=self.output_size,
            mode=mode,
            align_corners=False,
        ).squeeze(0).squeeze(0)
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[0] == self.output_size[0]:
            return self._resize_2d(img)
        return self._resize_3d(img)
        

class Slice:
    def __init__(self, channels: int = 1) -> None:
        self.c = channels
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        axis = random.randint(0, 2)
        idx = random.randint(0, img.shape[axis] - 1 - self.c)
        if axis == 0:
            return img[idx:idx+self.c, :, :]
        elif axis == 1:
            return img[:, idx:idx+self.c, :]
        else:
            return img[:, :, idx:idx+self.c]
        

class Permute:
    def __init__(self, skip_first: bool = True) -> None:
        self.skip_first = skip_first

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.skip_first:
            dims_order = [1, 2]
            dims_order = random.shuffle(dims_order)
            dims_order = [0] + dims_order
        else:
            dims_order = [0, 1, 2]
            dims_order = random.shuffle(dims_order)

        return img.permute(dims_order)


class Flip:
    def __init__(self, skip_first=True) -> None:
        self.flip_dims = [1, 2] if skip_first else [0, 1, 2]

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        flip_dims = [dim for dim in self.flip_dims if random.random() < 0.5]
        return img.flip(dims=flip_dims)


class GaussianBlur:
    def __init__(
        self, p: float = 1.0, sigma: Tuple[float, float] | float = (0.1, 0.5)
    ) -> None:
        self.p = p
        self.sigma = sigma

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
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

    def __iadd__(self, new_transform: Callable) -> None:
        self.transforms.append(new_transform)
        return self

    def __call__(
        self, img: torch.Tensor, spacing: Tuple[float, ...] = None
    ) -> torch.Tensor:
        for i, transform in enumerate(self.transforms):
            if i == 0:
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

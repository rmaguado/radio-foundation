import random
import torch
from typing import Tuple, Callable
from torchvision.transforms.functional import gaussian_blur

class Crop:
    def __init__(
        self,
        scale: Tuple[float, float],
        size: Tuple[int, int, int],
    ) -> None:
        self.crop_size = torch.tensor(size, dtype=torch.float32)
        self.crop_scale = scale

    def get_scale(self) -> float:
        return random.uniform(self.crop_scale[0], self.crop_scale[1])

    def crop_anisotropic(
        self,
        img: torch.Tensor,
        spacing: Tuple[float, float, float],
    ) -> torch.Tensor:

        original_shape = torch.tensor(img.shape, dtype=torch.float32)
        spacing = torch.tensor(spacing, dtype=torch.float32)
        min_spacing = spacing.min()

        crop_physical_shape = self.crop_size * spacing / min_spacing

        scale = self.get_scale()
        scaled_physical_crop = crop_physical_shape * scale

        crop_dims_voxels = torch.floor(scaled_physical_crop / spacing).to(torch.int32)

        crop_dims_voxels = torch.minimum(crop_dims_voxels, original_shape.to(torch.int32))
        crop_dims_voxels = torch.maximum(
            crop_dims_voxels, torch.tensor([1, 1, 1], dtype=torch.int32)
        )

        max_start = (original_shape.to(torch.int32) - crop_dims_voxels).clamp(min=0)

        start = [random.randint(0, int(max_start[i])) for i in range(3)]
        end = [start[i] + crop_dims_voxels[i] for i in range(3)]

        cropped = img[start[0] : end[0], start[1] : end[1], start[2] : end[2]]
        cropped = cropped.unsqueeze(0).unsqueeze(0)

        resampled = torch.nn.functional.interpolate(
            cropped,
            size=self.crop_size,
            mode="trilinear",
            align_corners=False,
        )

        return resampled.squeeze(0).squeeze(0)
    
    def crop2d(
        self,
        img: torch.Tensor,
    ) -> torch.Tensor:
        
        xy_shape = torch.tensor(img.shape[1:], dtype=torch.float32)
        scale = self.get_scale()

        crop_shape = torch.floor(xy_shape * scale)
        max_start = xy_shape - crop_shape

        start = [random.randint(0, int(max_start[i])) for i in range(2)]
        end = [start[i] + crop_shape[i] for i in range(2)]

        img_crop = img[:,start[0]:end[0],start[1]:end[1]].unsqueeze(0)

        resampled = torch.nn.functional.interpolate(
            img_crop,
            size=self.crop_size,
            mode="bilinear",
            align_corners=False,
        )
        return resampled.squeeze(0)
    

    def crop3d(
        self,
        img: torch.Tensor,
    ) -> torch.Tensor:
        img_shape = torch.tensor(img.shape, dtype=torch.float32)
        scale = self.get_scale()

        crop_shape = torch.floor(img_shape * scale)
        max_start = img_shape - crop_shape

        start = [random.randint(0, int(max_start[i])) for i in range(3)]
        end = [start[i] + crop_shape[i] for i in range(3)]

        img_crop = img[start[0]:end[0],start[1]:end[1],start[2]:end[2]].unsqueeze(0).unsqueeze(0)

        resampled = torch.nn.functional.interpolate(
            img_crop,
            size=self.crop_size,
            mode="trilinear",
            align_corners=False,
        )
        return resampled.squeeze(0).squeeze(0)
    
    def __call__(self, img, spacing=None) -> torch.Tensor:        
        if spacing is not None:
            return self.crop_anisotropic(img, spacing)
        if img.shape[0] == self.crop_size[0]:
            return self.crop2d(img)
        return self.crop3d(img)


class Flip:
    def __init__(self, skip_first=True) -> None:
        self.flip_dims = [1,2] if skip_first else [0,1,2]

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        flip_dims = [dim for dim in self.flip_dims if random.random() < 0.5]
        return img.flip(dims=flip_dims)


class GaussianBlur:
    def __init__(self, p: float = 1.0, sigma: Tuple[float, float] | float = (0.1, 0.5)) -> None:
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
    def __call__(self, img: torch.Tensor, spacing: Tuple[float, ...] = None) -> torch.Tensor:
        for i, transform in enumerate(self.transforms):
            if i == 0:
                img = transform(img, spacing)
            else:
                img = transform(img)
        return img
    
def get_transform(name, kwargs) -> Callable:
    transform_keys = {
        "crop": Crop,
        "flip": Flip,
        "gaussian_blur": GaussianBlur,
        "norm": Norm
    }
    if name not in transform_keys:
        raise ValueError(f"Transform name '{name}' not recognized.")
    return transform_keys[name](**kwargs)
    
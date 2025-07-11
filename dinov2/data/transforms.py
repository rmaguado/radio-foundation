import random
import torch
from torchvision import transforms
from typing import Union, Tuple, Callable
import copy


class RandomApply:
    def __init__(self, transform_function: Callable, p: float, kwargs: dict):
        self.transform_function = transform_function
        self.p = p
        self.kwargs = kwargs

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return self.transform_function(img, **self.kwargs)
        return img


class ImageTransforms:

    def __init__(self, lower_bound: float, upper_bound: float) -> None:
        """
        Initializes a Transform object.
        Contains a list of transformations to be applied to an image.

        Args:
            lower_bound (float): The minimum value of the image pixel.
            upper_bound (float): The maximum value of the image pixel.
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.transform_list = []

        self._transforms = {
            "crop": self._crop,
            "sharpness": self._sharpness,
            "slice": self._slice,
            "rotate": self._rotate,
            "flip2d": self._flip2d,
            "flip3d": self._flip3d,
            "transpose3d": self._transpose3d,
            "gaussian_blur": self._gaussian_blur,
        }

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Applies the transformations to the image.

        Args:
            img (torch.Tensor): The image to be transformed.

        Returns:
            torch.Tensor: The transformed image.
        """
        for transform in self.transform_list:
            img = transform(img)

        return img

    def keys(self):
        """
        Returns the keys of the transformations dictionary.
        """
        return self._transforms.keys()

    def add_transform(self, transform_name: str, kwargs):
        """
        Adds a transformation to the transform list.

        Args:
            transform_name (str): The name of the transformation to be added.
            **kwargs: The parameters of the transformation.
        """
        self.transform_list.append(self._get_random_transform(transform_name, kwargs))

    def add_normalize(self, mean: float, std: float):
        """
        Adds a normalization transformation to the transform list.

        Args:
            mean (float): The mean value of the image pixel.
            std (float): The standard deviation of the image pixel.
        """
        self.transform_list.append(transforms.Normalize(mean=mean, std=std))

    def _get_random_transform(self, transform_name: str, kwargs: dict) -> Callable:
        kwargs = copy.deepcopy(kwargs)
        p = kwargs.pop("p", 1.0)
        transform_function = self._transforms[transform_name]

        return RandomApply(transform_function, p, kwargs)

    
    def _crop(self, img: torch.Tensor, croptype: str, crop_size: int, scale: Tuple = (0.08, 1.0)) -> torch.Tensor:
        if croptype == "3d":
            return self._resized_crop_3d(img, crop_size, scale)
        else:
            return self._resized_crop_2d(img, crop_size, scale)

    def _resized_crop_3d(
        self, img: torch.Tensor, crop_size: int, scale: Tuple
    ) -> torch.Tensor:
        crop_shape = (crop_size, crop_size, crop_size)

        d, h, w = img.shape  # Expecting [D, H, W]

        scale_factor = random.uniform(*scale)
        target_d = min(d, int(d * scale_factor))
        target_h = min(h, int(h * scale_factor))
        target_w = min(w, int(w * scale_factor))

        start_d = random.randint(0, d - target_d) if d > target_d else 0
        start_h = random.randint(0, h - target_h) if h > target_h else 0
        start_w = random.randint(0, w - target_w) if w > target_w else 0

        cropped = img[
            start_d : start_d + target_d,
            start_h : start_h + target_h,
            start_w : start_w + target_w,
        ]
        resized = (
            torch.nn.functional.interpolate(
                cropped.unsqueeze(0).unsqueeze(0),
                size=crop_shape,
                mode="trilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

        return resized

    def _resized_crop_2d(
        self, img: torch.Tensor, crop_size: int, scale: Tuple
    ) -> torch.Tensor:
        crop_shape = (crop_size, crop_size)

        _, h, w = img.shape  # [C, H, W]

        scale_factor = random.uniform(*scale)
        target_h = min(h, int(h * scale_factor))
        target_w = min(w, int(w * scale_factor))

        start_h = random.randint(0, h - target_h) if h > target_h else 0
        start_w = random.randint(0, w - target_w) if w > target_w else 0

        cropped = img[:, start_h : start_h + target_h, start_w : start_w + target_w]
        resized = torch.nn.functional.interpolate(
            cropped.unsqueeze(0), size=crop_shape, mode="bilinear", align_corners=False
        ).squeeze(0)

        return resized

    def _slice(self, img: torch.Tensor, n_slices: int = 1) -> torch.Tensor:
        shape = img.shape  # Expecting [C, D, H, W]
        axis = random.randint(0, 2)
        idx = random.randint(0, shape[axis] - n_slices - 1)

        if axis == 0:
            img = img[idx : idx + n_slices, :, :]
        elif axis == 1:
            img = img[:, idx : idx + n_slices, :]
        else:
            img = img[:, :, idx : idx + n_slices]

        img = img.permute(*([axis] + [i for i in [0, 1, 2] if i != axis]))
        return img

    def _sharpness(self, img: torch.Tensor, bounds: Tuple = (0.9, 1.5)) -> torch.Tensor:
        img = img.unsqueeze(0)

        factor = random.uniform(bounds[0], bounds[1])
        kernel = (
            (
                torch.tensor(
                    [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=torch.float32
                )
                * factor
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        img = torch.nn.functional.conv2d(img, kernel, padding=1)

        return img.squeeze(0)

    def _rotate(
        self,
        img: torch.Tensor,
    ) -> torch.Tensor:
        angle = 180.0 * random.uniform(-1, 1)
        return transforms.functional.rotate(img, angle, fill=self.lower_bound)

    def _transpose3d(self, img: torch.Tensor) -> torch.Tensor:
        dims = [0, 1, 2]
        random.shuffle(dims)
        return img.permute(*dims)

    def _flip2d(self, img: torch.Tensor) -> torch.Tensor:
        dim = random.randint(1, 2)
        return img.flip(dim)

    def _flip3d(self, img: torch.Tensor) -> torch.Tensor:
        dim = random.randint(0, 2)
        return img.flip(dim)

    def _gaussian_blur(
        self, img: torch.Tensor, bounds: Tuple = (0.1, 0.9)
    ) -> torch.Tensor:
        sigma = random.uniform(bounds[0], bounds[1])
        return transforms.functional.gaussian_blur(img, kernel_size=3, sigma=sigma)

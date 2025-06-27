import random
import torch
from torchvision import transforms
from typing import Union, Tuple, Callable
import copy

Param = Union[str, float]


class ImageTransforms:

    def __init__(self, lower_bound: float, upper_bound: float, channels: int) -> None:
        """
        Initializes a Transform object.
        Contains a list of transformations to be applied to an image.

        Args:
            lower_bound (float): The minimum value of the image pixel.
            upper_bound (float): The maximum value of the image pixel.
            channels (int): The number of channels in the image.
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.channels = channels

        self.transform_list = []

        self._transforms = {
            "sharpness": self._sharpness,
            "slice": self._slice,
            "rotate": self._rotate,
            "flip": self._flip,
            "gaussian_blur": self._gaussian_blur,
            "noise": self._noise,
            "window": self._window,
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

    def add_crop(self, crop_size: int, crop_scale: Tuple):
        self.transform_list.append(
            transforms.RandomResizedCrop(  # type: ignore
                crop_size,
                scale=crop_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,  # type: ignore
                antialias=True,
            )
        )

    def add_normalize(self, mean: float, std: float):
        """
        Adds a normalization transformation to the transform list.

        Args:
            mean (float): The mean value of the image pixel.
            std (float): The standard deviation of the image pixel.
        """
        self.transform_list.append(transforms.Normalize(mean=mean, std=std))  # type: ignore

    def _get_random_transform(self, transform_name: str, kwargs: dict) -> Callable:
        kwargs = copy.deepcopy(kwargs)
        p = kwargs.pop("p")
        transform_function = self._transforms[transform_name]

        def random_apply(img):
            if random.random() < p:
                return transform_function(img, **kwargs)
            return img

        return random_apply

    def _slice(self, img: torch.Tensor) -> torch.Tensor:
        shape = img.shape
        axis = random.randint(0, 3)
        slice_index = random.randint(0, shape[axis] - 1)

        if axis == 0:
            return img[slice_index, :, :]
        elif axis == 1:
            return img[:, slice_index, :]
        else:
            return img[:, :, slice_index]

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
        return transforms.functional.rotate(img, angle, fill=self.lower_bound)  # type: ignore

    def _flip(self, img: torch.Tensor) -> torch.Tensor:
        return transforms.functional.hflip(img)  # type: ignore

    def _gaussian_blur(
        self, img: torch.Tensor, bounds: Tuple = (0.1, 0.9)
    ) -> torch.Tensor:
        sigma = random.uniform(bounds[0], bounds[1])
        return transforms.functional.gaussian_blur(img, kernel_size=3, sigma=sigma)  # type: ignore

    def _noise(
        self, img: torch.Tensor, mean: float = 0.0, std: float = 1.0
    ) -> torch.Tensor:
        noise = torch.normal(mean, std, size=img.size())
        return img + noise

    def _window(
        self, img: torch.Tensor, width_bounds: Tuple, height_bounds: Tuple
    ) -> torch.Tensor:

        width = int(random.uniform(width_bounds[0], width_bounds[1]))
        height = int(random.uniform(height_bounds[0], height_bounds[1]))

        img = torch.clip(img, height, height + width)

        img = (img - height) / width * (
            self.upper_bound - self.lower_bound
        ) + self.lower_bound

        return img

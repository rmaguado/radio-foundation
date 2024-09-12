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
            "brightness": self._brightness,
            "contrast": self._contrast,
            "saturation": self._saturation,
            "hue": self._hue,
            "sharpness": self._sharpness,
            "rotate": self._rotate,
            "flip": self._flip,
            "color_jitter": self._color_jitter,
            "gaussian_blur": self._gaussian_blur,
            "solarize": self._solarize,
            "gray_scale": self._gray_scale,
            "noise": self._noise,
            "gamma_correction": self._gamma_correction,
            "window": self._window,
        }

    def __call__(self, img: torch.tensor) -> torch.tensor:
        """
        Applies the transformations to the image.

        Args:
            img (torch.tensor): The image to be transformed.

        Returns:
            torch.tensor: The transformed image.
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
        """
        Adds a random resized crop transformation to the transform list.

        Args:
            crop_size (int): The size of the crop.
            crop_scale (Tuple): The range of scale of the crop.
        """
        return transforms.RandomResizedCrop(
            crop_size,
            scale=crop_scale,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )

    def add_normalize(self, mean: float, std: float):
        """
        Adds a normalization transformation to the transform list.

        Args:
            mean (float): The mean value of the image pixel.
            std (float): The standard deviation of the image pixel.
        """
        return transforms.Normalize(mean=mean, std=std)

    def _get_random_transform(self, transform_name: str, kwargs: dict) -> Callable:
        kwargs = copy.deepcopy(kwargs)
        kwargs.pop("name")
        p = kwargs.pop("p")
        transform_function = self._transforms[transform_name]

        def random_apply(img):
            if random.random() < p:
                return transform_function(self, img, **kwargs)
            return img

        return random_apply

    def _brightness(
        self, img: torch.tensor, bounds: Tuple = (0.1, 0.4)
    ) -> torch.tensor:
        factor = random.uniform(bounds[0], bounds[1])
        img = img + factor
        return torch.clip(img, self.lower_bound, self.upper_bound)

    def _contrast(self, img: torch.tensor, bounds: Tuple = (0.1, 0.4)) -> torch.tensor:
        mean = torch.mean(img)
        factor = 1.0 + random.uniform(bounds[0], bounds[1])
        img = (img - mean) * factor + mean
        return torch.clip(img, self.lower_bound, self.upper_bound)

    def _saturation(
        self, img: torch.tensor, bounds: Tuple = (0.1, 0.4)
    ) -> torch.tensor:
        factor = random.uniform(bounds[0], bounds[1])
        return transforms.functional.adjust_saturation(img, factor)

    def _hue(self, img: torch.tensor) -> torch.tensor:
        factor = random.uniform(-0.5, 0.5)
        return transforms.functional.adjust_hue(img, factor)

    def _sharpness(self, img: torch.tensor, bounds: Tuple = (0.9, 1.5)) -> torch.tensor:
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
        img: torch.tensor,
    ) -> torch.tensor:
        angle = 180.0 * random.uniform(-1, 1)
        return transforms.functional.rotate(img, angle)

    def _flip(self, img: torch.tensor) -> torch.tensor:
        return transforms.functional.hflip(img)

    def _color_jitter(
        self,
        img: torch.tensor,
        brightness_bounds: Tuple = (0.1, 0.4),
        contrast_bounds: Tuple = (0.1, 0.4),
        saturation_bounds: Tuple = (0.1, 0.4),
    ) -> torch.tensor:

        img = self._brightness(img, brightness_bounds)
        img = self._contrast(img, contrast_bounds)
        img = self._saturation(img, saturation_bounds)
        return self._hue(img)

    def _gaussian_blur(
        self, img: torch.tensor, bounds: Tuple = (0.1, 2.0)
    ) -> torch.tensor:
        sigma = random.uniform(bounds[0], bounds[1])
        return transforms.functional.gaussian_blur(img, kernel_size=3, sigma=sigma)

    def _solarize(self, img: torch.tensor, bounds: Tuple) -> torch.tensor:
        threshold = random.uniform(bounds[0], bounds[1])
        mask = img > threshold
        img[mask] = self.upper_bound - img[mask]

        return img

    def _gray_scale(self, img: torch.tensor) -> torch.tensor:
        return transforms.functional.rgb_to_grayscale(img)

    def _noise(
        self, img: torch.tensor, mean: float = 0.0, std: float = 1.0
    ) -> torch.tensor:
        noise = torch.normal(mean, std, size=img.size())
        return img + noise

    def _gamma_correction(
        self, img: torch.tensor, bounds: Tuple = (0.1, 2.0)
    ) -> torch.tensor:
        gamma = random.uniform(bounds[0], bounds[1])
        return img**gamma

    def _window(
        self, img: torch.tensor, width_bounds: Tuple, height_bounds: Tuple
    ) -> torch.tensor:

        width = int(random.uniform(width_bounds[0], width_bounds[1]))
        height = int(random.uniform(height_bounds[0], height_bounds[1]))

        img = torch.clip(img, height, height + width)

        img = (img - height) / width * (
            self.upper_bound - self.lower_bound
        ) + self.lower_bound

        return img

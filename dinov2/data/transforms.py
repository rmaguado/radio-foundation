# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import random
import torch
from torchvision import transforms


class RandomRotation:
    def __init__(self, p: str | float, degrees: str | float):
        """
        Initializes a RandomRotation transform object.
        Applies random rotation to an image about the center. Applied with a given probability.

        Args:
            p (float): The probability of applying the rotation.
            degrees (float): The range of degrees to randomly rotate the image.
        """
        self.p = float(p)
        self.degrees = float(degrees)
        self.rotate = transforms.RandomRotation(
            self.degrees, interpolation=transforms.InterpolationMode.BILINEAR
        )

    def __call__(self, img: torch.tensor) -> torch.tensor:
        if random.random() < self.p:
            return self.rotate(img)
        return img


class RandomColorJitter:
    def __init__(
        self,
        p: str | float,
        brightness: str | float,
        contrast: str | float,
        saturation: str | float,
        hue: str | float,
    ):
        """
        Initializes a RandomColorJitter transform object.
        Applies random color jittering transformations to an image. Applied with a given probability.

        Args:
            p (float): Probability of applying the color jittering transformations.
            brightness (float): How much to jitter the brightness. Should be a float value.
            contrast (float): How much to jitter the contrast. Should be a float value.
            saturation (float): How much to jitter the saturation. Should be a float value.
            hue (float): How much to jitter the hue. Should be a float value.
        """
        self.p = float(p)
        self.jitter = transforms.ColorJitter(
            brightness=float(brightness),
            contrast=float(contrast),
            saturation=float(saturation),
            hue=float(hue),
        )

    def __call__(self, img: torch.tensor) -> torch.tensor:
        if random.random() < self.p:
            return self.jitter(img)
        return img


class RandomContrast:
    def __init__(self, p: str | float, contrast: str | float):
        """
        Initializes a RandomContrast transform object.
        Applies random contrast adjustment to an image. Applied with a given probability.

        Args:
            p (float): Probability of applying the contrast adjustment.
            contrast (float): Range of contrast adjustment to be applied.
        """
        self.p = float(p)
        self.contrast = float(contrast)

    def __call__(self, img: torch.tensor) -> torch.tensor:
        if random.random() < self.p:
            mean = torch.mean(img)
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            img = (img - mean) * factor + mean
            return torch.clip(img, 0.0, 1.0)
        return img


class RandomBrightness:
    def __init__(self, p: str | float, brightness: str | float):
        """
        Initializes a RandomBrightness transform object.
        Applies random brightness adjustment to an image. Applied with a given probability.

        Args:
            p (float): The probability of applying the brightness transformation.
            brightness (float): The maximum brightness adjustment factor.
        """
        self.p = float(p)
        self.brightness = float(brightness)

    def __call__(self, img: torch.tensor) -> torch.tensor:
        if random.random() < self.p:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = img * factor
            return torch.clip(img, 0.0, 1.0)
        return img


class RandomSharpness:
    def __init__(self, p: str | float, sharpness: str | float):
        """
        Initializes a RandomSharpness transform object.
        Convolves a sharpening kernel to an image. Applied with a given probability.

        Args:
            p (float): The probability of applying the sharpness transformation.
            sharpness (float): The maximum sharpness adjustment factor.
        """
        self.p = float(p)
        self.sharpness = float(sharpness)

    def __call__(self, img: torch.tensor) -> torch.tensor:
        if random.random() < self.p:
            factor = random.uniform(1 - self.sharpness, 1 + self.sharpness)

            kernel = (
                torch.tensor(
                    [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=torch.float32
                )
                * factor
            )

            img = self.convolve2d(img, kernel)
            return torch.clip(img, 0.0, 1.0)
        return img

    def convolve2d(self, image: torch.tensor, kernel: torch.tensor) -> torch.tensor:
        image = image.unsqueeze(0)
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        output_image = torch.nn.functional.conv2d(image, kernel, padding=1)

        return output_image.squeeze(0)


class RandomGaussianBlur:
    def __init__(self, p: str | float):
        """
        Initializes a RandomGaussianBlur transform object.
        Applies Gaussian blur to an image. Applied with a given probability.

        Args:
            p (float): The probability of applying the Gaussian blur transformation.
        """
        self.p = float(p)
        self.transform = transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))

    def __call__(self, img: torch.tensor) -> torch.tensor:
        if random.random() < self.p:
            return self.transform(img)
        return img


class RandomSolarize:
    def __init__(self, p: str | float, threshold: str | float, max_value: str | float):
        """
        Initializes a RandomSolarize transform object.
        Inverts the image if the pixel value is above a threshold. Applied with a given probability.

        Args:
            p (float): The probability of applying the solarize transformation.
            threshold (float): The threshold value for solarizing the image.
            max_value (float): The maximum value for the image (1 for float image. 255 for 8 bit).
        """
        self.p = float(p)
        self.threshold = float(threshold)
        self.max_value = float(max_value)

    def __call__(self, img: torch.tensor) -> torch.tensor:
        if random.random() < self.p:
            mask = img > self.threshold
            img[mask] = self.max_value - img[mask]

            return img
        return img


class RandomGrayscale:
    def __init__(self, p: str | float):
        """
        Initializes a RandomGrayscale transform object.
        Converts the image to grayscale. Applied with a given probability.

        Args:
            p (float): The probability of applying the grayscale transformation.
        """
        self.p = float(p)
        self.gray = transforms.RandomGrayscale(self.p)

    def __call__(self, img: torch.tensor) -> torch.tensor:
        return self.gray(img)


class RandomNoise:
    def __init__(
        self, p: str | float, noise_level: str | float, max_value: str | float
    ):
        """
        Initializes a RandomNoise transform object.
        Applies normaly distributed noise to an image. Applied with a given probability.

        Args:
            p (float): The probability of applying the noise transformation.
            noise_level (float): The standard deviation of the noise to be added.
            max_value (float): The maximum value for the image (1 for float image. 255 for 8 bit).
        """
        self.p = float(p)
        self.noise_level = float(noise_level)
        self.max_value = float(max_value)

    def __call__(self, img: torch.tensor) -> torch.tensor:
        if random.random() < self.p:
            noise = torch.normal(0, self.noise_level, img.shape)
            return torch.clip(img + noise, 0, self.max_value)
        return img


class RandomFlip:
    def __init__(self, p: str | float):
        """
        Initializes a RandomFlip transform object.
        Flips the image horizontally. Applied with a given probability.

        Args:
            p (float): The probability of applying the flip transformation.
        """
        self.p = float(p)
        self.flip = transforms.RandomHorizontalFlip(self.p)

    def __call__(self, img: torch.tensor) -> torch.tensor:
        return self.flip(img)


class RandomGamma:
    def __init__(self, p: str | float, gamma: str | float):
        """
        Initializes a RandomGamma transform object.
        Applies gamma transformation to an image. Applied with a given probability.
        Raises the pixel values to the power of gamma.

        Args:
            p (float): The probability of applying the gamma transformation.
            gamma (float): The gamma value to be applied.
        """
        self.p = float(p)
        self.gamma = float(gamma)

    def __call__(self, img: torch.tensor) -> torch.tensor:
        if random.random() < self.p:
            return torch.pow(img, self.gamma)
        return img


class RandomWindow:
    def __init__(self, p: str | float, height: str | float, width: str | float):
        """
        Initializes a RandomWindow transform object.
        Increases the contrast of the image based on height (minimum intensity value) and width of the window.
        Applied with a given probability.

        Args:
            p (float): The probability of applying the window transformation.
            height (float): The height of the window.
            width (float): The width of the window.
        """
        self.p = float(p)
        self.height = float(height)
        self.width = float(width)
        assert abs(self.height) < 0
        assert abs(self.width) < 0
        assert self.height + self.width <= 1

    def __call__(self, img: torch.tensor) -> torch.tensor:
        if random.random() < self.p:
            width = random.uniform(self.width, 1)
            height = random.uniform(0, 1 - self.width)
            return torch.clip(img, height, height + width)
        return img


transformkeys = {
    "rotation": RandomRotation,
    "colorjitter": RandomColorJitter,
    "contrast": RandomContrast,
    "brightness": RandomBrightness,
    "sharpness": RandomBrightness,
    "blur": RandomGaussianBlur,
    "solarize": RandomSolarize,
    "grayscale": RandomGrayscale,
    "noise": RandomNoise,
    "flip": RandomFlip,
    "gamma": RandomGamma,
    "window": RandomWindow,
}

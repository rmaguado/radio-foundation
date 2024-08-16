# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Sequence

import random
import numpy as np

import torch
from torchvision import transforms


class RandomRotation:
    def __init__(self, p, degrees):
        self.p = float(p)
        self.degrees = float(degrees)
        self.rotate = transforms.RandomRotation(
            self.degrees, interpolation=transforms.InterpolationMode.BILINEAR
        )

    def __call__(self, img):
        if random.random() < self.p:
            return self.rotate(img)
        return img
    

class RandomColorJitter:
    def __init__(self, p, brightness, contrast, saturation, hue):
        self.p = float(p)
        self.jitter = transforms.ColorJitter(
            brightness=float(brightness),
            contrast=float(contrast),
            saturation=float(saturation),
            hue=float(hue)
        )

    def __call__(self, img):
        if random.random() < self.p:
            return self.jitter(img)
        return img


class RandomContrast:
    def __init__(self, p, contrast):
        self.p = float(p)
        self.contrast = float(contrast)

    def __call__(self, img):
        if random.random() < self.p:
            mean = torch.mean(img)
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            img = (img - mean) * factor + mean
            return torch.clip(img, 0.0, 1.0)
        return img


class RandomBrightness:
    def __init__(self, p, brightness):
        self.p = float(p)
        self.brightness = float(brightness)

    def __call__(self, img):            
        if random.random() < self.p:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = img * factor
            return torch.clip(img, 0.0, 1.0)
        return img


class RandomSharpness:
    def __init__(self, p, sharpness):
        self.p = float(p)
        self.sharpness = float(sharpness)

    def __call__(self, img):
        if random.random() < self.p:
            factor = random.uniform(1 - self.sharpness, 1 + self.sharpness)

            kernel = torch.tensor([
                [-1, -1, -1],
                [-1, 9, -1],
                [-1, -1, -1]
            ], dtype=torch.float32) * factor

            img = self.convolve2d(img, kernel)
            return torch.clip(img, 0.0, 1.0)
        return img

    def convolve2d(self, image, kernel):
        image = image.unsqueeze(0)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        
        output_image = F.conv2d(image, kernel, padding=1)
        
        return output_image.squeeze(0)


class RandomGaussianBlur:
    def __init__(self, p):
        self.p = float(p)
        self.transform = transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))
    
    def __call__(self, img):
        if random.random() < self.p:
            return self.transform(img)
        return img


class RandomSolarize:
    def __init__(self, p, threshold, max_value):
        self.p = float(p)
        self.threshold = float(threshold)
        self.max_value = float(max_value)
    
    def __call__(self, img):
        if random.random() < self.p:
            mask = img > self.threshold
            img[mask] = self.max_value - img[mask]
            
            return img
        return img
    
    
class RandomGrayscale:
    def __init__(self, p):
        self.p = float(p)
        self.gray = transforms.RandomGrayscale(self.p)
    def __call__(self, img):
        return self.gray(img)


class RandomNoise:
    def __init__(self, p, noise_level, max_value):
        self.p = float(p)
        self.noise_level = float(noise_level)
        self.max_value = float(max_value)
    
    def __call__(self, img):
        if random.random() < self.p:
            noise = torch.normal(0, self.noise_level, img.shape)
            return torch.clip(img + noise, 0, self.max_value)
        return img


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


def make_normalize_transform(
    mean: Sequence[float],
    std: Sequence[float],
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = 0.5,
    std: Sequence[float] = 0.5,
):
    transforms_list = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
    transforms_list.extend(
        [
            MaybeToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
    return transforms.Compose(transforms_list)


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = 0.5,
    std: Sequence[float] = 0.5,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)


TRANSFORMS = {
    "rotation" : RandomRotation,
    "colorjitter": RandomColorJitter,
    "contrast": RandomContrast,
    "brightness": RandomBrightness,
    "sharpness": RandomBrightness,
    "blur": RandomGaussianBlur,
    "solarize": RandomSolarize,
    "grayscale": RandomGrayscale,
    "noise": RandomNoise,
}
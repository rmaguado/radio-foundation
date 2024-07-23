# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Sequence

import random

import torch
from torchvision import transforms


class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)
        

class RandomSolarize:
    def __init__(self, threshold=128, p=0.2):
        self.threshold = threshold
        self.p = p

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError('Input should be a PIL image.')

        if img.mode != 'F':
            raise TypeError('Image mode should be F.')

        if random.random() < self.p:
            img_array = np.array(img)
            img_array[img_array > self.threshold] = 1.0 - img_array[img_array > self.threshold]
            img = Image.fromarray(img_array, mode='F')
        
        return img


class RandomGrayJitter:
    def __init__(self, brightness=0.4, contrast=0.4, p=0.8):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError('Input should be a PIL image.')

        if img.mode != 'F':
            raise TypeError('Image mode should be F.')

        if random.random() < self.p:
            img_array = np.array(img)
            img_array = self.adjust_brightness(img_array)
            img_array = self.adjust_contrast(img_array)
            img = Image.fromarray(img_array, mode='F')

        return img

    def adjust_brightness(self, img_array):
        factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        img_array = img_array * factor
        img_array = np.clip(img_array, 0, 1)
        return img_array

    def adjust_contrast(self, img_array):
        mean = np.mean(img_array)
        factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        img_array = (img_array - mean) * factor + mean
        img_array = np.clip(img_array, 0, 1)
        return img_array


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


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
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
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)

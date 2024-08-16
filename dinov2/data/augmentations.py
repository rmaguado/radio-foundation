# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from torchvision import transforms

from .transforms import (
    TRANSFORMS,
    make_normalize_transform
)


logger = logging.getLogger("dinov2")


class DataAugmentationDINO(object):
    def __init__(self, cfg):
        self.local_crops_number = cfg.crops.local_crops_number

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {cfg.crops.global_crops_scale}")
        logger.info(f"local_crops_scale: {cfg.crops.local_crops_scale}")
        logger.info(f"local_crops_number: {cfg.crops.local_crops_number}")
        logger.info(f"global_crops_size: {cfg.crops.global_crops_size}")
        logger.info(f"local_crops_size: {cfg.crops.local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        crop_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    cfg.crops.global_crops_size,
                    scale=cfg.crops.global_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        crop_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    cfg.crops.local_crops_size,
                    scale=cfg.crops.local_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
        
        normalize = make_normalize_transform(cfg.norm.mean, cfg.norm.std)

        self.global_transfo1 = parse_transforms(cfg.global_1, crop_global, normalize)
        self.global_transfo2 = parse_transforms(cfg.global_2, crop_global, normalize)
        self.local_transfo = parse_transforms(cfg.local, crop_local, normalize)

    def __call__(self, image):
        output = {}

        # Global crops
        global_crop_1 = self.global_transfo1(image)
        global_crop_2 = self.global_transfo2(image)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # Global crops for teacher
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # Local crops
        local_crops = [self.local_transfo(image) for _ in range(self.local_crops_number)]
        
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output


def parse_transforms(arguments, crop, norm):
    transforms_list = []
    for arg in arguments:
        f = arg.split("_")
        name = f[0]
        if name == "crop":
            transforms_list.append(crop)
        else:
            transforms_list.append(TRANSFORMS[name](*f[1:]))
    transforms_list.append(norm)
    return transforms.Compose(transforms_list)

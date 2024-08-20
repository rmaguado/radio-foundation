# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from torchvision import transforms

from .transforms import transformkeys


logger = logging.getLogger("dinov2")


err_not_recognized = "Transform '{s}' is not recognized. \
Please check the config file under augmentations."


class DataAugmentationDINO(object):
    def __init__(self, cfg, use_full_image: bool):
        self.local_crops_number = cfg.crops.local_crops_number

        self.global1, self.global2, self.local = load_transforms_from_cfg(cfg, use_full_image)

    def __call__(self, image):
        output = {}

        global_crop_1 = self.global(image)
        global_crop_2 = self.global2(image)

        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        local_crops = [self.local(image) for _ in range(self.crops.local_crops_number)]
        
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output

def get_local_crop(cfg):
    return transforms.RandomResizedCrop(
        cfg.crops.local_crops_size,
        scale=cfg.crops.local_crops_scale,
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True
    )

def get_global_crop(cfg, use_full_image: bool):
    global_crops_size = cfg.student.full_image_size \
        if use_full_image else cfg.crops.global_crops_size
        
    return transforms.RandomResizedCrop(
        global_crops_size,
        scale=cfg.crops.global_crops_scale,
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True
    )

def load_transforms_from_cfg(cfg, use_full_image: bool):
    def create_transform(transform_cfg):
        name = transform_cfg['name']
        if name == "localcrop":
            return localcrop
        elif name == "globalcrop":
            return globalcrop
        elif name in transformkeys:
            transform_cls = transformkeys[name]
            params = {k: float(v) if k != "name" else v for k, v in transform_cfg.items()}
            return transform_cls(**params)
        else:
            raise ValueError(err_not_recognized.format(s=name))

    def build_transform_group(transform_group):
        transforms_list = [create_transform(tc) for tc in cfg.augmentations[transform_group]]
        transforms_list.append(normalize)
        return transforms.Compose(transforms_list)
    
    localcrop = get_local_crop(cfg)
    globalcrop = get_global_crop(cfg, use_full_image)
    normalize = transforms.Normalize(mean=cfg.norm.mean, std=cfg.norm.std)

    transform_groups = [build_transform_group(group) for group in ["global_1", "global_2", "local"]]
    return tuple(transform_groups)


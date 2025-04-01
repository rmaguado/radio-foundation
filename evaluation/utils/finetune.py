import os
import torch
import torch.nn as nn

from functools import partial
from omegaconf import OmegaConf
from torchvision import transforms

from dinov2.models import build_model_from_cfg
from dinov2.utils.utils import load_pretrained_weights


class ImageTransform:
    def __init__(self, img_size, mean, std):
        self.img_size = img_size
        self.resize = transforms.Resize((img_size, img_size))
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, image):
        resized = self.resize(image)
        normalized = self.normalize(resized)
        return normalized


class ImageTransformResampleSlices:
    def __init__(self, img_size, mean, std, target_slices=240, channels=10):
        self.img_size = img_size
        self.resize = transforms.Resize((img_size, img_size))
        self.normalize = transforms.Normalize(mean=mean, std=std)

        assert isinstance(target_slices, int), "target_slices must be an integer"

        self.target_slices = target_slices
        self.channels = channels

    def resample_z(self, image):
        groups, channels, w, h = image.shape
        image = image.view(1, 1, groups * channels, w, h)
        image = torch.nn.functional.interpolate(
            image, size=(self.target_slices, w, h), mode="trilinear"
        )
        return image.view(-1, self.channels, w, h)

    def __call__(self, image):
        image = self.resize(image)
        image = self.resample_z(image)
        image = self.normalize(image)
        return image


class ImageTargetTransform:
    def __init__(self, img_size, mean, std):
        self.img_size = img_size
        self.resize = transforms.Resize((img_size, img_size))
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, image, target):
        image_resized = self.resize(image)
        target_resized = self.resize(target)
        return self.normalize(image_resized), target_resized


class ModelWithIntermediateLayers(nn.Module):
    """
    Copyright (c) Meta Platforms, Inc. and affiliates.

    This source code is licensed under the Apache License, Version 2.0
    found in the LICENSE file in the root directory of this source tree.

    taken from from dinov2.eval.utils.py
    """

    def __init__(self, feature_model, n_last_blocks, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images, self.n_last_blocks, return_class_token=True
                )
        return features


def extract_class_tokens(x_tokens_list, n_last_blocks=4):
    class_tokens = torch.cat(
        [class_token for _, class_token in x_tokens_list[-n_last_blocks:]],
        dim=-1,
    )
    class_tokens = class_tokens.unsqueeze(1)
    return class_tokens


def extract_patch_tokens(x_tokens_list, n_last_blocks=4):
    patch_tokens = torch.cat(
        [
            layer_patch_tokens
            for layer_patch_tokens, _ in x_tokens_list[-n_last_blocks:]
        ],
        dim=-1,
    )
    return patch_tokens


def extract_all_tokens(x_tokens_list, n_last_blocks=4):
    all_tokens = torch.cat(
        [
            torch.cat([class_token.unsqueeze(1), layer_patch_tokens], dim=1)
            for layer_patch_tokens, class_token in x_tokens_list[-n_last_blocks:]
        ],
        dim=-1,
    )
    return all_tokens


def get_config(path_to_run):
    path_to_config = os.path.join(path_to_run, "config.yaml")
    return OmegaConf.load(path_to_config)


def get_autocast_dtype(cfg):
    teacher_dtype_str = (
        cfg.compute_precision.teacher.backbone.mixed_precision.param_dtype
    )
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


def load_model(path_to_run, checkpoint_name, device, intermediate_layers=4):
    path_to_checkpoint = os.path.join(
        path_to_run, "eval", checkpoint_name, "teacher_checkpoint.pth"
    )

    config = get_config(path_to_run)

    model, _ = build_model_from_cfg(config, only_teacher=True)
    load_pretrained_weights(model, path_to_checkpoint, "teacher")
    model.eval()
    model.to(device)

    autocast_dtype = get_autocast_dtype(config)
    autocast_ctx = partial(
        torch.autocast, enabled=True, dtype=autocast_dtype, device_type="cuda"
    )
    feature_model = ModelWithIntermediateLayers(
        model, intermediate_layers, autocast_ctx
    )

    return feature_model, config

import os
import torch
import torch.nn as nn

from functools import partial
from omegaconf import OmegaConf
from torchvision import transforms
from einops import rearrange

from dinov2.models import build_model_from_cfg


class ImageTransform:
    def __init__(
        self,
        img_size,
        mean,
        std,
        channels,
        zspacing=None,
        max_slices=None,
        pad_value=-1000,
    ):
        self.img_size = img_size
        self.normalize = transforms.Normalize(mean=mean, std=std)

        self.channels = channels
        self.pad_value = pad_value

        self.zspacing = zspacing
        self.max_slices = max_slices

        if max_slices is not None:
            assert (
                max_slices % channels == 0
            ), "max_slices must be divisible by channels"

    def pad_square(self, image):
        s, w, h = image.shape

        if w == h:
            return image

        if w > h:
            pad_1 = (w - h) // 2
            pad_2 = (w - h) - pad_1
            image = torch.nn.functional.pad(
                image, (pad_1, pad_2, 0, 0, 0, 0), value=self.pad_value
            )
        else:
            pad_1 = (h - w) // 2
            pad_2 = (h - w) - pad_1
            image = torch.nn.functional.pad(
                image, (0, 0, pad_1, pad_2, 0, 0), value=self.pad_value
            )

        return image

    def clip_slices(self, image):
        slices, w, h = image.shape

        if slices > self.max_slices:
            start = (slices - self.max_slices) // 2
            end = start + self.max_slices

            image = image[start:end]

        return image

    def resize(self, image, slice_thickness=None):
        slices, w, h = image.shape

        target_width = self.img_size if w >= h else self.img_size * w // h
        target_height = self.img_size if h >= w else self.img_size * h // w

        if slice_thickness is None:
            target_slices = slices
        else:
            target_slices = int(slices * slice_thickness / self.zspacing)

        groups = target_slices // self.channels
        target_slices = groups * self.channels

        image = image.unsqueeze(0).unsqueeze(0)
        image = torch.nn.functional.interpolate(
            image, size=(target_slices, target_width, target_height), mode="trilinear"
        ).squeeze()
        image = self.pad_square(image)
        image = self.clip_slices(image)
        return rearrange(
            image,
            "(g c) w h -> g c w h",
            c=self.channels,
            w=self.img_size,
            h=self.img_size,
        )

    def __call__(self, image, slice_thickness=None):
        image = self.resize(image, slice_thickness)
        image = self.normalize(image)
        return image


class ModelWithIntermediateLayers(nn.Module):
    """
    Copyright (c) Meta Platforms, Inc. and affiliates.

    This source code is licensed under the Apache License, Version 2.0
    found in the LICENSE file in the root directory of this source tree.
    """

    def __init__(self, feature_model, select_layers: int | List[int], autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.select_layers = select_layers
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images, self.select_layers, return_class_token=True
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


def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    
    torch.load(
        pretrained_weights, map_location="cpu", weights_only=False
    )


def load_model_eval(path_to_checkpoint, config, device, select_layers):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    load_pretrained_weights(model, path_to_checkpoint, "teacher")
    model.eval()
    model.to(device)

    autocast_dtype = get_autocast_dtype(config)
    autocast_ctx = partial(
        torch.autocast, enabled=True, dtype=autocast_dtype, device_type="cuda"
    )
    feature_model = ModelWithIntermediateLayers(model, select_layers, autocast_ctx)

    return feature_model


if __name__ == "__main__":
    path_to_run = "runs/test"
    path_to_checkpoint = os.path.join(
        path_to_run, "eval", checkpoint_name, "teacher_checkpoint.pth"
    )
    config = get_config(path_to_run)
    device = torch.device("cuda")

    model = load_model_eval(path_to_checkpoint, config, device, select_layers=[11])

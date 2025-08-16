import os
import torch
import torch.nn as nn

from functools import partial
from omegaconf import OmegaConf

from dino.models import build_model_eval


class ModelWithIntermediateLayers(nn.Module):
    """
    Copyright (c) Meta Platforms, Inc. and affiliates.

    This source code is licensed under the Apache License, Version 2.0
    found in the LICENSE file in the root directory of this source tree.
    """

    def __init__(self, feature_model, select_layers, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.select_layers = select_layers
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images, select_layers=self.select_layers
                )
        return features


class Model(nn.Module):
    def __init__(self, feature_model, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.feature_model(images)
        return features


def get_config(path_to_config):
    return OmegaConf.load(path_to_config)


def get_autocast_dtype(cfg):
    teacher_dtype_str = cfg.compute_precision
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


def load_model_intermediates(path_to_checkpoint, config, device, select_layers):
    model = build_model_eval(config)

    state_dict = torch.load(path_to_checkpoint, map_location="cpu")["teacher"]
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("dino_head")}
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)

    autocast_dtype = get_autocast_dtype(config)
    autocast_ctx = partial(
        torch.autocast, enabled=True, dtype=autocast_dtype, device_type="cuda"
    )
    feature_model = ModelWithIntermediateLayers(model, select_layers, autocast_ctx)

    return feature_model


def load_model(path_to_checkpoint, config, device):
    model = build_model_eval(config)

    state_dict = torch.load(path_to_checkpoint, map_location="cpu")["teacher"]
    state_dict = {
        k.removeprefix("backbone."): v
        for k, v in state_dict.items()
        if not k.startswith("dino_head")
    }
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)

    autocast_dtype = get_autocast_dtype(config)
    autocast_ctx = partial(
        torch.autocast, enabled=True, dtype=autocast_dtype, device_type="cuda"
    )
    feature_model = Model(model, autocast_ctx)

    return feature_model


if __name__ == "__main__":
    path_to_run = "runs/test"
    checkpoint_name = "training_99999"
    path_to_checkpoint = os.path.join(
        path_to_run, "eval", checkpoint_name, "teacher_checkpoint.pth"
    )
    config = get_config(path_to_run)
    device = torch.device("cuda")

    model = load_model(path_to_checkpoint, config, device)

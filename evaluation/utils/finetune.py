import os
import torch
import torch.nn as nn
import numpy as np

from functools import partial
from omegaconf import OmegaConf

from dinov2.models import build_model_from_cfg
from dinov2.utils.utils import load_pretrained_weights


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

class LinearClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_labels, dropout=0.5):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x):
        return self.mlp(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class DinoClassifier(nn.Module):
    USE_N_BLOCKS = 4

    def __init__(
        self,
        feature_model,
        embed_dim=384 * 4,
        hidden_dim=2048,
        num_labels=2,
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.feature_model = feature_model
        self.classifier = LinearClassifier(embed_dim, hidden_dim, num_labels).to(device)

    def forward(self, image):
        with torch.no_grad():
            x_tokens_list = self.feature_model(image)
        intermediate_output = x_tokens_list[-DinoClassifier.USE_N_BLOCKS :]
        class_tokens = torch.cat(
            [class_token for _, class_token in intermediate_output], dim=-1
        )
        return self.classifier(class_tokens)


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


def load_model(path_to_run, checkpoint_name, device):
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
    feature_model = ModelWithIntermediateLayers(model, 4, autocast_ctx)

    return feature_model, config


def multiclass_accuracy_logits(outputs, targets):
    predicted_labels = torch.argmax(outputs.detach().cpu(), dim=1)
    correct_predictions = predicted_labels == targets.cpu()
    return correct_predictions.sum().item() / targets.size(0)


def binary_accuracy_logits(outputs, targets):
    predicted_labels = (outputs > 0).int()
    targets = targets.int()

    true_pred_positives = (predicted_labels * targets).sum().item()
    true_pred_negatives = ((1 - predicted_labels) * (1 - targets)).sum().item()

    positives = targets.sum().item()
    negatives = targets.numel() - positives

    accuracy = (predicted_labels == targets).float().mean().item()

    return accuracy, [positives, true_pred_positives, negatives, true_pred_negatives]


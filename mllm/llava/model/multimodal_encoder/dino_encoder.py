import torch
import torch.nn as nn
from omegaconf import OmegaConf
from functools import partial
import logging

from dinov2.models import build_model_from_cfg
from dinov2.utils.utils import load_pretrained_weights


logger = logging.getLogger("DeepSpeed")


class DINOVisionTower(nn.Module):
    def __init__(self, vision_tower, args, torch_dtype, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.model_config = OmegaConf.load(args.mm_vision_config_path)
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        self.use_cls = self.select_feature in ["cls", "cls_patch"]
        if self.select_feature == "cls":
            self.extract_fnc = self.extract_cls
        elif self.select_feature == "cls_patch":
            self.extract_fnc = self.extract_cls_patch
        elif self.select_feature == "patch":
            self.extract_fnc = self.extract_patch
        else:
            raise ValueError(
                f"Feature to extract from vision tower '{self.select_feature}' not recognized"
            )

        self.path_to_checkpoint = args.mm_vision_checkpoint_path

        if not delay_load:
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            self.load_model()
        else:
            self.cfg_only = self.model_config

        self.output_type = torch_dtype

    def load_model(self):
        if self.is_loaded:
            print(
                "{} is already loaded, `load_model` called again, skipping.".format(
                    self.vision_tower_name
                )
            )
            return

        self.vision_tower, _ = build_model_from_cfg(
            self.model_config, only_teacher=True
        )
        load_pretrained_weights(self.vision_tower, self.path_to_checkpoint, "teacher")

        if self.select_layer < 0:
            self.select_layer = self.vision_tower.n_blocks + self.select_layer

        logger.info(f"Chosen select layer for vision tower: {self.select_layer}.")

        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def extract_cls_patch(self, img_features):
        return torch.cat(
            [
                torch.cat([class_token.unsqueeze(1), layer_patch_tokens], dim=1)
                for layer_patch_tokens, class_token in img_features
            ],
            dim=-1,
        )

    def extract_patch(self, img_features):
        return torch.cat(
            [layer_patch_tokens for layer_patch_tokens, _ in img_features],
            dim=-1,
        )

    def extract_cls(self, img_features):
        return torch.cat(
            [class_token for _, class_token in img_features],
            dim=-1,
        ).unsqueeze(1)

    def forward(self, images: list):
        features = []

        for img in images:

            img_features = []

            x_tokens = self.vision_tower.get_intermediate_layers(
                img, [self.select_layer], return_class_token=True
            )
            feat = self.extract_fnc(x_tokens)
            features.append(feat)

        return features

    @property
    def dummy_feature(self):
        num_tokens = self.num_patches
        if self.use_cls:
            num_tokens += 1

        return torch.zeros(
            1, num_tokens, self.hidden_size, device=self.device, dtype=self.dtype
        )

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.model_config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.vision_tower.embed_dim

    @property
    def num_patches_per_side(self):
        return (
            self.model_config.student.full_image_size
            // self.model_config.student.patch_size
        )

    @property
    def num_patches(self):
        return (
            self.model_config.student.full_image_size
            // self.model_config.student.patch_size
        ) ** 2

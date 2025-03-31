import torch
import torch.nn as nn
from omegaconf import OmegaConf
from functools import partial

from dinov2.models import build_model_from_cfg
from dinov2.utils.utils import load_pretrained_weights


class DINOVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.model_config = OmegaConf.load(args.mm_vision_config_path)
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        self.use_cls = self.select_feature == "cls_patch"
        self.path_to_checkpoint = args.mm_vision_checkpoint_path

        self.load_model()

        if not delay_load:
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            self.load_model()
        else:
            self.cfg_only = self.model_config

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

        teacher_dtype_str = (
            self.model_config.compute_precision.teacher.backbone.mixed_precision.param_dtype
        )
        if teacher_dtype_str == "fp16":
            autocast_dtype = torch.half
        elif teacher_dtype_str == "bf16":
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = torch.float

        self.autocast_ctx = partial(
            torch.autocast, enabled=True, dtype=autocast_dtype, device_type="cuda"
        )

        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images: list):
        features = []
        with self.autocast_ctx():
            for img in images:
                img_features = self.vision_tower.get_intermediate_layers(
                    img, [self.select_layer], return_class_token=self.use_cls
                )
                features.append(img_features[0])

        print(features[0].shape)

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
        return self.model_config.student.embed_dim

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

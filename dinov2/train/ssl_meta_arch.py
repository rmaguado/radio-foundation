# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging
from typing import Dict, Any, Tuple

import torch
from torch import nn

from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov2.models import build_model_from_cfg
from dinov2.layers import DINOHead


try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


logger = logging.getLogger("dinov2")


class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        student_model_dict = dict()
        teacher_model_dict = dict()

        student_backbone, teacher_backbone = build_model_from_cfg(cfg)
        if student_backbone is None:
            raise ValueError(
                "student_backbone is None. Check build_model_from_cfg(cfg)."
            )
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone

        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights)
            student_backbone.load_state_dict(chkpt["model"], strict=False)

        self.embed_dim = student_backbone.embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head

        self.dino_loss_weight = cfg.dino.loss_weight
        dino_head = partial(
            DINOHead,
            in_dim=self.embed_dim,
            out_dim=cfg.dino.head_n_prototypes,
            hidden_dim=cfg.dino.head_hidden_dim,
            bottleneck_dim=cfg.dino.head_bottleneck_dim,
            nlayers=cfg.dino.head_nlayers,
        )
        student_model_dict["dino_head"] = dino_head()
        teacher_model_dict["dino_head"] = dino_head()

        self.dino_loss = DINOLoss(self.dino_out_dim)
        if self.do_koleo:
            self.koleo_loss = KoLeoLoss()

        if self.do_ibot:
            self.ibot_loss_weight = cfg.ibot.loss_weight
            self.ibot_out_dim = (
                cfg.ibot.head_n_prototypes
                if self.ibot_separate_head
                else cfg.dino.head_n_prototypes
            )
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            if self.ibot_separate_head:
                ibot_head = partial(
                    DINOHead,
                    in_dim=self.embed_dim,
                    out_dim=cfg.ibot.head_n_prototypes,
                    hidden_dim=cfg.ibot.head_hidden_dim,
                    bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                    nlayers=cfg.ibot.head_nlayers,
                )
                student_model_dict["ibot_head"] = ibot_head()
                teacher_model_dict["ibot_head"] = ibot_head()
            else:
                student_model_dict["ibot_head"] = student_model_dict["dino_head"]
                teacher_model_dict["ibot_head"] = teacher_model_dict["dino_head"]

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.view_metadata = {}
        self.target_group_names = []
        self.student_group_names = []
        for group_cfg in self.cfg.crops.crop_groups:
            group_cfg_copy = group_cfg.copy()
            name = group_cfg_copy.pop("name")
            self.view_metadata[name] = group_cfg_copy
            self.student_group_names.append(name)
            if group_cfg_copy.get("is_target", False):
                self.target_group_names.append(name)

    def forward(self, inputs):
        raise NotImplementedError(
            "forward method is not implemented in SSLMetaArch. Use forward_backward instead."
        )

    def forward_backward(self, collated_views, teacher_temp):
        image_views = {
            k: v["images"].cuda(non_blocking=True) for k, v in collated_views.items()
        }
        masks = {
            k: v["masks"].cuda(non_blocking=True) for k, v in collated_views.items()
        }

        teacher_dino_tokens = {}
        teacher_ibot_tokens = {}

        with torch.no_grad():
            for group_name in self.target_group_names:
                group_data = image_views[group_name]
                group_masks = masks[group_name]
                embed_layer = collated_views[group_name]["embed_layer"]

                B, V = group_data.shape[:2]
                flattened_group_data = rearrange(group_data, "b v ... -> (b v) ...")
                flattened_group_masks = rearrange(group_masks, "b v ... -> (b v) (...)")

                teacher_output = self.teacher["backbone"](
                    group_data, embed_layer=embed_layer, masks=None
                )

                teacher_cls_tokens = teacher_output["clstoken"]
                teacher_dino_tokens_flattened = self.teacher.dino_head(
                    teacher_cls_tokens
                )
                teacher_dino_tokens_flattened_centered = (
                    self.dino_loss.softmax_center_teacher(
                        teacher_dino_tokens_flattened, teacher_temp
                    )
                )

                teacher_dino_tokens[group_name] = rearrange(
                    teacher_dino_tokens_flattened_centered, "(b v) d -> b v d", b=B, v=V
                )

                if self.do_ibot:
                    teacher_patch_tokens = teacher_output["patchtokens"]
                    teacher_masked_patch_tokens = torch.masked_select(
                        teacher_patch_tokens,
                        flattened_group_masks,
                    )
                    teacher_ibot_tokens_flattened = self.teacher.ibot_head(
                        teacher_masked_patch_tokens_centered
                    )
                    teacher_ibot_tokens_flattened_centered = (
                        self.ibot_patch_loss.softmax_center_teacher(
                            teacher_ibot_tokens_flattened, teacher_temp
                        )
                    )
                    teacher_ibot_tokens[group_name] = (
                        teacher_ibot_tokens_flattened_centered
                    )

        student_dino_tokens = {}
        student_ibot_tokens = {}
        for group_name in image_views.keys():
            group_data = image_views[group_name]
            group_masks = masks[group_name]
            embed_layer = collated_views[group_name]["embed_layer"]

            is_target = collated_views[group_name]["is_target"]
            if is_target:
                B, V = group_data.shape[:2]
                flattened_group_data = rearrange(group_data, "b v ... -> (b v) ...")
                flattened_group_masks = rearrange(group_masks, "b v ... -> (b v) (...)")
                student_output = self.student["backbone"](
                    flattened_group_data,
                    embed_layer=embed_layer,
                    masks=flattened_group_masks,
                )
                student_cls_tokens = student_output["clstoken"]
                student_dino_tokens_flattened = self.student.dino_head(
                    student_cls_tokens
                )
                student_dino_tokens[group_name] = rearrange(
                    student_dino_tokens_flattened, "(b v) d -> b v d", b=B, v=V
                )
                if self.do_ibot:
                    student_patch_tokens = student_output["patchtokens"]
                    student_masked_patch_tokens = torch.masked_select(
                        student_patch_tokens,
                        flattened_group_masks,
                    )
                    student_ibot_tokens[group_name] = self.student.ibot_head(
                        student_masked_patch_tokens
                    )
            else:
                B, V_target, V = group_data.shape[:3]
                flattened_group_data = rearrange(group_data, "b t v... -> (b t v) ...")
                student_output = self.student["backbone"](
                    flattened_group_data,
                    embed_layer=embed_layer,
                )
                student_cls_tokens = student_output["clstoken"]
                student_dino_tokens_flattened = self.student.dino_head(
                    student_cls_tokens
                )
                student_dino_tokens[group_name] = rearrange(
                    student_dino_tokens_flattened,
                    "(b t v) d -> b t v d",
                    b=B,
                    t=V_target,
                    v=V,
                )

        loss_accumulator = 0.0
        loss_dict = {}

        total_dino_loss = 0
        total_dino_terms = 0

        for group_name in image_views.keys():
            student_dino_tokens_group = student_dino_tokens[group_name]
            targets = collated_views[group_name]["targets"]
            is_target = collated_views[group_name]["is_target"]

            for i, target_group in enumerate(targets):
                student_dino_tokens_target = (
                    student_dino_tokens_group
                    if is_target
                    else student_dino_tokens_group[:, i, ...]
                )
                teacher_dino_tokens_target = teacher_dino_tokens[target_group]
                num_comparisons = (
                    teacher_dino_tokens_target.shape[1]
                    * student_dino_tokens_target.shape[1]
                )
                dino_loss_term = self.dino_loss(
                    student_dino_tokens_target,
                    teacher_dino_tokens_target,
                    group_name,
                )
                total_dino_loss += dino_loss_term
                total_dino_terms += num_comparisons

        loss_accumulator += self.dino_loss_weight * total_dino_loss / total_dino_terms
        loss_dict["dino_loss"] = total_dino_loss / total_dino_terms

        if self.do_ibot:
            pass

        loss_accumulator.backward()

        return loss_dict

    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                for ms, mt in zip(self.student[k].modules(), self.teacher[k].modules()):
                    student_param_list += ms.params
                    teacher_param_list += mt.params
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def train(self, mode):  # type: ignore
        super().train(mode)
        self.teacher.eval()

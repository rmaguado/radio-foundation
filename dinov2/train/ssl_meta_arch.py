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

        loss_accumulator = 0.0
        loss_dict = {}

        @torch.no_grad()
        def get_teacher_output() -> Tuple[Dict[str, Any], Any]:

            teacher_backbone_output = self.teacher.backbone(
                teacher_inputs, is_training=True
            )
            teacher_cls_tokens = teacher_backbone_output["clstoken"]

            # iBOT setup
            teacher_patch_tokens = teacher_backbone_output["patchtokens"]
            _dim = teacher_patch_tokens.shape[-1]

            # Use separate heads if configured
            if self.do_ibot and self.ibot_separate_head:
                teacher_cls_tokens_after_head = self.teacher.dino_head(
                    teacher_cls_tokens
                )

                buffer_tensor_teacher = teacher_patch_tokens.new_zeros(upperbound, _dim)
                torch.index_select(
                    teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[:n_masked_patches],
                )
                masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(
                    buffer_tensor_teacher
                )[:n_masked_patches]

            # Use a single head for both cls and patch tokens
            elif self.do_ibot:
                n_cls = teacher_cls_tokens.shape[0]
                buffer_tensor_teacher = teacher_patch_tokens.new_zeros(
                    upperbound + n_cls, _dim
                )
                buffer_tensor_teacher[:n_cls].copy_(teacher_cls_tokens)
                torch.index_select(
                    teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[n_cls : n_cls + n_masked_patches],
                )
                tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                teacher_cls_tokens_after_head = tokens_after_head[:n_cls]
                masked_teacher_patch_tokens_after_head = tokens_after_head[
                    n_cls : n_cls + n_masked_patches
                ]
            # No iBOT
            else:
                teacher_cls_tokens_after_head = self.teacher.dino_head(
                    teacher_cls_tokens
                )
                masked_teacher_patch_tokens_after_head = None

            # Centering and Softmax
            # DINO cls tokens
            teacher_dino_softmaxed_centered = self.dino_loss.softmax_center_teacher(
                teacher_cls_tokens_after_head, teacher_temp
            )
            self.dino_loss.update_center(teacher_cls_tokens_after_head)

            # iBOT patch tokens
            masked_teacher_ibot_softmaxed_centered = None
            if self.do_ibot:
                masked_teacher_ibot_softmaxed_centered = (
                    self.ibot_patch_loss.softmax_center_teacher(
                        masked_teacher_patch_tokens_after_head, teacher_temp
                    )
                )
                self.ibot_patch_loss.update_center(
                    masked_teacher_patch_tokens_after_head
                )

            # De-interleave the teacher outputs and store them by group name
            teacher_outputs = {}
            cls_tokens_split = teacher_dino_softmaxed_centered.split(
                [
                    self.view_metadata[name]["num_crops"]
                    for name in self.target_group_names
                ]
            )
            for i, name in enumerate(self.target_group_names):
                teacher_outputs[name] = cls_tokens_split[i]

            return teacher_outputs, masked_teacher_ibot_softmaxed_centered

        # Execute teacher pass
        teacher_outputs, masked_teacher_ibot_softmaxed_centered = get_teacher_output()

        # Student forward pass (on all views)

        student_inputs_list = [
            collated_views[name] for name in self.student_group_names
        ]
        # The mask is applied only to target views
        masks_list = [
            masks if self.view_metadata[name]["is_target"] else None
            for name in self.student_group_names
        ]

        # Run student backbone
        student_backbone_outputs = self.student.backbone(
            student_inputs_list, masks=masks_list, is_training=True
        )

        # Prepare inputs for the student head
        student_head_inputs = []
        student_outputs_pre_head = {}

        # De-interleave backbone outputs
        cls_tokens_split = student_backbone_outputs["clstoken"].split(
            [self.view_metadata[name]["num_crops"] for name in self.student_group_names]
        )
        patch_tokens_split = student_backbone_outputs["patchtokens"].split(
            [self.view_metadata[name]["num_crops"] for name in self.student_group_names]
        )

        for i, name in enumerate(self.student_group_names):
            student_outputs_pre_head[name] = {
                "cls": cls_tokens_split[i],
                "patch": patch_tokens_split[i],
            }
            student_head_inputs.append(cls_tokens_split[i].unsqueeze(0))

        # Add masked patch tokens for iBOT to the head input list
        if self.do_ibot:
            # Gather patch tokens from all target groups that were masked
            student_ibot_patch_tokens = torch.cat(
                [
                    student_outputs_pre_head[name]["patch"]
                    for name in self.target_group_names
                ],
                dim=0,
            )

            _dim = student_ibot_patch_tokens.shape[-1]
            buffer_tensor_patch_tokens = student_ibot_patch_tokens.new_zeros(
                upperbound, _dim
            )
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(
                    student_ibot_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                )
            )

            if self.ibot_separate_head:  # With separate head, process patch tokens now
                student_masked_patch_tokens_after_head = self.student.ibot_head(
                    buffer_tensor_patch_tokens
                )[:n_masked_patches]
            else:
                student_head_inputs.append(buffer_tensor_patch_tokens.unsqueeze(0))

        # NOTE: Using a placeholder for BlockDiagonalMask if xformers is not installed.
        # This part might need adjustment depending on your `fmha` implementation.
        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(
            student_head_inputs
        )
        head_outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))

        # De-interleave head outputs
        student_outputs_after_head = {}
        for i, name in enumerate(self.student_group_names):
            student_outputs_after_head[name] = {"cls": head_outputs_list[i].squeeze(0)}

        if self.do_ibot and not self.ibot_separate_head:
            student_masked_patch_tokens_after_head = head_outputs_list[-1].squeeze(0)[
                :n_masked_patches
            ]

        # loss calculation

        total_dino_loss_terms = 0
        total_dino_loss = 0
        for group_name, student_data in student_outputs_after_head.items():
            student_cls_tokens = student_data["cls"]
            target_names = self.view_metadata[group_name]["targets"]

            # Prepare the list of teacher tensors for this student group
            teacher_targets_for_student = [
                teacher_outputs[t_name] for t_name in target_names
            ]

            # The number of comparisons for this view group
            num_crops_student = self.view_metadata[group_name]["num_crops"]
            num_comparisons = sum(
                self.view_metadata[t_name]["num_crops"] for t_name in target_names
            )
            total_dino_loss_terms += num_crops_student * num_comparisons

            # --- THIS IS THE SIMPLIFIED CALL ---
            # No more local helper function needed. Call the DINOLoss instance directly.
            group_loss = self.dino_loss(student_cls_tokens, teacher_targets_for_student)

            total_dino_loss += group_loss
            loss_dict[f"dino_loss_{group_name}"] = group_loss / (
                num_crops_student * num_comparisons
            )

        # Normalize and accumulate DINO loss
        dino_loss = total_dino_loss / total_dino_loss_terms
        loss_accumulator += self.dino_loss_weight * dino_loss
        loss_dict["dino_total_loss"] = dino_loss

        # iBOT Loss
        if self.do_ibot:
            ibot_loss = self.ibot_patch_loss.forward_masked(
                student_masked_patch_tokens_after_head,
                masked_teacher_ibot_softmaxed_centered,
                student_masks_flat=masks,
                n_masked_patches=n_masked_patches,
                masks_weight=masks_weight,
            )

            # Normalize by the number of target views used for iBOT
            ibot_loss = ibot_loss / len(self.target_group_names)
            loss_dict["ibot_loss"] = ibot_loss
            loss_accumulator += self.ibot_loss_weight * ibot_loss

        # Backpropagation

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

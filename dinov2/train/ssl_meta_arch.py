# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging
from typing import Dict, Any, Tuple

import torch
from torch import nn
from einops import rearrange

from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov2.models import build_model_from_cfg
from dinov2.layers import DINOHead
from dinov2.utils.param_groups import get_params_groups_with_decay


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
            self.koleo_loss_weight = cfg.dino.koleo_loss_weight
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

    def _prepare_inputs(
        self, collated_views: Dict[str, Any]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Moves images and masks to the appropriate device."""
        images = {
            k: v["images"].cuda(non_blocking=True) for k, v in collated_views.items()
        }
        masks = {
            k: v["masks"].cuda(non_blocking=True) for k, v in collated_views.items()
        }
        return images, masks

    def _process_group(
        self,
        model: nn.Module,
        images: torch.Tensor,
        masks: torch.Tensor,
        embed_layer: int,
        is_target: bool,
    ) -> Dict[str, torch.Tensor]:
        """
        Processes a single group of views through a model (student or teacher).
        Handles different tensor shapes for target vs. non-target views.
        """
        if is_target:
            B, V = images.shape[:2]
            V_target = None
            flat_images = rearrange(images, "b v ... -> (b v) ...")
            flat_masks = (
                rearrange(masks, "b v ... -> (b v) (...)")
                if masks is not None
                else None
            )
            backbone_masks = flat_masks
        else:
            B, V_target, V = images.shape[:3]
            flat_images = rearrange(images, "b t v ... -> (b t v) ...")
            backbone_masks = None

        backbone_output = model.backbone(
            flat_images,
            embed_layer=embed_layer,
            masks=backbone_masks,
        )

        student_cls_tokens = backbone_output["clstoken"]
        dino_tokens_flat = model.dino_head(student_cls_tokens)

        if is_target:
            dino_tokens = rearrange(dino_tokens_flat, "(b v) d -> b v d", b=B, v=V)
        else:
            dino_tokens = rearrange(
                dino_tokens_flat, "(b t v) d -> b t v d", b=B, t=V_target, v=V
            )

        output = {"dino": dino_tokens}

        if self.do_ibot and is_target:
            patch_tokens = backbone_output["patchtokens"]
            masked_patch_tokens = torch.masked_select(patch_tokens, flat_masks)  # type: ignore

            ibot_head = model.ibot_head if self.ibot_separate_head else model.dino_head
            output["ibot"] = ibot_head(masked_patch_tokens)

        return output

    def _run_teacher_pass(
        self,
        images: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
        collated_views: Dict[str, Any],
        teacher_temp: float,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Runs the teacher model and computes centered tokens."""
        teacher_outputs = {"dino": {}, "ibot": {}}
        with torch.no_grad():
            for group_name in self.target_group_names:
                # Process the group to get raw tokens
                group_output = self._process_group(
                    model=self.teacher,
                    images=images[group_name],
                    masks=masks[group_name],
                    embed_layer=collated_views[group_name]["embed_layer"],
                    is_target=True,  # Teacher only sees target views
                )

                # Center DINO tokens
                dino_tokens_centered = self.dino_loss.softmax_center_teacher(
                    group_output["dino"], teacher_temp
                )
                teacher_outputs["dino"][group_name] = dino_tokens_centered

                # Center iBOT tokens
                if self.do_ibot:
                    ibot_tokens_centered = self.ibot_patch_loss.softmax_center_teacher(
                        group_output["ibot"], teacher_temp
                    )
                    teacher_outputs["ibot"][group_name] = ibot_tokens_centered
        return teacher_outputs

    def _run_student_pass(
        self,
        images: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
        collated_views: Dict[str, Any],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Runs the student model across all view groups."""
        student_outputs = {"dino": {}, "ibot": {}}
        for group_name in images.keys():
            group_output = self._process_group(
                model=self.student,
                images=images[group_name],
                masks=masks[group_name],
                embed_layer=collated_views[group_name]["embed_layer"],
                is_target=collated_views[group_name]["is_target"],
            )
            student_outputs["dino"][group_name] = group_output["dino"]
            if self.do_ibot and "ibot" in group_output:
                student_outputs["ibot"][group_name] = group_output["ibot"]
        return student_outputs

    def _calculate_dino_loss(
        self,
        student_dino_tokens: Dict[str, torch.Tensor],
        teacher_dino_tokens: Dict[str, torch.Tensor],
        collated_views: Dict[str, Any],
    ) -> torch.Tensor:
        """Calculates the total DINO loss across all student-teacher view pairs."""
        total_loss = torch.tensor(0.0)
        total_terms = 0

        for group_name, s_tokens in student_dino_tokens.items():
            is_target = collated_views[group_name]["is_target"]
            targets = collated_views[group_name]["targets"]

            for i, target_group_name in enumerate(targets):
                s_tokens_for_comparison = s_tokens if is_target else s_tokens[:, i, ...]
                t_tokens = teacher_dino_tokens[target_group_name]

                loss_term: torch.Tensor = self.dino_loss(
                    s_tokens_for_comparison, t_tokens
                )
                num_comparisons: int = (
                    s_tokens_for_comparison.shape[1] * t_tokens.shape[1]
                )

                total_loss += loss_term
                total_terms += num_comparisons

        if total_terms == 0:
            logger.warning("No DINO comparisons were made. DINO loss is 0.")
            return torch.tensor(0.0)

        return total_loss / total_terms

    def _calculate_koleo_loss(
        self,
        student_dino_tokens: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Calculates the total KoLeo loss among student target views within a batch."""
        if not self.do_koleo:
            return torch.tensor(0.0)

        total_loss = torch.tensor(0.0)
        total_terms = 0

        # KoLeo loss is computed only on target views
        for group_name, s_tokens in student_dino_tokens.items():
            if not self.view_metadata[group_name].get("is_target", False):
                continue

            # Flatten the tokens to compute pairwise distances
            flat_s_tokens = rearrange(s_tokens, "b v d -> (b v) d")
            loss_term = self.koleo_loss(flat_s_tokens)

            total_loss += loss_term
            total_terms += 1

        if total_terms == 0:
            return torch.tensor(0.0)

        return total_loss / total_terms

    def _calculate_ibot_loss(
        self,
        student_ibot_tokens: Dict[str, torch.Tensor],
        teacher_ibot_tokens: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Calculates the total iBOT loss across all student-teacher view pairs."""
        if not self.do_ibot:
            return torch.tensor(0.0)

        total_loss = torch.tensor(0.0)
        total_terms = 0

        # iBOT loss is a direct comparison between student and teacher tokens from the same group
        for group_name, s_tokens in student_ibot_tokens.items():
            if group_name in teacher_ibot_tokens:
                t_tokens = teacher_ibot_tokens[group_name]
                loss_term = self.ibot_patch_loss(s_tokens, t_tokens)

                # Here we assume the loss is already averaged; if not, adjust accordingly
                total_loss += loss_term
                total_terms += 1

        if total_terms == 0:
            return torch.tensor(0.0)

        return total_loss / total_terms

    def forward(
        self, collated_views: Dict[str, Any], teacher_temp: float
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Main forward pass for DINOv2 training.
        """
        images, masks = self._prepare_inputs(collated_views)

        # Teacher forward pass (with no_grad)
        teacher_outputs = self._run_teacher_pass(
            images, masks, collated_views, teacher_temp
        )

        # Student forward pass
        student_outputs = self._run_student_pass(images, masks, collated_views)

        # Calculate losses
        dino_loss = self._calculate_dino_loss(
            student_outputs["dino"], teacher_outputs["dino"], collated_views
        )
        ibot_loss = self._calculate_ibot_loss(
            student_outputs["ibot"], teacher_outputs["ibot"]
        )
        koleo_loss = self._calculate_koleo_loss(student_outputs["dino"])

        total_loss = (
            (self.dino_loss_weight * dino_loss)
            + (self.ibot_loss_weight * ibot_loss)
            + (self.koleo_loss_weight * koleo_loss)
        )
        loss_dict = {
            "dino_loss": dino_loss.detach(),
            "ibot_loss": ibot_loss.detach(),
            "total_loss": total_loss.detach(),
        }

        return total_loss, loss_dict

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

    def get_params_groups(self):

        all_params_groups = []
        for m in self.student.values():
            all_params_groups += get_params_groups_with_decay(
                m,
                lr_decay_rate=self.cfg.optim.layerwise_decay,
                patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
            )
        return all_params_groups

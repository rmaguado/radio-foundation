# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging
from typing import Dict, Any, Tuple, Optional

import torch
from torch import nn
from einops import rearrange, repeat

from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov2.models import build_model_from_cfg
from dinov2.layers import DINOHead
from dinov2.train.param_groups import get_params_groups_with_decay


logger = logging.getLogger("dinov2")


class SSLMetaArch(nn.Module):
    """
    Self-supervised learning meta-architecture for DINO/iBOT-style training.

    This class manages the student and teacher networks, their heads, and the computation
    of DINO, iBOT, and KoLeo losses. It supports loading pretrained weights, updating teacher
    parameters, and orchestrating the forward pass for self-supervised learning.

    Args:
        cfg: Configuration object with all model and loss settings.
        device: Device to run the model on (e.g., 'cuda' or 'cpu').
    """

    def __init__(self, cfg, device, dtype):
        super().__init__()
        self.cfg = cfg
        self.device = device

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

        self.autocast_ctx = partial(
            torch.autocast, enabled=True, dtype=dtype, device_type="cuda"
        )

    def _prepare_inputs(self, collated_views: Dict[str, Any]) -> None:
        """
        Moves images and masks in the collated views dictionary to the correct device.

        Args:
            collated_views (Dict[str, Any]): Dictionary of collated batch data.
        """
        for group_name, view_info in collated_views.items():
            images = view_info["images"]
            masks = view_info.get("masks", None)

            images = images.to(self.device, non_blocking=True)
            collated_views[group_name]["images"] = images

            if masks is not None:
                masks = masks.to(self.device, non_blocking=True)
                collated_views[group_name]["masks"] = masks

    def _process_group(
        self,
        model: nn.Module,
        images: torch.Tensor,
        masks: torch.Tensor,
        embed_layer: int,
        is_target: bool,
        mask_inputs: bool,
    ) -> Dict[str, torch.Tensor]:
        """
        Runs a model (student or teacher) on a group of images and masks, returning DINO/iBOT tokens.

        Args:
            model (nn.Module): Model containing backbone and heads.
            images (torch.Tensor): Batch of images.
            masks (torch.Tensor): Batch of masks.
            embed_layer (int): Which embedding layer to use.
            is_target (bool): Whether this group is a target for loss computation.
            mask_inputs (bool): Whether to apply masks to the input.

        Returns:
            Dict[str, torch.Tensor]: Output tokens for DINO/iBOT heads and mask weights if applicable.
        """
        view_shape = images.shape[:-3]
        flat_images = rearrange(images, "... d w h -> (...) d w h")
        flat_masks = rearrange(masks, "... m -> (...) m") if masks is not None else None

        backbone_output = model.backbone(
            flat_images,
            embed_layer=embed_layer,
            masks=flat_masks if mask_inputs else None,
        )

        cls_tokens = backbone_output["clstoken"]
        dino_tokens_flat = model.dino_head(cls_tokens)

        dino_tokens = dino_tokens_flat.view(*view_shape, -1)

        output = {"dino": dino_tokens}

        if self.do_ibot and is_target:
            patch_tokens = backbone_output["patchtokens"]
            patch_tokens = rearrange(patch_tokens, "a p d -> (a p) d")
            masked_patch_tokens = patch_tokens[masks.view(-1)]

            ibot_head = model.ibot_head if self.ibot_separate_head else model.dino_head
            output["ibot"] = ibot_head(masked_patch_tokens)

            mask_weights = 1 / (masks.sum(-1).clamp(min=1.0))
            mask_weights = mask_weights.unsqueeze(-1).expand_as(masks)
            mask_weights = rearrange(mask_weights, "... -> (...)")
            output["mask_weights"] = mask_weights[masks.view(-1)]

        return output

    def _update_teacher_centers(
        self, uncentered_views: Dict[str, Dict[str, torch.Tensor]]
    ) -> None:
        """
        Updates the teacher's DINO and iBOT token centers for centering softmax outputs.

        Args:
            uncentered_views (Dict[str, Dict[str, torch.Tensor]]): Uncentered output tokens from teacher.
        """
        combined_dino_views = [
            rearrange(tokens, "... e -> (...) e")
            for tokens in uncentered_views["dino"].values()
        ]
        combined_dino_views = torch.cat(combined_dino_views, dim=0)
        self.dino_loss.update_center(combined_dino_views)

        if self.do_ibot:
            combined_ibot_views = [
                tokens for tokens in uncentered_views["ibot"].values()
            ]
            combined_ibot_views = torch.cat(combined_ibot_views, dim=0)
            self.ibot_patch_loss.update_center(combined_ibot_views)

    def _run_teacher_pass(
        self,
        collated_views: Dict[str, Any],
        teacher_temp: float,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Runs the teacher model on the collated batch and computes centered tokens for DINO/iBOT.

        Args:
            collated_views (Dict[str, Any]): Collated batch data.
            teacher_temp (float): Temperature for teacher softmax centering.

        Returns:
            Dict[str, Dict[str, torch.Tensor]]: Centered DINO and iBOT tokens for each group.
        """
        teacher_outputs = {"dino": {}, "ibot": {}}
        uncentered_views = {"dino": {}, "ibot": {}}
        with torch.no_grad():
            for group_name, view_info in collated_views.items():
                if not view_info.get("is_target", False):
                    continue

                group_output = self._process_group(
                    model=self.teacher,
                    images=view_info["images"],
                    masks=view_info["masks"],
                    embed_layer=collated_views[group_name]["embed_layer"],
                    is_target=True,
                    mask_inputs=False,
                )
                uncentered_views["dino"][group_name] = group_output["dino"]

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
                    uncentered_views["ibot"][group_name] = group_output["ibot"]
                    teacher_outputs["ibot"][group_name] = ibot_tokens_centered

        self._update_teacher_centers(uncentered_views)

        return teacher_outputs

    def _run_student_pass(
        self,
        collated_views: Dict[str, Any],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Runs the student model across all view groups."""
        student_outputs = {"dino": {}, "ibot": {}, "mask_weights": {}}
        for group_name, view_info in collated_views.items():
            group_output = self._process_group(
                model=self.student,
                images=view_info["images"],
                masks=view_info.get("masks", None),
                embed_layer=view_info["embed_layer"],
                is_target=view_info["is_target"],
                mask_inputs=True,
            )

            student_outputs["dino"][group_name] = group_output["dino"]

            if self.do_ibot and "ibot" in group_output:
                student_outputs["ibot"][group_name] = group_output["ibot"]
                student_outputs["mask_weights"][group_name] = group_output[
                    "mask_weights"
                ]
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

            targets = collated_views[group_name]["targets"]

            for i, target_group_name in enumerate(targets):

                view_dim = i + 1
                num_teacher_views = s_tokens.shape[view_dim]

                t_tokens = teacher_dino_tokens[target_group_name]

                for view_idx in range(num_teacher_views):

                    s_tokens_view = s_tokens.select(view_dim, view_idx).unsqueeze(
                        view_dim
                    )
                    t_tokens_view = t_tokens.select(view_dim, view_idx).unsqueeze(
                        view_dim
                    )

                    s_tokens_view = rearrange(s_tokens_view, "b ... d -> b (...) d")
                    t_tokens_view = rearrange(t_tokens_view, "b ... d -> b (...) d")

                    loss_term = self.dino_loss(s_tokens_view, t_tokens_view)
                    num_comparisons = s_tokens_view.shape[1] * t_tokens.shape[1]

                total_loss += loss_term
                total_terms += num_comparisons

        if total_terms == 0:
            logger.warning("No DINO comparisons were made. DINO loss is 0.")
            return torch.tensor(0.0)

        return total_loss / total_terms

    def _calculate_koleo_loss(
        self,
        student_global_dino_tokens: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Calculates the total KoLeo loss among student target views within a batch."""
        if not self.do_koleo:
            return torch.tensor(0.0)

        total_loss = torch.tensor(0.0)
        total_terms = 0

        for group_name, s_tokens in student_global_dino_tokens.items():

            flat_s_tokens = rearrange(s_tokens, "b ... d -> (...) b d")

            for i in range(flat_s_tokens.shape[0]):
                total_loss += self.koleo_loss(flat_s_tokens[i])
                total_terms += 1

        if total_terms == 0:
            return torch.tensor(0.0)

        return total_loss / total_terms

    def _calculate_ibot_loss(
        self,
        student_ibot_tokens: Dict[str, torch.Tensor],
        teacher_ibot_tokens: Dict[str, torch.Tensor],
        mask_weights: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Calculates the total iBOT loss across all student-teacher view pairs."""
        if not self.do_ibot:
            return torch.tensor(0.0)

        total_loss = torch.tensor(0.0)
        total_terms = 0

        for group_name, s_tokens in student_ibot_tokens.items():
            t_tokens = teacher_ibot_tokens[group_name]
            m_weights = mask_weights[group_name]
            loss_term = self.ibot_patch_loss(s_tokens, t_tokens, m_weights)

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
        self._prepare_inputs(collated_views)

        with self.autocast_ctx():
            teacher_outputs = self._run_teacher_pass(collated_views, teacher_temp)
            student_outputs = self._run_student_pass(collated_views)

        dino_loss = self._calculate_dino_loss(
            student_outputs["dino"], teacher_outputs["dino"], collated_views
        )
        ibot_loss = self._calculate_ibot_loss(
            student_outputs["ibot"],
            teacher_outputs["ibot"],
            student_outputs["mask_weights"],
        )
        student_target_dino_tokens = {
            group_name: tokens
            for group_name, tokens in student_outputs["dino"].items()
            if collated_views[group_name]["is_target"]
        }
        koleo_loss = self._calculate_koleo_loss(student_target_dino_tokens)

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

    def train(self, mode):
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

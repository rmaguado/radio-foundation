# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging
import time
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

        self.student = nn.ModuleDict()
        self.teacher = nn.ModuleDict()

        student_backbone, teacher_backbone = build_model_from_cfg(cfg)
        if student_backbone is None:
            raise ValueError(
                "student_backbone is None. Check build_model_from_cfg(cfg)."
            )
        self.student["backbone"] = student_backbone
        self.teacher["backbone"] = teacher_backbone

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
        self.student["dino_head"] = dino_head()
        self.teacher["dino_head"] = dino_head()

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
                self.student["ibot_head"] = ibot_head()
                self.teacher["ibot_head"] = ibot_head()

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
        process_start = time.time()

        # Time tensor reshaping
        reshape_start = time.time()
        view_shape = images.shape[:-3]
        flat_images = rearrange(images, "... d w h -> (...) d w h")
        flat_masks = rearrange(masks, "... m -> (...) m") if masks is not None else None
        reshape_time = time.time() - reshape_start
        logger.debug(f"Process group - Tensor reshaping: {reshape_time:.4f}s")

        # Time backbone forward pass
        backbone_start = time.time()
        backbone_output = model.backbone(
            flat_images,
            embed_layer=embed_layer,
            masks=flat_masks if mask_inputs else None,
        )
        backbone_time = time.time() - backbone_start
        logger.debug(
            f"Process group - Backbone forward pass: {backbone_time:.4f}s for images: {flat_images.shape} and masks: {masks.sum() if masks is not None else 0}"
        )

        # Time DINO head processing
        dino_start = time.time()
        cls_tokens = backbone_output["clstoken"]
        dino_tokens_flat = model.dino_head(cls_tokens)
        dino_tokens = dino_tokens_flat.view(*view_shape, -1)
        dino_time = time.time() - dino_start
        logger.debug(f"Process group - DINO head processing: {dino_time:.4f}s")

        output = {"dino": dino_tokens}

        # Time iBOT processing if needed
        if self.do_ibot and is_target:
            ibot_start = time.time()
            patch_tokens = backbone_output["patchtokens"]
            patch_tokens = rearrange(patch_tokens, "a p d -> (a p) d")
            masked_patch_tokens = patch_tokens[masks.view(-1)]

            ibot_head = model.ibot_head if self.ibot_separate_head else model.dino_head
            output["ibot"] = ibot_head(masked_patch_tokens)

            mask_weights = 1 / (masks.sum(-1).clamp(min=1.0))
            mask_weights = mask_weights.unsqueeze(-1).expand_as(masks)
            mask_weights = rearrange(mask_weights, "... -> (...)")
            output["mask_weights"] = mask_weights[masks.view(-1)]
            ibot_time = time.time() - ibot_start
            logger.debug(f"Process group - iBOT processing: {ibot_time:.4f}s")

        total_process_time = time.time() - process_start
        logger.debug(
            f"Process group - Total processing time: {total_process_time:.4f}s"
        )

        return output

    def _update_teacher_centers(
        self, uncentered_views: Dict[str, Dict[str, torch.Tensor]]
    ) -> None:
        """
        Updates the teacher's DINO and iBOT token centers for centering softmax outputs.

        Args:
            uncentered_views (Dict[str, Dict[str, torch.Tensor]]): Uncentered output tokens from teacher.
        """
        center_update_start = time.time()

        # Time DINO center update
        dino_center_start = time.time()
        combined_dino_views = [
            rearrange(tokens, "... e -> (...) e")
            for tokens in uncentered_views["dino"].values()
        ]
        combined_dino_views = torch.cat(combined_dino_views, dim=0)
        self.dino_loss.update_center(combined_dino_views)
        dino_center_time = time.time() - dino_center_start
        logger.debug(f"Center update - DINO center update: {dino_center_time:.4f}s")

        # Time iBOT center update
        if self.do_ibot:
            ibot_center_start = time.time()
            combined_ibot_views = [
                tokens for tokens in uncentered_views["ibot"].values()
            ]
            combined_ibot_views = torch.cat(combined_ibot_views, dim=0)
            self.ibot_patch_loss.update_center(combined_ibot_views)
            ibot_center_time = time.time() - ibot_center_start
            logger.debug(f"Center update - iBOT center update: {ibot_center_time:.4f}s")

        total_center_update_time = time.time() - center_update_start
        logger.debug(
            f"Center update - Total center update time: {total_center_update_time:.4f}s"
        )

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
        teacher_pass_start = time.time()
        teacher_outputs = {"dino": {}, "ibot": {}}
        uncentered_views = {"dino": {}, "ibot": {}}

        # Time teacher forward pass
        teacher_forward_start = time.time()
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
                dino_center_start = time.time()
                dino_tokens_centered = self.dino_loss.softmax_center_teacher(
                    group_output["dino"], teacher_temp
                )
                teacher_outputs["dino"][group_name] = dino_tokens_centered
                dino_center_time = time.time() - dino_center_start
                logger.debug(
                    f"Teacher pass - DINO centering for {group_name}: {dino_center_time:.4f}s"
                )

                # Center iBOT tokens
                if self.do_ibot:
                    ibot_center_start = time.time()
                    ibot_tokens_centered = self.ibot_patch_loss.softmax_center_teacher(
                        group_output["ibot"], teacher_temp
                    )
                    uncentered_views["ibot"][group_name] = group_output["ibot"]
                    teacher_outputs["ibot"][group_name] = ibot_tokens_centered
                    ibot_center_time = time.time() - ibot_center_start
                    logger.debug(
                        f"Teacher pass - iBOT centering for {group_name}: {ibot_center_time:.4f}s"
                    )

        teacher_forward_time = time.time() - teacher_forward_start
        logger.debug(
            f"Teacher pass - Total teacher forward pass: {teacher_forward_time:.4f}s"
        )

        # Time center updates
        center_update_start = time.time()
        self._update_teacher_centers(uncentered_views)
        center_update_time = time.time() - center_update_start
        logger.debug(f"Teacher pass - Center updates: {center_update_time:.4f}s")

        total_teacher_pass_time = time.time() - teacher_pass_start
        logger.debug(
            f"Teacher pass - Total teacher pass time: {total_teacher_pass_time:.4f}s"
        )

        return teacher_outputs

    def _run_student_pass(
        self,
        collated_views: Dict[str, Any],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Runs the student model across all view groups."""
        student_pass_start = time.time()
        student_outputs = {"dino": {}, "ibot": {}, "mask_weights": {}}

        # Time student forward pass
        student_forward_start = time.time()
        for group_name, view_info in collated_views.items():
            group_start = time.time()
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

            group_time = time.time() - group_start
            logger.debug(
                f"Student pass - Group {group_name} processing: {group_time:.4f}s"
            )

        student_forward_time = time.time() - student_forward_start
        logger.debug(
            f"Student pass - Total student forward pass: {student_forward_time:.4f}s"
        )

        total_student_pass_time = time.time() - student_pass_start
        logger.debug(
            f"Student pass - Total student pass time: {total_student_pass_time:.4f}s"
        )

        return student_outputs

    def _calculate_dino_loss(
        self,
        student_dino_tokens: Dict[str, torch.Tensor],
        teacher_dino_tokens: Dict[str, torch.Tensor],
        collated_views: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Calculates the total DINO loss, handling both self-comparison and hierarchical comparison.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        n_loss_terms = 0

        for group_name, s_tokens in student_dino_tokens.items():
            target_group_names = collated_views[group_name]["targets"]

            for target_idx, target_group_name in enumerate(target_group_names):

                t_tokens = teacher_dino_tokens[target_group_name]

                if target_group_name == group_name:

                    student_views_flat = rearrange(s_tokens, "b ... e -> b (...) e")
                    teacher_views_flat = rearrange(t_tokens, "b ... e -> b (...) e")

                    num_views = student_views_flat.shape[1]

                    for i in range(num_views):
                        student_view = student_views_flat[:, i, :].unsqueeze(1)
                        teacher_view = teacher_views_flat[:, i, :].unsqueeze(1)

                        loss = self.dino_loss(student_view, teacher_view)
                        total_loss += loss
                        n_loss_terms += 1

                else:
                    t_tokens_grouped = rearrange(t_tokens, "b ... d -> b (...) d")
                    num_teacher_lineages = t_tokens_grouped.shape[1]

                    B = s_tokens.shape[0]
                    H = s_tokens.shape[-1]

                    s_tokens_grouped = s_tokens.view(B, num_teacher_lineages, -1, H)

                    for i in range(num_teacher_lineages):
                        student_views_for_teacher_i = s_tokens_grouped[:, i, :, :]
                        teacher_view_i = t_tokens_grouped[:, i, :].unsqueeze(1)

                        loss = self.dino_loss(
                            student_views_for_teacher_i, teacher_view_i
                        )
                        total_loss += loss
                        n_loss_terms += 1

        if n_loss_terms == 0:
            return torch.tensor(0.0, device=self.device)

        return total_loss / n_loss_terms

    def _calculate_koleo_loss(
        self,
        student_global_dino_tokens: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Calculates the total KoLeo loss among student target views within a batch."""
        if not self.do_koleo:
            return torch.tensor(0.0, device=self.device)

        total_loss = torch.tensor(0.0, device=self.device)
        total_terms = 0

        for group_name, s_tokens in student_global_dino_tokens.items():

            flat_s_tokens = rearrange(s_tokens, "b ... d -> (...) b d")

            for i in range(flat_s_tokens.shape[0]):
                total_loss += self.koleo_loss(flat_s_tokens[i])
                total_terms += 1

        if total_terms == 0:
            return torch.tensor(0.0, device=self.device)

        return total_loss / total_terms

    def _calculate_ibot_loss(
        self,
        student_ibot_tokens: Dict[str, torch.Tensor],
        teacher_ibot_tokens: Dict[str, torch.Tensor],
        mask_weights: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Calculates the total iBOT loss across all student-teacher view pairs."""
        if not self.do_ibot:
            return torch.tensor(0.0, device=self.device)

        total_loss = torch.tensor(0.0, device=self.device)
        total_terms = 0

        for group_name, s_tokens in student_ibot_tokens.items():
            t_tokens = teacher_ibot_tokens[group_name]
            m_weights = mask_weights[group_name]
            if m_weights.numel() == 0:
                continue
            loss_term = self.ibot_patch_loss(s_tokens, t_tokens, m_weights)

            total_loss += loss_term
            total_terms += 1

        if total_terms == 0:
            return torch.tensor(0.0, device=self.device)

        return total_loss / total_terms

    def forward(
        self, collated_views: Dict[str, Any], teacher_temp: float
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Main forward pass for DINOv2 training.
        """
        start_time = time.time()

        # Step 1: Prepare inputs
        step_start = time.time()
        self._prepare_inputs(collated_views)
        prepare_time = time.time() - step_start
        logger.debug(f"Forward step 1 - Prepare inputs: {prepare_time:.4f}s")

        # Step 2: Run teacher and student passes
        step_start = time.time()
        with self.autocast_ctx():
            teacher_outputs = self._run_teacher_pass(collated_views, teacher_temp)
            student_outputs = self._run_student_pass(collated_views)
        model_pass_time = time.time() - step_start
        logger.debug(f"Forward step 2 - Model passes: {model_pass_time:.4f}s")

        # Step 3: Calculate DINO loss
        step_start = time.time()
        dino_loss = self._calculate_dino_loss(
            student_outputs["dino"], teacher_outputs["dino"], collated_views
        )
        dino_loss_time = time.time() - step_start
        logger.debug(f"Forward step 3 - DINO loss calculation: {dino_loss_time:.4f}s")

        # Step 4: Calculate iBOT loss
        step_start = time.time()
        ibot_loss = self._calculate_ibot_loss(
            student_outputs["ibot"],
            teacher_outputs["ibot"],
            student_outputs["mask_weights"],
        )
        ibot_loss_time = time.time() - step_start
        logger.debug(f"Forward step 4 - iBOT loss calculation: {ibot_loss_time:.4f}s")

        # Step 5: Calculate KoLeo loss
        step_start = time.time()
        student_target_dino_tokens = {
            group_name: tokens
            for group_name, tokens in student_outputs["dino"].items()
            if collated_views[group_name]["is_target"]
        }
        koleo_loss = self._calculate_koleo_loss(student_target_dino_tokens)
        koleo_loss_time = time.time() - step_start
        logger.debug(f"Forward step 5 - KoLeo loss calculation: {koleo_loss_time:.4f}s")

        # Step 6: Combine losses and create loss dict
        step_start = time.time()
        total_loss = (
            (self.dino_loss_weight * dino_loss)
            + (self.ibot_loss_weight * ibot_loss)
            + (self.koleo_loss_weight * koleo_loss)
        )
        loss_dict = {
            "dino_loss": dino_loss.detach(),
            "ibot_loss": ibot_loss.detach(),
            "koleo_loss": koleo_loss.detach(),
            "total_loss": total_loss.detach(),
        }
        combine_time = time.time() - step_start
        logger.debug(f"Forward step 6 - Combine losses: {combine_time:.4f}s")

        total_forward_time = time.time() - start_time
        logger.debug(f"Total forward pass time: {total_forward_time:.4f}s")

        return total_loss, loss_dict

    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                for ms, mt in zip(self.student[k].modules(), self.teacher[k].modules()):
                    student_param_list += list(ms.parameters())
                    teacher_param_list += list(mt.parameters())
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        logger.debug("Set SSLMetaArch to train.")

    def get_params_groups(self):

        all_params_groups = []
        for m in self.student.values():
            all_params_groups += get_params_groups_with_decay(
                m,
                lr_decay_rate=self.cfg.optim.layerwise_decay,
                patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
            )
        return all_params_groups

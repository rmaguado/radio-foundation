# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from encodings.punycode import T
from functools import partial
import logging
from typing import Dict, List, Tuple, Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from einops import rearrange

from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov2.models import build_model
from dinov2.layers import DINOHead
from dinov2.train.param_groups import get_params_groups_with_decay


logger = logging.getLogger("dinov2")


class SSLMetaArch(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.student = nn.ModuleDict()
        self.teacher = nn.ModuleDict()

        student_backbone, teacher_backbone = build_model(cfg)
        self.student["backbone"] = student_backbone
        self.teacher["backbone"] = teacher_backbone

        self.embed_dim = cfg.student.embed_dim
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

        for p in self.teacher.parameters():
            p.requires_grad = False

        crops_cfg = cfg.crops.crops_number
        self.num_global_2d = crops_cfg.global_2d
        self.num_local_3d = crops_cfg.local_3d
        self.num_local_2d = crops_cfg.local_2d
        self.global_view_multiple = crops_cfg.global_view_multiple

    def _prepare_inputs(self, collated_views: Dict[str, Any]) -> None:
        """
        Moves images and masks in the collated views dictionary to the correct device.

        Args:
            collated_views (Dict[str, Any]): Dictionary of collated batch data.
        """
        for group_name, view_info in collated_views.items():
            images = view_info["images"]
            masks = view_info.get("masks", None)

            images = images.cuda(non_blocking=True)
            collated_views[group_name]["images"] = images

            if masks is not None:
                masks = masks.cuda(non_blocking=True)
                collated_views[group_name]["masks"] = masks

    def _process_group(
        self,
        model: nn.Module,
        images: torch.Tensor,
        masks: torch.Tensor,
        embed_layer: int,
        is_target: bool,
        apply_mask: bool,
    ) -> Dict[str, torch.Tensor]:
        """
        Runs a model (student or teacher) on a group of images and masks, returning DINO/iBOT tokens.

        Args:
            model (nn.Module): Model containing backbone and heads.
            images (torch.Tensor): Batch of images.
            masks (torch.Tensor): Batch of masks.
            embed_layer (int): Which embedding layer to use.
            is_target (bool): Whether this group is a target for loss computation.
            apply_mask (bool): Whether to apply masks to the input.

        Returns:
            Dict[str, torch.Tensor]: Output tokens for DINO/iBOT heads and mask weights if applicable.
        """
        view_shape = images.shape[:-3]
        flat_images = rearrange(images, "... d w h -> (...) d w h")
        flat_masks = rearrange(masks, "... m -> (...) m") if masks is not None else None

        backbone_output = model.backbone(
            flat_images,
            embed_layer=embed_layer,
            masks=flat_masks if apply_mask else None,
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
                if group_name not in ["global_3d", "global_2d"]:
                    continue

                group_output = self._process_group(
                    model=self.teacher,
                    images=view_info["images"],
                    masks=view_info["masks"],
                    embed_layer=view_info["embed_layer"],
                    is_target=True,
                    apply_mask=False,
                )
                uncentered_views["dino"][group_name] = group_output["dino"]

                dino_tokens_centered = self.dino_loss.softmax_center_teacher(
                    group_output["dino"], teacher_temp
                )
                teacher_outputs["dino"][group_name] = dino_tokens_centered

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
                apply_mask=True,
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
    ) -> torch.Tensor:
        """
        Calculates the total DINO loss, handling both self-comparison and hierarchical comparison.
        """
        total_loss = torch.tensor(0.0).cuda()
        n_loss_terms = 0

        for group_name, group_student_tokens in student_dino_tokens.items():

            if group_name == "global_2d":
                s_tokens_grouped = rearrange(
                    group_student_tokens,
                    "b (v0 v1) e -> b v0 v1 e",
                    v0=self.num_global_2d,
                    v1=self.global_view_multiple,
                )
                t_tokens_grouped = rearrange(
                    teacher_dino_tokens["global_2d"],
                    "b (v0 v1) e -> b v0 v1 e",
                    v0=self.num_global_2d,
                    v1=self.global_view_multiple,
                )

                for v0 in range(self.num_global_2d):
                    s_tokens = s_tokens_grouped[:, v0, :, :]
                    t_tokens = t_tokens_grouped[:, v0, :, :]

                    for v1 in range(self.global_view_multiple):
                        s_tokens_i = torch.cat(
                            [t_tokens[:, :v1, :], t_tokens[:, v1 + 1 :, :]], dim=1
                        )
                        t_tokens_i = s_tokens[:, v1, :]

                        loss = self.dino_loss(s_tokens_i, t_tokens_i)
                        total_loss += loss
                        n_loss_terms += 1

            if group_name == "local_2d":
                s_tokens_grouped = rearrange(
                    group_student_tokens,
                    "b (v0 v1) e -> b v0 v1 e",
                    v0=self.num_global_2d,
                    v1=self.num_local_2d,
                )
                t_tokens_grouped = rearrange(
                    teacher_dino_tokens["global_2d"],
                    "b (v0 v1) e -> b v0 v1 e",
                    v0=self.num_global_2d,
                    v1=self.global_view_multiple,
                )

                for v0 in range(self.num_global_2d):
                    s_tokens = s_tokens_grouped[:, v0, :, :]
                    t_tokens = t_tokens_grouped[:, v0, :, :]

                    for v1 in range(self.global_view_multiple):
                        t_tokens_i = t_tokens[:, v1, :]

                        loss = self.dino_loss(s_tokens, t_tokens_i)
                        total_loss += loss
                        n_loss_terms += 1

                if "global_3d" in teacher_dino_tokens.keys():

                    s_tokens = group_student_tokens
                    t_tokens = teacher_dino_tokens["global_3d"]

                    for v0 in range(self.global_view_multiple):
                        t_tokens_i = t_tokens[:, v0, :]
                        loss = self.dino_loss(s_tokens, t_tokens_i)
                        total_loss += loss
                        n_loss_terms += 1

            if group_name == "global_3d":
                s_tokens = group_student_tokens
                t_tokens = teacher_dino_tokens["global_3d"]

                for v0 in range(self.global_view_multiple):
                    s_tokens_i = torch.cat(
                        [s_tokens[:, :v0, :], s_tokens[:, v0 + 1 :, :]], dim=1
                    )
                    t_tokens_i = t_tokens[:, v0, :]

                    loss = self.dino_loss(s_tokens_i, t_tokens_i)
                    total_loss += loss
                    n_loss_terms += 1

            if group_name == "local_3d":
                s_tokens = group_student_tokens
                t_tokens = teacher_dino_tokens["global_3d"]

                for v0 in range(self.global_view_multiple):
                    t_tokens_i = t_tokens[:, v0, :]

                    loss = self.dino_loss(s_tokens, t_tokens_i)
                    total_loss += loss
                    n_loss_terms += 1

        if n_loss_terms == 0:
            return torch.tensor(0.0).cuda()

        return total_loss / n_loss_terms

    def _calculate_koleo_loss(
        self,
        student_global_dino_tokens: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Calculates the total KoLeo loss among student target views within a batch."""
        if not self.do_koleo:
            return torch.tensor(0.0).cuda()

        total_loss = torch.tensor(0.0).cuda()
        total_terms = 0

        for group_name, s_tokens in student_global_dino_tokens.items():

            flat_s_tokens = rearrange(s_tokens, "b ... d -> (...) b d")

            for i in range(flat_s_tokens.shape[0]):
                total_loss += self.koleo_loss(flat_s_tokens[i])
                total_terms += 1

        if total_terms == 0:
            return torch.tensor(0.0).cuda()

        return total_loss / total_terms

    def _calculate_ibot_loss(
        self,
        student_ibot_tokens: Dict[str, torch.Tensor],
        teacher_ibot_tokens: Dict[str, torch.Tensor],
        mask_weights: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Calculates the total iBOT loss across all student-teacher view pairs."""
        if not self.do_ibot:
            return torch.tensor(0.0).cuda()

        total_loss = torch.tensor(0.0).cuda()
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
            return torch.tensor(0.0).cuda()

        return total_loss / total_terms

    def forward(
        self, collated_views: Dict[str, Any], teacher_temp: float
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Main forward pass for DINOv2 training.
        """
        self._prepare_inputs(collated_views)

        teacher_outputs = self._run_teacher_pass(collated_views, teacher_temp)
        student_outputs = self._run_student_pass(collated_views)

        dino_loss = self._calculate_dino_loss(
            student_outputs["dino"], teacher_outputs["dino"]
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
            "koleo_loss": koleo_loss.detach(),
        }

        return total_loss, loss_dict

    def update_teacher(self, m) -> None:
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                student_module = (
                    self.student[k].module
                    if hasattr(self.student[k], "module")
                    else self.student[k]
                )
                teacher_module = self.teacher[k]

                for ms, mt in zip(student_module.modules(), teacher_module.modules()):
                    student_param_list += list(ms.parameters())
                    teacher_param_list += list(mt.parameters())
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

        self.dino_loss.apply_center_update()
        self.ibot_patch_loss.apply_center_update()

    def train(self) -> None:
        super().train()
        self.teacher.eval()

    def get_params_groups(self) -> List[Any]:
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += get_params_groups_with_decay(
                m,
                lr_decay_rate=self.cfg.optim.layerwise_decay,
                patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
                num_layers=self.cfg.student.depth,
            )
        return all_params_groups

    def prepare_for_distributed_training(self, rank) -> None:

        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())

            self.student[k] = DDP(
                module=self.student[k],
                device_ids=[rank],
            )

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging
from typing import Dict, List, Tuple, Any, Optional

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from einops import rearrange

from dino.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dino.models import build_models
from dino.layers import DINOHead
from dino.train.param_groups import get_params_groups_with_decay


logger = logging.getLogger("dinov2")


class SSLMetaArch(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.student = nn.ModuleDict()
        self.teacher = nn.ModuleDict()

        student_backbone, teacher_backbone = build_models(cfg)
        self.student["backbone"] = student_backbone  # type: ignore
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

        self.num_crops_global = cfg.crops.num_crops_global

    def _process_group(
        self,
        model: nn.ModuleDict,
        images: torch.Tensor,
        masks: Optional[torch.Tensor],
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
        flat_images = rearrange(images, "b v d w h -> (b v) d w h")
        flat_masks = rearrange(masks, "b v m -> (b v) m") if masks is not None else None

        backbone_output = model["backbone"](
            flat_images,
            masks=flat_masks if apply_mask else None,
        )

        cls_tokens = backbone_output["clstoken"]
        dino_tokens_flat = model["dino_head"](cls_tokens)
        dino_tokens = dino_tokens_flat.view(*view_shape, -1)

        output = {"cls": dino_tokens, "cls_pre": cls_tokens}

        if self.do_ibot and is_target and masks:
            patch_tokens = backbone_output["patchtokens"]
            patch_tokens = rearrange(patch_tokens, "a p d -> (a p) d")
            masked_patch_tokens = patch_tokens[masks.view(-1)]

            ibot_head = (
                model["ibot_head"] if self.ibot_separate_head else model["dino_head"]
            )
            output["patch"] = ibot_head(masked_patch_tokens)

            mask_weights = 1 / (masks.sum(-1).clamp(min=1.0))
            mask_weights = mask_weights.unsqueeze(-1).expand_as(masks)
            mask_weights = rearrange(mask_weights, "... -> (...)")
            output["mask_weights"] = mask_weights[masks.view(-1)]

        return output

    def _update_teacher_centers(
        self, uncentered_views: Dict[str, torch.Tensor]
    ) -> None:
        """
        Updates the teacher's DINO and iBOT token centers for centering softmax outputs.
        """
        combined_dino_views = rearrange(uncentered_views["dino"], "b v e -> (b v) e")
        self.dino_loss.update_center(combined_dino_views)

        if self.do_ibot:
            combined_ibot_views = uncentered_views["ibot"]
            self.ibot_patch_loss.update_center(combined_ibot_views)

    def _run_teacher_pass(
        self,
        collated_views: Dict[str, Any],
        teacher_temp: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Runs the teacher model on the collated batch and computes centered tokens for DINO/iBOT.

        Args:
            collated_views (Dict[str, Any]): Collated batch data.
            teacher_temp (float): Temperature for teacher softmax centering.

        Returns:
            Dict[str, torch.Tensor]: Centered DINO and iBOT tokens.
        """
        teacher_outputs = {}
        uncentered_views = {}

        with torch.no_grad():

            group_output = self._process_group(
                model=self.teacher,
                images=collated_views["global"],
                masks=collated_views["masks"],
                is_target=True,
                apply_mask=False,
            )
            uncentered_views["dino"] = group_output["cls"]

            dino_tokens_centered = self.dino_loss.softmax_center_teacher(
                group_output["cls"], teacher_temp
            )
            teacher_outputs["global_cls"] = dino_tokens_centered

            if self.do_ibot:
                ibot_tokens_centered = self.ibot_patch_loss.softmax_center_teacher(
                    group_output["patch"], teacher_temp
                )
                uncentered_views["ibot"] = group_output["patch"]
                teacher_outputs["patch"] = ibot_tokens_centered

        self._update_teacher_centers(uncentered_views)

        return teacher_outputs

    def _run_student_pass(
        self,
        collated_views: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """Runs the student model across all view groups."""
        student_outputs = {}

        global_output = self._process_group(
            model=self.student,
            images=collated_views["global"],
            masks=collated_views["masks"],
            is_target=True,
            apply_mask=True,
        )

        student_outputs["global_cls"] = global_output["cls"]
        student_outputs["global_cls_pre"] = global_output["cls_pre"]

        if self.do_ibot:
            student_outputs["global_patch"] = global_output["patch"]
            student_outputs["mask_weights"] = global_output["mask_weights"]

        local_output = self._process_group(
            model=self.student,
            images=collated_views["local"],
            masks=None,
            is_target=True,
            apply_mask=False,
        )

        student_outputs["local_cls"] = local_output["cls"]

        return student_outputs

    def _calculate_dino_loss(
        self,
        student_output: Dict[str, torch.Tensor],
        teacher_output: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0).cuda()
        n_loss_terms = 0

        global_cls_student = student_output["global_cls"]
        local_cls_student = student_output["local_cls"]
        global_cls_teacher = teacher_output["global_cls"]

        for v0 in range(self.num_crops_global):
            t_tokens = global_cls_teacher[:, v0, :]

            s_tokens = torch.cat(
                [
                    global_cls_student[:, :v0, :],
                    global_cls_student[:, v0 + 1 :, :],
                    local_cls_student,
                ],
                dim=1,
            )

            loss = self.dino_loss(s_tokens, t_tokens)
            total_loss += loss
            n_loss_terms += 1

        if n_loss_terms == 0:
            return torch.tensor(0.0).cuda()

        return total_loss / n_loss_terms

    def _calculate_koleo_loss(
        self,
        student_output: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not self.do_koleo:
            return torch.tensor(0.0).cuda()

        total_loss = torch.tensor(0.0).cuda()
        total_terms = 0

        flat_s_tokens = rearrange(student_output["global_cls_pre"], "b v d -> v b d")

        for i in range(flat_s_tokens.shape[0]):
            total_loss += self.koleo_loss(flat_s_tokens[i])
            total_terms += 1

        if total_terms == 0:
            return torch.tensor(0.0).cuda()

        return total_loss / total_terms

    def _calculate_ibot_loss(
        self,
        student_output: Dict[str, torch.Tensor],
        teacher_output: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not self.do_ibot:
            return torch.tensor(0.0).cuda()

        student_ibot_tokens = student_output["global_patch"]
        teacher_ibot_tokens = teacher_output["global_path"]
        mask_weights = student_output["mask_weights"]

        return self.ibot_patch_loss(
            student_ibot_tokens, teacher_ibot_tokens, mask_weights
        )

    def forward(
        self, collated_views: Dict[str, Any], teacher_temp: float
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Main forward pass for DINOv2 training.
        """
        for k, v in collated_views.items():
            collated_views[k] = v.cuda(non_blocking=True)

        teacher_outputs = self._run_teacher_pass(collated_views, teacher_temp)
        student_outputs = self._run_student_pass(collated_views)

        dino_loss = self._calculate_dino_loss(student_outputs, teacher_outputs)

        ibot_loss = self._calculate_ibot_loss(
            student_outputs,
            teacher_outputs,
        )

        koleo_loss = self._calculate_koleo_loss(student_outputs)

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

                for ms, mt in zip(student_module.modules(), teacher_module.modules()):  # type: ignore
                    student_param_list += list(ms.parameters())
                    teacher_param_list += list(mt.parameters())
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

        self.dino_loss.apply_center_update()
        self.ibot_patch_loss.apply_center_update()

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

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

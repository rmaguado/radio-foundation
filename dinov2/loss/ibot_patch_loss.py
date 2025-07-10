# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed import ReduceOp

import logging

from dinov2.train.distributed import all_reduce


logger = logging.getLogger("dinov2")


class iBOTPatchLoss(nn.Module):
    def __init__(self, patch_out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, patch_out_dim))
        self.register_buffer("new_center", torch.zeros(1, patch_out_dim))
        self.update_counter = 0

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_patch_tokens, teacher_temp):
        return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)

    def forward(
        self,
        student_patch_tokens: torch.Tensor,
        teacher_patch_tokens: torch.Tensor,
        mask_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_patch_tokens: (N, D) tensor
        teacher_patch_tokens: (N, D) tensor
        mask_weights: (N,) tensor, weights for each sample
        """
        t = teacher_patch_tokens
        s = student_patch_tokens

        loss = -torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)
        loss = loss * mask_weights

        return loss.mean()

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens):
        self.new_center += teacher_patch_tokens.mean(dim=0)
        self.update_counter += 1

    @torch.no_grad()
    def apply_center_update(self):
        _t = self.new_center / self.update_counter
        all_reduce(_t, op=ReduceOp.AVG)

        self.new_center = torch.zeros_like(
            self.new_center, device=self.new_center.device
        )
        self.update_counter = 0

        self.center = self.center * self.center_momentum + _t * (
            1 - self.center_momentum
        )

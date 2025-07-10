# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed import ReduceOp

from einops import rearrange

from dinov2.train.distributed import all_reduce


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("new_center", torch.zeros(1, out_dim))
        self.update_counter = 0

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    def forward(
        self, student_outputs: torch.Tensor, teacher_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-entropy between softmax a single teacher target and possibly several student embeddings.
        """
        total_loss = torch.tensor(0.0, device=student_outputs.device)

        B, num_student_views, head_dim = student_outputs.shape

        student_flat = rearrange(student_outputs, "b v d -> (b v) d")
        lsm = F.log_softmax(student_flat / self.student_temp, dim=-1)
        lsm = rearrange(lsm, "(b v) d -> b v d", b=B, v=num_student_views)

        for s_idx in range(num_student_views):
            student_output = lsm[:, s_idx, :]

            loss_per_student_view = torch.sum(teacher_output * student_output, dim=-1)

            total_loss -= loss_per_student_view.mean()

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor) -> None:
        self.new_center += teacher_output.mean(dim=0)
        self.update_counter += 1

    @torch.no_grad()
    def apply_center_update(self) -> None:
        _t = self.new_center / self.update_counter
        all_reduce(_t, op=ReduceOp.AVG)

        self.new_center = torch.zeros_like(
            self.new_center, device=self.new_center.device
        )
        self.update_counter = 0

        self.center = self.center * self.center_momentum + _t * (
            1 - self.center_momentum
        )

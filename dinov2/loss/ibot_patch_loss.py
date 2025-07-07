# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

import logging


logger = logging.getLogger("dinov2")


try:
    from xformers.ops import cross_entropy

    def lossfunc(t, s, temp):
        s = s.float()
        t = t.float()
        if s.ndim == 2:
            return -cross_entropy(
                s.unsqueeze(0), t.unsqueeze(0), temp, bw_inplace=True
            ).squeeze(0)
        elif s.ndim == 3:
            return -cross_entropy(s, t, temp, bw_inplace=True)

except ImportError:

    def lossfunc(t, s, temp):
        return torch.sum(t * F.log_softmax(s / temp, dim=-1), dim=-1)


class iBOTPatchLoss(nn.Module):
    def __init__(self, patch_out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, 1, patch_out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_patch_tokens = None
        self.async_batch_center = None

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_patch_tokens, teacher_temp):
        self.apply_center_update()
        # WARNING: as self.center is a float32, everything gets casted to float32 afterwards

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
        t = student_patch_tokens
        s = teacher_patch_tokens

        loss = lossfunc(t, s, self.student_temp)
        loss = loss * mask_weights

        return -loss.mean()

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens):
        self.reduce_center_update(teacher_patch_tokens)

    @torch.no_grad()
    def reduce_center_update(self, teacher_patch_tokens):
        self.updated = False
        self.len_teacher_patch_tokens = len(teacher_patch_tokens)
        self.async_batch_center = torch.sum(
            teacher_patch_tokens.mean(1), dim=0, keepdim=True
        )
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()

            assert self.async_batch_center is not None
            assert self.len_teacher_patch_tokens is not None

            _t = self.async_batch_center / (self.len_teacher_patch_tokens * world_size)

            self.center = self.center * self.center_momentum + _t * (
                1 - self.center_momentum
            )

            self.updated = True

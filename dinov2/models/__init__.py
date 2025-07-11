# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from dinov2.models.vision_transformer import build_model


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(
        cfg.student, only_teacher=only_teacher, img_size=cfg.student.embed_layers[0].img_size
    )

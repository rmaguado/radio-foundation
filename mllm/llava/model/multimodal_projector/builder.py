import torch.nn as nn

from .attentional_pooler import (
    ClsAttentionalPoolProjector,
    PatchAttentionalPoolProjector,
)


def build_vision_projector(config, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "cls_attn_pool":
        mm_projector = ClsAttentionalPoolProjector(
            embed_dim=config.mm_vision_hidden_size,
            hidden_dim=config.mm_projector_hidden_size,
            output_dim=config.hidden_size,
            resample_tokens=config.image_tokens,
        )
        return mm_projector

    if projector_type == "patch_attn_pool":
        mm_projector = PatchAttentionalPoolProjector(
            embed_dim=config.mm_vision_hidden_size,
            hidden_dim=config.mm_projector_hidden_size,
            output_dim=config.hidden_size,
            axial_resample_tokens=config.image_tokens,
        )
        return mm_projector

    raise ValueError(f"Unknown projector type: {projector_type}")

import torch.nn as nn

from .attentional_pooler import (
    OneStepAttentionalPoolProjector,
    TwoStepAttentionalPoolProjector,
)


def build_vision_projector(config, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "one_step_attn":
        mm_projector = OneStepAttentionalPoolProjector(
            embed_dim=config.mm_vision_hidden_size,
            hidden_dim=config.mm_projector_hidden_size,
            output_dim=config.hidden_size,
            resample_tokens=config.image_tokens,
        )
        return mm_projector

    if projector_type == "two_step_attn":
        mm_projector = TwoStepAttentionalPoolProjector(
            embed_dim=config.mm_vision_hidden_size,
            hidden_dim=config.mm_projector_hidden_size,
            output_dim=config.hidden_size,
            axial_resample_tokens=config.image_tokens,
        )
        return mm_projector

    raise ValueError(f"Unknown projector type: {projector_type}")

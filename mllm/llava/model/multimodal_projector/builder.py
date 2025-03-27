import torch.nn as nn

from .attentional_pooler import AttentionalPoolProjector


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "attn_pool":
        mm_projector = AttentionalPoolProjector(
            embed_dim=config.mm_hidden_size,
            hidden_dim=config.hidden_size,
            axial_resample_tokens=config.image_tokens,
        )
        return mm_projector

    raise ValueError(f"Unknown projector type: {projector_type}")

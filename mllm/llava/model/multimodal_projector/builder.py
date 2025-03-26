import torch.nn as nn

from .attentional_pooler import AttentionalPoolProjector


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "attn_pool":
        mm_projector = AttentionalPoolProjector(
            embed_dim=config.mm_hidden_size,
            context_dim=config.mm_context_size,
        )
        return mm_projector

    raise ValueError(f"Unknown projector type: {projector_type}")

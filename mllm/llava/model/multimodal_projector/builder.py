import torch.nn as nn

from .attentional_pooler import AttentionalPoolProjector


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == "attn_pool":
        mm_projector = AttentionalPoolProjector(
            embed_dim=config.mm_hidden_size,
            context_dim=config.mm_context_size,
        )
        return mm_projector

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")

import torch
import torch.nn as nn

from einops import rearrange, repeat


class PerceiverResampler(nn.Module):
    def __init__(self, embed_dim, latent_dim, num_queries, num_heads=8, dropout=0.1):
        """
        Perceiver Resampler module. Resamples the input tokens by attending to them with a set of learnable queries.

        Args:
            embed_dim (int): Dimensionality of input tokens.
            latent_dim (int): Dimensionality of latent representations.
            num_queries (int): Number of queries.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super().__init__()

        self.num_queries = num_queries

        self.query = nn.Parameter(torch.randn(num_queries, latent_dim))
        self.key = nn.Linear(embed_dim, latent_dim)
        self.value = nn.Linear(embed_dim, latent_dim)

        self.attn = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=False
        )

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): Input tokens of shape (batch_size, seq_len, input_dim).
            mask (torch.Tensor): Attention mask of shape (batch_size, num_queries, seq_len).

        Returns:
            torch.Tensor: Resampled tokens of shape (batch_size, num_queries, latent_dim).
        """
        batch_size, seq_len, _ = x.size()

        q = repeat(self.query, "q h -> q b h", b=batch_size)
        k = rearrange(self.key(x), "b k h -> k b h")
        v = rearrange(self.value(x), "b k h -> k b h")

        attn_output, attn_weights = self.attn(q, k, v, key_padding_mask=mask)

        return rearrange(attn_output, "b k h -> k b h"), attn_weights


class FullScanPatchPredictor(nn.Module):
    def __init__(
        self, embed_dim, hidden_dim, num_labels, patch_resample_dim=64, dropout=0.5
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.patch_resample_dim = patch_resample_dim
        self.num_labels = num_labels

        self.token_resampler = PerceiverResampler(
            embed_dim=embed_dim, latent_dim=hidden_dim, num_queries=patch_resample_dim
        )
        self.axial_resampler = PerceiverResampler(
            embed_dim=hidden_dim, latent_dim=hidden_dim, num_queries=1
        )
        self.mlp = nn.Linear(hidden_dim, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, axial_dim, num_tokens, embed_dim = x.size()

        x = rearrange(x, "b a t e -> (b a) t e")
        x, attn_map = self.token_resampler(x)
        x = rearrange(x, "(b a) p h -> b (a p) h", a=axial_dim)
        attn_map = rearrange(attn_map, "(b a) h t -> b a h t", a=axial_dim)

        mask = repeat(mask, "b a -> b p a", p=self.patch_resample_dim)
        mask = rearrange(mask, "b p a -> b (p a)")

        x, _ = self.axial_resampler(x, mask=mask)
        x = rearrange(x, "b 1 h -> b h")
        x = self.dropout(x)

        return self.mlp(x), attn_map


class FullScanClassPredictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_labels, dropout=0.5):
        super().__init__()
        self.axial_resampler = PerceiverResampler(
            embed_dim=embed_dim, latent_dim=hidden_dim, num_queries=1
        )
        self.mlp = nn.Linear(hidden_dim, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, axial_dim, num_tokens, embed_dim = x.size()

        x = x.view(batch_size, axial_dim, embed_dim)

        x, attn_map = self.axial_resampler(x, mask=mask)
        x = rearrange(x, "b 1 h -> b h")
        attn_map = rearrange(attn_map, "b 1 h -> b h")

        x = self.dropout(x)

        return self.mlp(x), attn_map


class FullScanClassPatchPredictor(nn.Module):
    def __init__(
        self, embed_dim, hidden_dim, num_labels, patch_resample_dim=16, dropout=0.5
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.patch_resample_dim = patch_resample_dim
        self.num_labels = num_labels

        self.token_resampler = PerceiverResampler(
            embed_dim=embed_dim, latent_dim=hidden_dim, num_queries=patch_resample_dim
        )
        self.axial_resampler = PerceiverResampler(
            embed_dim=hidden_dim, latent_dim=hidden_dim, num_queries=1
        )
        self.class_resampler = PerceiverResampler(
            embed_dim=embed_dim, latent_dim=hidden_dim, num_queries=1
        )
        self.mlp = nn.Linear(hidden_dim * 2, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, axial_dim, num_tokens, embed_dim = x.size()

        cls_tokens = x[:, :, :1, :]
        patch_tokens = x[:, :, 1:, :]

        patch_tokens = rearrange(patch_tokens, "b a t e -> (b a) t e")
        patch_tokens, attn_map = self.token_resampler(patch_tokens)
        patch_tokens = rearrange(patch_tokens, "(b a) t h -> b (a t) h", a=axial_dim)
        attn_map = rearrange(attn_map, "(b a) h t -> b a h t", a=axial_dim)

        patch_mask = repeat(mask, "b a -> b p a", p=self.patch_resample_dim)
        patch_mask = rearrange(patch_mask, "b p a -> b (p a)")

        patch_embed, _ = self.axial_resampler(patch_tokens, mask=patch_mask)

        cls_tokens = rearrange(cls_tokens, "b a 1 e -> b a e")
        cls_embed, _ = self.class_resampler(cls_tokens, mask=mask)

        cls_patch_embed = torch.cat([cls_embed, patch_embed], dim=2)
        cls_patch_embed = rearrange(cls_patch_embed, "b 1 h -> b h")
        cls_patch_embed = self.dropout(cls_patch_embed)

        return self.mlp(cls_patch_embed), attn_map

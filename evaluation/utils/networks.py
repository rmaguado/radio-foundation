import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_labels, dropout=0.5):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x):
        return self.mlp(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


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

        self.attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): Input tokens of shape (batch_size, seq_len, input_dim).
            mask (torch.Tensor): Attention mask of shape (batch_size, num_queries, seq_len).

        Returns:
            torch.Tensor: Resampled tokens of shape (batch_size, num_queries, latent_dim).
        """
        batch_size, seq_len, _ = x.size()

        q = self.query.unsqueeze(1).repeat(1, batch_size, 1)
        k = self.key(x).transpose(0, 1)
        v = self.value(x).transpose(0, 1)

        attn_output, _ = self.attn(q, k, v, key_padding_mask=mask)

        return attn_output.transpose(0, 1)


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
        # x: (batch_size 8, axial_dim 20, num_tokens 1297, embed_dim 768)
        # mask: (batch_size 8, axial_dim 20)

        batch_size, axial_dim, num_tokens, embed_dim = x.size()

        x = x.view(batch_size * axial_dim, num_tokens, embed_dim)

        # x: (batch_size * axial_dim 160, num_tokens 1297, embed_dim 768)

        x = self.token_resampler(x)

        # x: (batch_size * axial_dim 160, patch_resample_dim 64, hidden_dim 768)

        x = x.reshape(batch_size, axial_dim, self.patch_resample_dim, self.hidden_dim)

        x = x.view(batch_size, axial_dim * self.patch_resample_dim, self.hidden_dim)

        # x: (batch_size 8, axial_dim * patch_resample_dim 1280, hidden_dim 768)

        mask = mask.unsqueeze(1).repeat(1, self.patch_resample_dim, 1)
        mask = mask.view(batch_size, axial_dim * self.patch_resample_dim)

        # mask: (batch_size 8, axial_dim * patch_resample_dim 1280)

        x = self.axial_resampler(x, mask=mask)
        x = self.dropout(x)

        x = self.mlp(x)

        return x.view(batch_size, self.num_labels)


class FullScanClassPredictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_labels, dropout=0.5):
        super().__init__()
        self.axial_resampler = PerceiverResampler(
            embed_dim=embed_dim, latent_dim=hidden_dim, num_queries=1
        )
        self.mlp = nn.Linear(hidden_dim, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (batch_size 8, axial_dim 20, num_tokens 1, embed_dim 768)
        # mask: (batch_size 8, axial_dim 20)
        batch_size, axial_dim, num_tokens, embed_dim = x.size()

        x = x.view(batch_size, axial_dim, embed_dim)
        # mask = mask.unsqueeze(2)

        # x: (batch_size 8, axial_dim 20, embed_dim 768)
        # mask: (batch_size 8, axial_dim 20)

        x = self.axial_resampler(x, mask=mask)
        x = self.dropout(x)

        # x: (batch_size 8, 1, hidden_dim 768)

        x = x.view(batch_size, -1)

        x = self.mlp(x)

        return x.view(batch_size, -1)


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

        patch_tokens = patch_tokens.view(
            batch_size * axial_dim, num_tokens - 1, embed_dim
        )
        patch_tokens = self.token_resampler(patch_tokens)
        patch_tokens = patch_tokens.reshape(
            batch_size, axial_dim, self.patch_resample_dim, self.hidden_dim
        )
        patch_tokens = patch_tokens.view(
            batch_size, axial_dim * self.patch_resample_dim, self.hidden_dim
        )

        patch_mask = mask.unsqueeze(1).repeat(1, self.patch_resample_dim, 1)
        patch_mask = patch_mask.view(batch_size, axial_dim * self.patch_resample_dim)

        patch_embed = self.axial_resampler(patch_tokens, mask=patch_mask)

        cls_tokens = cls_tokens.view(batch_size, axial_dim, embed_dim)
        cls_embed = self.class_resampler(cls_tokens, mask=mask)

        cls_patch_embed = torch.cat([cls_embed, patch_embed], dim=2)
        cls_patch_embed = self.dropout(cls_patch_embed)

        return self.mlp(cls_patch_embed).view(batch_size, self.num_labels)

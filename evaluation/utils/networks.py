import torch
import torch.nn as nn
import torch.nn.functional as F


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


class FullScanPredictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_labels, patch_resample_dim=64, dropout=0.5):
        super().__init__()

        self.patch_resample_dim = patch_resample_dim

        self.token_resampler = PerceiverResampler(
            embed_dim=embed_dim, latent_dim=hidden_dim, num_queries=patch_resample_dim
        )
        self.axial_resampler = PerceiverResampler(
            embed_dim=hidden_dim, latent_dim=hidden_dim, num_queries=1
        )
        self.mlp = nn.Linear(hidden_dim, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, axial_dim, num_tokens, embed_dim = x.size()
        x = x.view(axial_dim, num_tokens, embed_dim)
        x = self.token_resampler(x)

        axial_dim, resample_len, resample_dim = x.size()

        x = x.reshape(1, axial_dim * resample_len, resample_dim)
        x = self.axial_resampler(x)
        x = self.dropout(x)

        x = x.squeeze(0)
        return self.mlp(x)

import torch
import torch.nn as nn


class AveragePool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, masks=None):
        B, N, D = x.shape

        if masks is not None:
            return x.sum(dim=1) / masks.float().sum(dim=1).unsqueeze(1)
        return x.mean(dim=1)


class GatedPool(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())

    def forward(self, x, mask=None):
        B, N, D = x.shape
        gate_weights = self.gate_mlp(x)

        gated_features = x * gate_weights

        if mask is not None:
            gated_features = gated_features * mask.unsqueeze(-1)

        return torch.sum(gated_features, dim=1)


class SoftAttentionPool(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention_weights_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x, mask=None):
        B, N, D = x.shape
        raw_scores = self.attention_weights_mlp(x)

        if mask is not None:
            raw_scores = raw_scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        attention_weights = torch.softmax(raw_scores, dim=1)

        weighted_embeddings = x * attention_weights

        return torch.sum(weighted_embeddings, dim=1)


class AttentionPool(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, embed_dim))

        self.attention_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Linear(embed_dim, 1)
        )

        nn.init.xavier_uniform_(self.query)

    def forward(self, x, mask=None):
        B, N, D = x.shape

        query_expanded = self.query.expand(B, N, -1)

        combined = x + query_expanded

        raw_scores = self.attention_net(combined)

        if mask is not None:
            raw_scores = raw_scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        attention_weights = torch.softmax(raw_scores, dim=1)
        weighted_sum = torch.sum(x * attention_weights, dim=1)

        return weighted_sum

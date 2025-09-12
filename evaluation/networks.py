import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, pooler, embed_dim: int, num_classes: int):
        super().__init__()
        self.pooler = pooler
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x, mask=None):
        x = self.pooler(x, mask)
        logits = self.fc(x)
        return logits


class Regressor(nn.Module):
    def __init__(self, pooler, embed_dim: int):
        super().__init__()
        self.pooler = pooler
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x, mask=None):
        x = self.pooler(x, mask)
        prediction = self.fc(x)
        return prediction

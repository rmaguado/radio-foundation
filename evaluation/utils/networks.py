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


class AggregateClassTokens(nn.Module):
    def __init__(self, embed_dim=384 * 4, hidden_dim=1024, num_labels=1):
        super().__init__()
        self.linear = nn.Linear(embed_dim, hidden_dim, bias=True)
        self.attention_weights = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, class_tokens, mask=None):
        x = self.linear(class_tokens)
        x = self.dropout(x)

        attn_scores = self.attention_weights(x).squeeze(-1)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)

        attention_output = torch.sum(attn_weights * x, dim=1)
        attention_output = self.dropout(attention_output)

        return self.classifier(attention_output)

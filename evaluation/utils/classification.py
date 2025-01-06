from torchvision import transforms
import torch
import torch.nn as nn
from dinov2.data.samplers import InfiniteSampler


class ImageTransform:
    def __init__(self, img_size, mean, std):
        self.img_size = img_size
        self.resize = transforms.Resize((img_size, img_size))
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, image):
        resized = self.resize(image)
        normalized = self.normalize(resized)
        return normalized


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
    def __init__(
        self,
        embed_dim=384 * 4,
        num_labels=1,
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.classifier = nn.Linear(embed_dim, num_labels).to(device)

    def forward(self, class_tokens):
        cls_token = self.cls_token.expand(class_tokens.size(0), -1, -1)
        query = cls_token.permute(1, 0, 2)
        key = value = class_tokens.permute(1, 0, 2)
        attn_output, _ = self.multihead_attn(query, key, value)
        return self.classifier(attn_output.squeeze(0))


def _genertic_dataloader(dataset, is_infinite=False):
    """
    Generic Dataloader for classification with a single slice
    """

    def collate_fn(inputs):
        images = torch.stack([x[0] for x in inputs], dim=0)
        labels = torch.stack([x[1] for x in inputs], dim=0)

        return images, labels

    loader_kwargs = {
        "batch_size": 64,
        "num_workers": 10,
        "pin_memory": True,
        "collate_fn": collate_fn,
    }

    if is_infinite:
        sampler = InfiniteSampler(sample_count=len(dataset))
        loader = torch.utils.data.DataLoader(dataset, sampler=sampler, **loader_kwargs)
        return iter(loader)

    return torch.utils.data.DataLoader(dataset, **loader_kwargs)

from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.query = nn.Parameter(torch.randn(embed_dim))
        self.scale = embed_dim**0.5

        self.key_layer = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_layer = nn.Linear(embed_dim, embed_dim, bias=False)

        self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(self, class_tokens):
        keys = self.key_layer(class_tokens)
        values = self.value_layer(class_tokens)

        scores = torch.matmul(keys, self.query) / self.scale
        weights = F.softmax(scores, dim=0)

        attention_output = torch.sum(weights.unsqueeze(-1) * values, dim=0)

        return self.classifier(attention_output)


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

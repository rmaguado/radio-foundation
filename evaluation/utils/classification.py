from torchvision import transforms
import torch
import torch.nn as nn
from dinov2.data.samplers import InfiniteSampler


class ImageTransform:
    def __init__(self, img_size, mean, std):
        self.img_size = img_size
        self.resize = transforms.Resize((img_size, img_size))
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, image, target=None):
        resized = self.resize(image)
        normalized = self.normalize(resized)
        return normalized, target


class AggregateClassTokens(nn.Module):
    def __init__(
        self,
        feature_model,
        embed_dim=384 * 4,
        hidden_dim=2048,
        num_labels=2,
        device=torch.device("cpu"),
    ):
        super().__init__()


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
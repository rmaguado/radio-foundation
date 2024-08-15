import os
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

from functools import partial
from omegaconf import OmegaConf

from dinov2.data.samplers import InfiniteSampler
from dinov2.data.datasets import LidcIdri, NsclcRadiomics
from dinov2.models import build_model_from_cfg
from dinov2.eval.utils import ModelWithIntermediateLayers
from dinov2.utils.utils import load_pretrained_weights


class EvalTransform:
    def __init__(self, img_size, mean, std):
        self.img_size = img_size
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, image):
        np_resized = np.expand_dims(
            np.array(
                image.resize((self.img_size, self.img_size))
            ).astype(np.float32),
            axis=0
        )
        return self.normalize(
            torch.from_numpy(np_resized)
        )


class LinearClassifier(nn.Module):
    USE_N_BLOCKS = 4
    def __init__(self, embed_dim, hidden_dim, num_labels):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, x_tokens_list):
        intermediate_output = x_tokens_list[-LinearClassifier.USE_N_BLOCKS:]
        output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
        return self.mlp(output)


class DinoClassifier(nn.Module):
    def __init__(self, feature_model, embed_dim, hidden_dim, num_labels, device):
        super().__init__()
        
        self.feature_model = feature_model
        self.classifier = LinearClassifier(
            embed_dim, hidden_dim, num_labels
        ).to(device)
        
    def forward(self, x):
        features = self.feature_model(x)
        return self.classifier(features)

    
def get_config(path_to_run):
    path_to_config = os.path.join(path_to_run, "config.yaml")
    return OmegaConf.load(path_to_config)

def get_autocast_dtype(config):
    teacher_dtype_str = config.compute_precision.teacher.backbone.mixed_precision.param_dtype
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float

def load_model(path_to_run, checkpoint_name, device):
    path_to_checkpoint = os.path.join(path_to_run, "eval", checkpoint_name, "teacher_checkpoint.pth")
    
    config = get_config(path_to_run)
    
    model, _ = build_model_from_cfg(config, only_teacher=True)
    load_pretrained_weights(model, path_to_checkpoint, "teacher")
    model.eval()
    model.to(device)
    
    autocast_dtype = get_autocast_dtype(config)
    autocast_ctx = partial(torch.autocast, enabled=True, dtype=autocast_dtype, device_type="cuda")
    feature_model = ModelWithIntermediateLayers(model, 4, autocast_ctx)
    
    return feature_model, config
    
def get_norm(config):
    return config.augmentations.norm.mean, config.augmentations.norm.std

def get_accuracy_logits(outputs, targets):    
    predicted_labels = torch.argmax(outputs.detach().cpu(), dim=1)
    correct_predictions = (predicted_labels == targets)
    return correct_predictions.sum().item() / targets.size(0)

def get_dataloader(dataset, is_infinite=False):
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
        sampler = InfiniteSampler(
            sample_count=len(dataset),
            shuffle=False
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            **loader_kwargs
        )
        return iter(loader)
    
    return torch.utils.data.DataLoader(
            dataset,
            **loader_kwargs
        )
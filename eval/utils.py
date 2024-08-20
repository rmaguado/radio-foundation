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


class ImageTransform:
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
    
class ImageTargetTransform:
    def __init__(self, img_size, mean, std):
        self.img_size = img_size
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, image, target):
        image_resized = np.expand_dims(
            np.array(
                image.resize((self.img_size, self.img_size))
            ).astype(np.float32),
            axis=0
        )
        target_resized = np.expand_dims(
            np.array(
                target.resize((self.img_size, self.img_size))
            ).astype(np.float32),
            axis=0
        )
        return self.normalize(
            torch.from_numpy(image_resized)
        ), torch.from_numpy(target_resized)


class LinearClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_labels):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, x):
        return self.mlp(x)


class DinoClassifier(nn.Module):
    USE_N_BLOCKS = 4
    def __init__(
        self,
        feature_model,
        embed_dim=384*4,
        hidden_dim=2048,
        num_labels=2,
        device=torch.device("cpu")
    ):
        super().__init__()
        
        self.feature_model = feature_model
        self.classifier = LinearClassifier(
            embed_dim, hidden_dim, num_labels
        ).to(device)
        
    def forward(self, image):
        x_tokens_list = self.feature_model(image)
        intermediate_output = x_tokens_list[-DinoClassifier.USE_N_BLOCKS:]
        class_tokens = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
        return self.classifier(class_tokens)
    

class DinoSegmentation(nn.Module):
    USE_N_BLOCKS = 4
    def __init__(
        self,
        feature_model,
        embed_dim=384*4,
        hidden_dim=2048,
        num_labels=2, 
        device=torch.device("cpu")
    ):
        super().__init__()
        
        self.feature_model = feature_model
        self.classifier = LinearClassifier(
            embed_dim, hidden_dim, num_labels
        ).to(device)
        
    def forward(self, image):
        x_tokens_list = self.feature_model(image)
        intermediate_output = x_tokens_list[-DinoSegmentation.USE_N_BLOCKS:]
        patch_tokens = torch.cat([patch_token for patch_token, _ in features], dim=-1)
        return self.classifier(patch_tokens)


def show_mask(image_slice, mask_slice):
    color_image = np.stack([np.array(image_slice).astype(np.float32)] * 3, axis=-1)
    color_image[:,:,0] += np.array(mask_slice).astype(np.float32)
    return np.clip(color_image, 0, 1)

def binary_mask_to_patch_labels(mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert binary masks to patch-level labels.
    
    Parameters:
    - mask (torch.Tensor): Input binary mask of shape (batch_size, 1, img_size, img_size)
    - patch_size (int): Size of the patch (both width and height)
    
    Returns:
    - patch_labels (torch.Tensor): Patch-level labels of shape (batch_size, num_patches * num_patches. 1)
    """
    batch_size, _, img_size, _ = mask.shape
    assert img_size % patch_size == 0, "Image size must be divisible by patch size"
    
    num_patches = img_size // patch_size
    
    mask = mask.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patch_labels = mask.max(dim=-1)[0].max(dim=-1)[0].squeeze(1)
    
    return patch_labels.view(batch_size, -1)
    
def get_config(path_to_run):
    path_to_config = os.path.join(path_to_run, "config.yaml")
    return OmegaConf.load(path_to_config)

def get_autocast_dtype(cfg):
    teacher_dtype_str = cfg.compute_precision.teacher.backbone.mixed_precision.param_dtype
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
    
def get_norm(cfg):
    return cfg.norm.mean, cfg.norm.std

def multiclass_accuracy_logits(outputs, targets):    
    predicted_labels = torch.argmax(outputs.detach().cpu(), dim=1)
    correct_predictions = (predicted_labels == targets.cpu())
    return correct_predictions.sum().item() / targets.size(0)

def binary_accuracy_logits(outputs, targets):    
    predicted_labels = (outputs.detach().cpu() > 0).int()
    targets = targets.cpu().int()
    
    total_samples = targets.size(0)
    
    positives = (targets == 1).sum().item()
    negatives = (targets == 0).sum().item()
    
    true_pred_positives = ((predicted_labels == 1) & (targets == 1)).sum().item()
    true_pred_negatives = ((predicted_labels == 0) & (targets == 0)).sum().item()
    
    correct_predictions = (predicted_labels == targets).sum().item()
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    return accuracy, [positives, true_pred_positives, negatives, true_pred_negatives]

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
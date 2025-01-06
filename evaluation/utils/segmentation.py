import torch
import torch.nn as nn
import numpy as np

from torchvision import transforms

import random


class ImageTargetTransform:
    def __init__(self, img_size, mean, std):
        self.img_size = img_size
        self.resize = transforms.Resize((img_size, img_size))
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, image, target):
        image_resized = self.resize(image)
        target_resized = self.resize(target)
        return self.normalize(image_resized), target_resized


class DinoSegmentation(nn.Module):
    USE_N_BLOCKS = 4

    def __init__(
        self,
        feature_model,
        embed_dim=384 * 4,
        hidden_dim=2048,
        num_labels=2,
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.feature_model = feature_model
        self.classifier = LinearClassifier(embed_dim, hidden_dim, num_labels).to(device)

    def forward(self, image):
        x_tokens_list = self.feature_model(image)
        intermediate_output = x_tokens_list[-DinoSegmentation.USE_N_BLOCKS :]
        patch_tokens = torch.cat(
            [patch_token for patch_token, _ in intermediate_output], dim=-1
        )
        return self.classifier(patch_tokens)


def show_mask(image_slice, mask_slice):
    rescale_image = image_slice - np.min(image_slice)
    rescale_image /= np.max(rescale_image)

    color_image = np.stack([np.array(rescale_image).astype(np.float32)] * 3, axis=-1)
    color_image[:, :, 0] += np.array(mask_slice).astype(np.float32)
    return np.clip(color_image, 0, 1)


def binary_mask_to_patch_labels(mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert binary masks to patch-level labels.

    Parameters:
    - mask (torch.Tensor): Input binary mask of shape (batch_size, channels, img_size, img_size)
    - patch_size (int): Size of the patch (both width and height)

    Returns:
    - patch_labels (torch.Tensor): Patch-level labels of shape (batch_size, num_patches * num_patches. 1)
    """
    batch_size, _, img_size, _ = mask.shape
    assert img_size % patch_size == 0, "Image size must be divisible by patch size"

    mask = mask.max(dim=1)[0]

    mask = mask.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patch_labels = mask.max(dim=-1)[0].max(dim=-1)[0].squeeze(1)

    return patch_labels.view(batch_size, -1)


def sample_from_queues(
    positive_patch_cache,
    negative_patch_queue,
    patch_tokens,
    patch_labels,
    num_resample,
    device,
):
    """
    Sample tokens and labels from both caches and new tokens.
    For very unbalanced datasets.
    """

    if len(positive_patch_cache) < 2:
        return

    # number of new samples to add
    num_positive_samples = num_resample // 2
    num_hard_negative_samples = num_resample // 4
    num_new_negative_samples = num_resample // 4

    if len(negative_patch_queue) < num_hard_negative_samples:
        difference = num_hard_negative_samples - len(negative_patch_queue)
        num_hard_negative_samples = len(negative_patch_queue)
        num_new_negative_samples += difference

    # sample positive tokens
    positive_indices = random.sample(
        range(len(positive_patch_cache)), num_positive_samples
    )
    positive_tokens = torch.stack(
        [positive_patch_cache[idx].to(device) for idx in positive_indices], dim=0
    )
    positive_labels = torch.ones(len(positive_tokens), 1, device=device)

    # sample hard negative tokens
    if num_hard_negative_samples > 0:
        hard_negative_indices = random.sample(
            range(len(negative_patch_queue)), num_hard_negative_samples
        )
        hard_negative_tokens = torch.stack(
            [negative_patch_queue[idx].to(device) for idx in hard_negative_indices],
            dim=0,
        )

    # if there are not enough hard negative samples
    else:
        hard_negative_tokens = torch.tensor([]).to(device)
        hard_negative_indices = []

    # sample new negative tokens
    negative_indices = torch.where(patch_labels == 0)[0]
    new_negative_indices = random.sample(
        negative_indices.tolist(), num_new_negative_samples
    )
    new_negative_tokens = patch_tokens[new_negative_indices]

    # create negative labels
    negative_labels = torch.zeros(
        num_new_negative_samples + num_hard_negative_samples, 1
    ).to(device)

    # concatenate all samples
    resampled_tokens = torch.cat(
        (positive_tokens, hard_negative_tokens, new_negative_tokens), dim=0
    )
    resampled_labels = torch.cat((positive_labels, negative_labels), dim=0)

    return resampled_tokens, resampled_labels, hard_negative_indices


def process_batch(
    feature_model, inputs, targets, embed_dim, patch_size, device, use_n_blocks=4
):
    """Extract and process patch tokens and labels."""

    with torch.no_grad():
        x_tokens_list = feature_model(inputs.to(device))
    intermediate_output = x_tokens_list[-use_n_blocks:]
    patch_tokens = torch.cat(
        [patch_token for patch_token, _ in intermediate_output], dim=-1
    ).view(-1, embed_dim)

    patch_labels = binary_mask_to_patch_labels(targets.to(device), patch_size).view(-1)

    return patch_tokens, patch_labels


def update_difficult_negatives(
    classifier_output, patch_labels, patch_tokens, negative_patch_queue
):
    """Identify and store difficult negative patches."""

    patch_labels = patch_labels.view(-1)
    predictions = torch.sigmoid(classifier_output).squeeze()
    negative_indices = (patch_labels == 0).nonzero(as_tuple=True)[0]

    difficult_negatives = negative_indices[(predictions[negative_indices] > 0.5)]

    for idx in difficult_negatives:
        negative_patch_queue.append(patch_tokens[idx].detach().cpu())

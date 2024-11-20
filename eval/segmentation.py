import torch
import random

from eval.utils import binary_mask_to_patch_labels


def sample_from_queues(
    patch_embedding_cache,
    negative_patch_queue,
    patch_tokens,
    patch_labels,
    num_resample,
    device,
):
    """Sample tokens and labels from both caches and new tokens."""

    cache_sample_size = num_resample // 2
    neg_sample_size = num_resample // 4
    new_sample_size = num_resample // 4

    cache_indices = random.sample(range(len(patch_embedding_cache)), cache_sample_size)
    cache_tokens = torch.stack(
        [patch_embedding_cache[idx].to(device) for idx in cache_indices], dim=0
    )
    cache_labels = torch.ones(len(cache_tokens), 1, device=device)

    neg_indices = random.sample(range(len(negative_patch_queue)), neg_sample_size)
    neg_tokens = torch.stack(
        [negative_patch_queue[idx].to(device) for idx in neg_indices], dim=0
    )
    neg_labels = torch.zeros(len(neg_tokens), 1, device=device)

    new_indices = random.sample(range(len(patch_labels)), new_sample_size)
    new_tokens = patch_tokens[new_indices]
    new_labels = patch_labels[new_indices].view(-1, 1)

    resampled_tokens = torch.cat((cache_tokens, neg_tokens, new_tokens), dim=0)
    resampled_labels = torch.cat((cache_labels, neg_labels, new_labels), dim=0)

    return resampled_tokens, resampled_labels, neg_indices


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

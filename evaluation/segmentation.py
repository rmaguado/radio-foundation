import torch
import random

from evaluation.utils import binary_mask_to_patch_labels


def sample_from_queues(
    positive_patch_cache,
    negative_patch_queue,
    patch_tokens,
    patch_labels,
    num_resample,
    device,
):
    """Sample tokens and labels from both caches and new tokens."""

    num_positive_samples = num_resample // 2
    num_hard_negative_samples = num_resample // 4
    num_new_samples = num_resample // 4

    positive_indices = random.sample(
        range(len(positive_patch_cache)), num_positive_samples
    )
    positive_tokens = torch.stack(
        [positive_patch_cache[idx].to(device) for idx in positive_indices], dim=0
    )
    positive_labels = torch.ones(len(positive_tokens), 1, device=device)

    if len(negative_patch_queue) >= num_hard_negative_samples:
        hard_negative_indices = random.sample(
            range(len(negative_patch_queue)), num_hard_negative_samples
        )
        hard_negative_tokens = torch.stack(
            [negative_patch_queue[idx].to(device) for idx in hard_negative_indices],
            dim=0,
        )
    else:
        hard_negative_indices = list(range(len(negative_patch_queue)))
        hard_negative_tokens = torch.stack(
            [negative_patch_queue[idx].to(device) for idx in hard_negative_indices],
            dim=0,
        )
        remaining_neg_sample_size = num_hard_negative_samples - len(
            hard_negative_tokens
        )
        additional_indices = random.sample(
            range(len(patch_tokens)), remaining_neg_sample_size
        )
        additional_tokens = patch_tokens[additional_indices]
        hard_negative_tokens = torch.cat(
            (hard_negative_tokens, additional_tokens), dim=0
        )
        hard_negative_indices.extend(additional_indices)

    neg_labels = torch.zeros(len(hard_negative_tokens), 1, device=device)

    new_indices = random.sample(range(len(patch_labels)), num_new_samples)
    new_tokens = patch_tokens[new_indices]
    new_labels = patch_labels[new_indices].view(-1, 1)

    resampled_tokens = torch.cat(
        (positive_tokens, hard_negative_tokens, new_tokens), dim=0
    )
    resampled_labels = torch.cat((positive_labels, neg_labels, new_labels), dim=0)

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

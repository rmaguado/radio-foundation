from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
import numpy as np
import torch


def split_dataset(dataset, split_ratio):
    n = len(dataset)
    indices = np.random.permutation(n)
    split = int(n * split_ratio)
    train_dataset = DatasetSplit(dataset, indices[:split])
    val_dataset = DatasetSplit(dataset, indices[split:])

    return train_dataset, val_dataset


def cross_validation_split(dataset, n_splits):
    n = len(dataset)
    positive_indices = [i for i in range(n) if dataset.get_target(i)]
    negative_indices = [i for i in range(n) if not i in positive_indices]

    np.random.shuffle(positive_indices)
    np.random.shuffle(negative_indices)

    positive_folds = np.array_split(positive_indices, n_splits)
    negative_folds = np.array_split(negative_indices, n_splits)

    folds = []
    for i in range(n_splits):
        train_positive_indices = np.concatenate(
            positive_folds[:i] + positive_folds[i + 1 :]
        )
        train_negative_indices = np.concatenate(
            negative_folds[:i] + negative_folds[i + 1 :]
        )

        train_indices = np.concatenate([train_positive_indices, train_negative_indices])
        val_indices = np.concatenate([positive_folds[i], negative_folds[i]])

        train_dataset = DatasetSplit(dataset, train_indices)
        val_dataset = DatasetSplit(dataset, val_indices)

        folds.append((train_dataset, val_dataset))

    return folds


class CombinedDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cumulative_lengths = [
            sum(self.lengths[: i + 1]) for i in range(len(self.lengths))
        ]
        self.total_length = sum(self.lengths)

    def get_relative_item(self, index):
        for dataset_idx, cum_length in enumerate(self.cumulative_lengths):
            if index < cum_length:
                dataset_offset = index - (cum_length - self.lengths[dataset_idx])
                return self.datasets[dataset_idx], dataset_offset

    def get_target(self, index):
        dataset, dataset_index = self.get_relative_item(index)
        return dataset.get_target(dataset_index)

    def __getitem__(self, index):
        dataset, dataset_index = self.get_relative_item(index)
        return dataset[dataset_index]

    def __len__(self):
        return self.total_length


class DatasetSplit(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def get_target(self, index):
        return self.dataset.get_target(self.indices[index])

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


class BalancedSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = [self.dataset.get_target(i) for i in range(len(dataset))]

        self.positive_indices = [i for i, label in enumerate(self.labels) if label]
        self.negative_indices = [i for i, label in enumerate(self.labels) if not label]

        self.n_positive = len(self.positive_indices)
        self.n_negative = len(self.negative_indices)

        self.n_samples = min(self.n_positive, self.n_negative)

    def __iter__(self):
        positive_indices = np.random.choice(self.positive_indices, self.n_samples)
        negative_indices = np.random.choice(self.negative_indices, self.n_samples)

        combined_indices = np.concatenate([positive_indices, negative_indices])
        np.random.shuffle(combined_indices)

        return iter(combined_indices)

    def __len__(self):
        return 2 * self.n_samples


def collate_sequences(batch):
    embeddings, labels = zip(*batch)
    batch_size = len(embeddings)

    _, num_tokens, embed_dim = embeddings[0].shape

    max_axial_dim = max([embedding.shape[0] for embedding in embeddings])

    padded_embeddings = torch.zeros(batch_size, max_axial_dim, num_tokens, embed_dim)
    mask = torch.zeros(batch_size, max_axial_dim)

    for i, embedding in enumerate(embeddings):
        padded_embeddings[i, : embedding.shape[0]] = embedding
        mask[i, : embedding.shape[0]] = 1

    return padded_embeddings, mask, torch.tensor(labels)


def collate_sequences_clip_len(batch):
    """
    clip the number of axial slices around the center
    """
    max_len = 24
    embeddings, labels = zip(*batch)
    batch_size = len(embeddings)

    max_lengths = [embedding.shape[0] for embedding in embeddings]
    max_len = min(max(max_lengths), max_len)

    _, num_tokens, embed_dim = embeddings[0].shape

    center = [embedding.shape[0] // 2 for embedding in embeddings]
    start_idx = [max(0, center[i] - max_len // 2) for i in range(batch_size)]

    padded_embeddings = torch.zeros(batch_size, max_len, num_tokens, embed_dim)
    mask = torch.zeros(batch_size, max_len)

    for i, embedding in enumerate(embeddings):
        end_idx = min(start_idx[i] + max_len, embedding.shape[0])
        padded_embeddings[i, : end_idx - start_idx[i]] = embedding[
            start_idx[i] : end_idx
        ]
        mask[i, : end_idx - start_idx[i]] = 1

    return padded_embeddings, mask, torch.tensor(labels)

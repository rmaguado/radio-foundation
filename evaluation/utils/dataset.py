from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
import numpy as np


def split_dataset(dataset, split_ratio):
    n = len(dataset)
    indices = np.random.permutation(n)
    split = int(n * split_ratio)
    train_dataset = DatasetSplit(dataset, indices[:split])
    val_dataset = DatasetSplit(dataset, indices[split:])

    return train_dataset, val_dataset


class CombinedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cumulative_lengths = [
            sum(self.lengths[: i + 1]) for i in range(len(self.lengths))
        ]
        self.total_length = sum(self.lengths)

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        for dataset_idx, cum_length in enumerate(self.cumulative_lengths):
            if index < cum_length:
                dataset_offset = index - (cum_length - self.lengths[dataset_idx])
                return self.datasets[dataset_idx][dataset_offset]


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

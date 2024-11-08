import logging
from typing import Any, Tuple, List
import numpy as np
import torch

logger = logging.getLogger("dinov2")


class MultiDataset:

    def __init__(self, datasets: list) -> None:
        """
        Initializes a MultiDataset object for collating various dataset objects.

        Args:
            datasets (list): A list of datasets.
        """
        self.datasets = datasets
        self.cumulative_sizes = np.cumsum([len(d) for d in datasets])

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def get_dataset_sizes(self) -> List[int]:
        return [len(d) for d in self.datasets]

    def get_dataset_names(self) -> List[str]:
        return [d.dataset_name for d in self.datasets]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        """
        Retrieves the item at the given index from the multidataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, Any]: The retrieved item from the multidataset.
        """
        dataset_idx = self._find_dataset_idx(index)

        if dataset_idx > 0:
            dataset_index = index - self.cumulative_sizes[dataset_idx - 1]
        else:
            dataset_index = index

        return self.datasets[dataset_idx][dataset_index]

    def _find_dataset_idx(self, index: int) -> int:
        """
        Find the dataset index based on the global index.

        Args:
            index (int): The global index.

        Returns:
            int: The dataset index.
        """
        return np.searchsorted(self.cumulative_sizes, index, side="right")

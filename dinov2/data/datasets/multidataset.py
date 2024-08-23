import logging
import os
from typing import Callable, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger("dinov2")


class MultiDataset:
    def __init__(self, datasets: list) -> None:
        """
        Args:
            datasets (list): List of dataset instances.
        """
        self.datasets = datasets
        self.cumulative_sizes = np.cumsum([len(d) for d in datasets])
        
    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        dataset_idx = self._find_dataset_idx(index)
        
        if dataset_idx > 0:
            dataset_index = index - self.cumulative_sizes[dataset_idx - 1]
        else:
            dataset_index = index
        
        data = self.datasets[dataset_idx][dataset_index]
        
        return data
    
    def _find_dataset_idx(self, index: int) -> int:
        """
        Find the dataset index based on the global index.
        """
        return np.searchsorted(self.cumulative_sizes, index, side="right")

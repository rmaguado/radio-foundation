import os
import torch
from typing import Tuple, Any
import numpy as np
import pandas as pd


class COVID_19_NY_SBU:
    def __init__(
        self,
        run_name: str,
        checkpoint_name: str,
    ):
        self.embeddings_path = os.path.join(
            "evaluation/cache/COVID-19-NY-SBU_eval",
            run_name,
            checkpoint_name,
        )
        
        self.map_ids = [
            file.split(".npy")[0] for file in os.listdir(self.embeddings_path)
        ]

    def get_embeddings(self, map_id: str):
        return np.load(os.path.join(self.embeddings_path, f"{map_id}.npy"))

    def get_target(self, index):
        return True

    def __len__(self) -> int:
        return len(self.map_ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        map_id = self.map_ids[index]
        embeddings = self.get_embeddings(map_id)

        label = self.get_target(index)

        return torch.tensor(embeddings), label

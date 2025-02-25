import os
import torch
from typing import Tuple, Any
import numpy as np
import pandas as pd


class CT_RATE:
    def __init__(
        self,
        run_name: str,
        checkpoint_name: str,
        multiabnormality_labels_path: str,
        label: str,
        cache_format: str = "npy",
    ):
        self.embeddings_path = os.path.join(
            "evaluation/cache/CT-RATE_valid_eval",
            run_name,
            checkpoint_name,
        )
        self.df = pd.read_csv(multiabnormality_labels_path)
        self.label = label

        pattern = "_ct2rep.npy" if cache_format == "npy" else "_ct2rep.npz"
        self.get_embeddings = self.load_npy if cache_format == "npy" else self.load_npz

        self.map_ids = [
            file.split(pattern)[0] for file in os.listdir(self.embeddings_path)
        ]

    def load_npy(self, map_id: str):
        return np.load(os.path.join(self.embeddings_path, f"{map_id}_ct2rep.npy"))

    def load_npz(self, map_id: str):
        return np.expand_dims(
            np.load(os.path.join(self.embeddings_path, f"{map_id}_ct2rep.npz"))[
                "arr_0"
            ],
            0,
        )

    def get_target(self, index):
        VolumeName = f"{self.map_ids[index]}.nii.gz"
        return self.df[self.df["VolumeName"] == VolumeName][self.label].values[0]

    def __len__(self) -> int:
        return len(self.map_ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        map_id = self.map_ids[index]
        embeddings = self.get_embeddings(map_id)

        label = self.get_target(index)

        return torch.tensor(embeddings), label

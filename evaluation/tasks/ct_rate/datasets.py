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
    ):
        self.embeddings_path = os.path.join(
            "evaluation/cache/CT-RATE_train_eval",
            run_name,
            checkpoint_name,
        )
        self.df = pd.read_csv(multiabnormality_labels_path)
        self.label = label

        self.map_ids = [
            file.split(".npy")[0] for file in os.listdir(self.embeddings_path)
        ]

        self.targets = self.index_targets()

    def get_embeddings(self, map_id: str):
        embeddings = np.load(os.path.join(self.embeddings_path, f"{map_id}.npy"))
        return torch.from_numpy(embeddings).float()

    def index_targets(self):
        volume_names = [f"{map_id}.nii.gz" for map_id in self.map_ids]
        df_filtered = self.df[self.df["VolumeName"].isin(volume_names)]

        targets = df_filtered.set_index("VolumeName")[self.label].to_dict()
        targets = {map_id: targets[f"{map_id}.nii.gz"] for map_id in self.map_ids}
        return targets

    def get_target(self, index):
        return self.targets[self.map_ids[index]]

    def __len__(self) -> int:
        return len(self.map_ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        map_id = self.map_ids[index]
        embeddings = self.get_embeddings(map_id)

        label = self.get_target(index)

        return embeddings, label


class CT_RATE_Clip(CT_RATE):
    def __init__(
        self,
        run_name: str,
        checkpoint_name: str,
        multiabnormality_labels_path: str,
        label: str,
    ):
        self.embeddings_path = os.path.join(
            "evaluation/cache/CT-RATE_train_eval",
            run_name,
            checkpoint_name,
        )
        self.df = pd.read_csv(multiabnormality_labels_path)
        self.label = label

        self.map_ids = [
            file.split("_ct2rep.npz")[0] for file in os.listdir(self.embeddings_path)
        ]

        self.targets = self.index_targets()

    def get_embeddings(self, map_id: str):
        embeddings = np.load(
            os.path.join(self.embeddings_path, f"{map_id}_ct2rep.npz")
        )["arr"]
        embeddings = torch.from_numpy(embeddings).float()

        _, x, y, z, embed_dim = embeddings.shape

        embeddings = embeddings.view(x, y * z, embed_dim)

        return embeddings


class CT_RATE_Diffusion(CT_RATE):
    def __init__(
        self,
        run_name: str,
        checkpoint_name: str,
        multiabnormality_labels_path: str,
        label: str,
    ):
        self.embeddings_path = os.path.join(
            "evaluation/cache/CT-RATE_valid_eval",
            run_name,
            checkpoint_name,
        )
        self.df = pd.read_csv(multiabnormality_labels_path)
        self.label = label

        self.map_ids = [
            file.split("_ct2rep.npz")[0] for file in os.listdir(self.embeddings_path)
        ]

        self.targets = self.index_targets()

    def get_embeddings(self, map_id: str):
        return np.expand_dims(
            np.load(os.path.join(self.embeddings_path, f"{map_id}_ct2rep.npz"))[
                "arr_0"
            ],
            0,
        )

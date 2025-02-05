import os
import torch
from typing import Tuple, Any
import numpy as np
import pandas as pd


class DeepRDT_lung:
    def __init__(
        self,
        metadata_path: str,
        run_name: str,
        checkpoint_name: str,
        label: str,
    ):
        self.embeddings_path = os.path.join(
            "evaluation/cache/DeepRDT-lung_eval",
            run_name,
            checkpoint_name,
        )

        self.metadata = pd.read_csv(metadata_path)
        self.map_ids = [
            file.split(".npy")[0] for file in os.listdir(self.embeddings_path)
        ]

        if label == "response":
            self.target_fnc = self.get_response
        elif label == "sex":
            self.target_fnc = self.get_sex
        elif label == "tabaco":
            self.target_fnc = self.get_tabaco
        else:
            raise ValueError(f"Label {label} not recognized.")

    def get_embeddings(self, map_id: str):
        return np.load(os.path.join(self.embeddings_path, f"{map_id}.npy"))

    def get_metadata(self, map_id) -> np.ndarray:
        return self.metadata[self.metadata["MAPID"] == int(map_id)].iloc[0]

    def get_response(self, metadata_row: pd.Series) -> bool:
        response_text = metadata_row["respuesta"]
        # 1-Completa, 2-Parcial, 3-Estable, 4-Progresion
        return response_text in ["1-Completa", "2-Parcial"]

    def get_sex(self, metadata_row: pd.Series) -> bool:
        sexo_text = metadata_row["sexo"]
        assert sexo_text in ["Hombre", "Mujer"]
        return sexo_text == "Hombre"

    def get_tabaco(self, metadata_row: pd.Series) -> bool:
        tabaco_text = metadata_row["tabaco"]
        return tabaco_text != "Nunca"

    def get_target(self, index):
        map_id = self.map_ids[index]
        metadata = self.get_metadata(map_id)
        return self.target_fnc(metadata)

    def __len__(self) -> int:
        return len(self.map_ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        map_id = self.map_ids[index]
        embeddings = self.get_embeddings(map_id)

        label = self.get_target(index)

        return torch.tensor(embeddings), label

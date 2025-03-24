from typing import Tuple, Any
import pandas as pd
import torch


class DeepRDT_lung:
    def __init__(
        self,
        embeddings_provider: Any,
        metadata_path: str,
        label: str,
    ):
        self.df = pd.read_csv(metadata_path)
        self.label = label
        self.embeddings_provider = embeddings_provider
        self.map_ids = self.embeddings_provider.map_ids
        self.targets = self.index_targets()

    def index_targets(self):
        map_id_alias = {}
        for map_id in self.map_ids:
            if "_" in map_id:
                map_id_alias[map_id] = map_id.split("_")[0]

        df_filtered = self.df[self.df["MAPID"].isin(map_id_alias.values())]

        if self.label == "respuesta":
            target_fn = self.get_response
        elif self.label == "sexo":
            target_fn = self.get_sex
        elif self.label == "tabaco":
            target_fn = self.get_tabaco
        else:
            raise ValueError(f"Label {self.label} not implemented.")

        targets = df_filtered.set_index("MAPID")[self.label].to_dict()
        targets = {
            map_id: target_fn(targets[map_id_alias[map_id]]) for map_id in self.map_ids
        }
        return targets

    def get_response(self, entry: str) -> bool:
        return entry in [
            "1-Completa",
            "2-Parcial",
        ]  # 1-Completa, 2-Parcial, 3-Estable, 4-Progresion

    def get_sex(self, entry: str) -> bool:
        assert entry in ["Hombre", "Mujer"]
        return entry == "Hombre"

    def get_tabaco(self, entry: str) -> bool:
        return entry != "Nunca"

    def get_target(self, index):
        return self.targets[self.map_ids[index]]

    def __len__(self) -> int:
        return len(self.map_ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        map_id = self.map_ids[index]
        label = self.get_target(index)
        embeddings = self.embeddings_provider.get_embeddings(map_id)

        return embeddings, label

from typing import Tuple, Any
import pandas as pd
import torch


class FibrosisDataset:
    def __init__(
        self,
        embeddings_provider: Any,
        labels_path: str,
    ):
        self.df = pd.read_csv(labels_path)
        self.embeddings_provider = embeddings_provider
        self.map_ids = self.embeddings_provider.map_ids
        self.targets = self.index_targets()

        self.map_ids = list(self.targets.keys())

    def index_targets(self):
        df_filtered = self.df[self.df["mapid"].isin(self.map_ids)]
        targets = df_filtered.set_index("mapid")["label"].to_dict()

        return targets

    def get_target(self, index):
        return self.targets[self.map_ids[index]]

    def __len__(self) -> int:
        return len(self.map_ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        map_id = self.map_ids[index]
        label = self.get_target(index)
        embeddings = self.embeddings_provider.get_embeddings(map_id)

        return map_id, embeddings, label

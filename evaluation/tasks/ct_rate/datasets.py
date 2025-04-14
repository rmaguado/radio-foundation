from typing import Tuple, Any
import pandas as pd
import torch


class CT_RATE:
    def __init__(
        self,
        embeddings_provider: Any,
        multiabnormality_labels_path: str,
        label: str,
    ):
        self.df = pd.read_csv(multiabnormality_labels_path)
        self.label = label
        self.embeddings_provider = embeddings_provider
        self.map_ids = self.embeddings_provider.map_ids
        self.targets = self.index_targets()

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
        label = self.get_target(index)
        embeddings = self.embeddings_provider.get_embeddings(map_id)

        return map_id, embeddings, label
